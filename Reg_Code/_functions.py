import matplotlib.pyplot as plt
import numpy as np
import csv
import math
import glob
import re
import SimpleITK as sitk
import scipy.stats
import matplotlib as mpl

from pyimzml.ImzMLParser import ImzMLParser
from pyimzml.ImzMLParser import getionimage
from scipy import ndimage
from sklearn import manifold

def parsing(filename, peak_1, tol_1):

    slide = ImzMLParser(filename)
    for i, (x,y,z) in enumerate(slide.coordinates):
        slide.getspectrum(i)

    image_1 = getionimage(slide, peak_1, tol=tol_1)

    plt.figure(figsize=(16, 4))
    plt.imshow(image_1)
    plt.show()

    return slide, peak_1, tol_1


def tissue_crop_rot(X1, X2, Y1, Y2, degree_rotation, slide, peak_1, tol_1):

    MALDI_image_raw = getionimage(slide, peak_1, tol=tol_1)[Y1:Y2, X1:X2]
    MALDI_rot = ndimage.rotate(MALDI_image_raw, degree_rotation, reshape=True)

    plt.figure(figsize=(12, 5))
    ax=plt.subplot(1,2,1)
    plt.imshow(MALDI_image_raw)
    plt.colorbar()
    plt.axis('off')
    plt.title('Cropped image')
    ax=plt.subplot(1,2,2)
    plt.imshow(MALDI_rot)
    plt.colorbar()
    plt.axis('off')
    plt.title('Rotated image')
    plt.show()

    return MALDI_rot, degree_rotation, X1, X2, Y1, Y2


def hotspot_removal_conditions(quantile, MALDI_rot):
    Quantile_99 = np.quantile(MALDI_rot, quantile) 

    MALDI_image_hot = MALDI_rot.copy()
    MALDI_image_hot[MALDI_image_hot > Quantile_99] = Quantile_99

    row_hot, col_hot = MALDI_image_hot.shape

    MALDI_vector_raw = MALDI_rot.reshape(row_hot*col_hot)
    MALDI_vector_hot = MALDI_image_hot.reshape(row_hot*col_hot)

    plt.figure(figsize=(18, 10))
    ax = plt.subplot(2, 2, 1)
    plt.imshow(MALDI_rot)
    plt.axis('off')
    plt.title('Before hotspot removal')
    ax = plt.subplot(2, 2, 2)
    plt.boxplot(MALDI_vector_raw)
    plt.title('Before hotspot removal')
    ax = plt.subplot(2, 2, 3)
    plt.imshow(MALDI_image_hot)
    plt.axis('off')
    plt.title('After hotspot removal')
    ax = plt.subplot(2, 2, 4)
    plt.boxplot(MALDI_vector_hot)
    plt.title('After hotspot removal')
    plt.show()


def MALDI_rendering(signal_list, slide, degree_rotation, quantile, X1, X2, Y1, Y2):
    datafile = open(signal_list, 'r')
    reader = csv.reader(datafile)

    Ions = []
    Tolerance = []
    for row in reader:
        Ions.append(float(row[0]))
        Tolerance.append(float(row[1]))
    
    images_MALDI = []

    for i,t in zip(Ions, Tolerance):
        image = (getionimage(slide, i, tol=t)[Y1:Y2, X1:X2])
        im = ndimage.rotate(image, degree_rotation, reshape=True)
        Q_im = np.quantile(im, quantile)
        im[im > Q_im] = Q_im 
        images_MALDI.append(im)
    return images_MALDI, Ions


def images_plot(images, image_columns):
         
    # Images of the selected signals
    length = len(images)
    rows_graph = math.ceil(length/image_columns)

    plt.figure(figsize=(18, 28))
    for n,im in enumerate(images):
        ax = plt.subplot(rows_graph, image_columns, (n+1))
        plt.imshow(im)
        plt.axis('off')
        plt.title('Image {0}'.format(n+1))
    plt.show()


def tsne(BM, images):

    MALDI_BM = np.loadtxt(BM, delimiter=',')
    MALDI_BM[MALDI_BM == 255] = 1

    rows = images[0].shape[0]
    columns = images[0].shape[1]

    n_images = len(images)
    n_pixels = len(MALDI_BM[MALDI_BM == 1])

    vector_raw = np.zeros((n_images, n_pixels))

    flat_mask = MALDI_BM.reshape(rows*columns)  

    for n, image in enumerate(images):
        flat_raw = image.reshape(rows*columns)
        vector_raw[n,:] = flat_raw[flat_mask==1]

    vector_seg_raw = vector_raw.T
    
    tsne_seg = manifold.TSNE(n_components=1, random_state=0)
    vector_seg_tsne = tsne_seg.fit_transform(vector_seg_raw)

    vector_seg_tsne_image = np.zeros((rows*columns, 1))

    vector_seg_tsne_image[flat_mask==1] = vector_seg_tsne - np.min(vector_seg_tsne)

    vector_seg_tsn_norm = (vector_seg_tsne_image/np.max(vector_seg_tsne_image))*100 

    image_tsne = vector_seg_tsn_norm.reshape(rows, columns)

    plt.figure(figsize=(10, 5))
    plt.imshow(image_tsne)
    plt.axis('off')
    plt.title('t-SNE image')
    plt.show()

    return image_tsne, MALDI_BM


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split('(\d+)', text)]

def LA_process(foldername,extension):    
    files = glob.glob1(foldername,extension)
    files.sort(key=natural_keys)
    LA_images = []
    for file in files:
        LA_image = np.loadtxt(foldername + file, delimiter=',')
        LA_images.append(LA_image)
    return LA_images

def hotspot_removal(images, quantile, BM_LA_path):
    BM_LA = np.loadtxt(BM_LA_path, delimiter=',')
    BM_LA[BM_LA == 255] = 1
    images_hotspot = []
    for n,im in enumerate(images):
        Q_im = np.quantile(im, quantile) 
        im[im > Q_im] = Q_im
        im_BS = im * BM_LA
        images_hotspot.append(im_BS)
    return images_hotspot, BM_LA


def registration(fixed_image, moving_image, parameter_map):
    
    FixedImage = sitk.GetImageFromArray(fixed_image)
    MovingImage = sitk.GetImageFromArray(moving_image)
    
    ParameterMap = sitk.GetDefaultParameterMap(parameter_map)

    elastixImageFilter = sitk.ElastixImageFilter()
    elastixImageFilter.SetFixedImage(FixedImage)
    elastixImageFilter.SetMovingImage(MovingImage)
    elastixImageFilter.SetParameterMap(ParameterMap)
    elastixImageFilter.Execute()

    ResultImage = elastixImageFilter.GetResultImage()
    TransParameterMap = elastixImageFilter.GetTransformParameterMap()
    
    ResultArray = sitk.GetArrayFromImage(ResultImage)
    FixedArray = sitk.GetArrayFromImage(FixedImage)
    MovingArray = sitk.GetArrayFromImage(MovingImage)
    
    return ResultArray, FixedArray, MovingArray, TransParameterMap


def registration_plot(ResultArray, FixedArray, MovingArray):
    
    ResultArrayNorm = ResultArray/np.amax(ResultArray)
    FixedArrayNorm = FixedArray/np.amax(FixedArray)
    MovingArrayNorm = MovingArray/np.amax(MovingArray)
    
    plt.figure(figsize=(18,8))
    ax = plt.subplot(1, 5, 1)
    plt.imshow(FixedArray, cmap='Blues')
    plt.axis('off')
    plt.title('Fixed Image')
    ax = plt.subplot(1, 5, 2)
    plt.imshow(MovingArray, cmap='Reds')
    plt.axis('off')
    plt.title('Moving Image')
    ax = plt.subplot(1, 5, 3)
    plt.imshow(ResultArray, cmap='Reds')
    plt.axis('off')
    plt.title('Result Image')
    ax = plt.subplot(1, 5, 4)
    plt.imshow(FixedArrayNorm, cmap='Blues', alpha=0.8)
    plt.axis('off')
    plt.imshow(MovingArrayNorm, cmap='Reds', alpha=0.4)
    plt.axis('off')
    plt.title('Before optimization')
    ax = plt.subplot(1, 5, 5)
    plt.imshow(FixedArrayNorm, cmap='Blues', alpha=0.8)
    plt.axis('off')
    plt.imshow(ResultArrayNorm, cmap='Reds', alpha=0.4)
    plt.axis('off')
    plt.title ('After optimization')
    plt.show()


def transformation(images_processed_LA, MALDI_BM, TransParameterMap):
    transformixImageFilter = sitk.TransformixImageFilter()
    transformixImageFilter.SetTransformParameterMap(TransParameterMap)
    Trans_LA = []
    for images in images_processed_LA:
        transformixImageFilter.SetMovingImage(sitk.GetImageFromArray(images))
        transformixImageFilter.Execute()
        LA_transformed = (sitk.GetArrayFromImage(transformixImageFilter.GetResultImage()))*MALDI_BM
        Trans_LA.append(LA_transformed)
    return Trans_LA


def correlation_coefficient(images_MALDI, transformed_images_LA, MALDI_BM):

    rows = images_MALDI[0].shape[0]
    columns = images_MALDI[0].shape[1]

    images_all = transformed_images_LA + images_MALDI
    corr_matrix = np.zeros((len(images_all), len(images_all)))

    for index1,im1 in enumerate(images_all):
        im1_vector = im1[MALDI_BM==1]
        for index2,im2 in enumerate(images_all):
            im2_vector = im2[MALDI_BM==1]
            corr, p = scipy.stats.pearsonr(im1_vector, im2_vector)
            corr_matrix[index1, index2] = corr
    return corr_matrix


def corr_plot(corr_matrix, Ions):
    metal_labels = ['Metal {0}'.format(x+1) for x in range(corr_matrix.shape[0]-len(Ions))]
    Ions_all = metal_labels + Ions
    
    fig = plt.figure(figsize=(18, 15))
    ax1 = plt.gca()
    cmap = plt.get_cmap('RdBu_r')
    ax1.imshow(corr_matrix, cmap=cmap, origin='upper', vmin=-1, vmax=1)
    ax1.set_xticks(range(len(Ions_all)))
    ax1.set_xticklabels(Ions_all, rotation=90)
    ax1.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, labelsize=16)
    ax1.set_aspect('equal')
    ax1.set_yticks(range(len(Ions_all)))
    ax1.set_yticklabels(Ions_all)

    norm = mpl.colors.Normalize(vmin=-1,vmax=1)
    norm = mpl.colors.Normalize(vmin=np.amin(corr_matrix),vmax=np.amax(corr_matrix))
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    plt.colorbar(sm, pad=0.01, aspect=30) 
    plt.show()