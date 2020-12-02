from __future__ import print_function
import os
import sys
import numpy as np
import h5py
from scipy import ndimage
from sklearn import feature_extraction
import matplotlib.pyplot as plt
import cv2
import random


def image_convolve(image, kernel, conv_or_corr='conv', mode='reflect', cval=0.0, origin=0):
    assert image.ndim in [2, 3]
    assert conv_or_corr in ['conv', 'convolve', 'convolution', 'corr', 'correlate', 'correlation']
    if image.ndim == 2:
        if conv_or_corr in ['conv', 'convolve', 'convolution']:
            return ndimage.filters.convolve(image, weights=kernel, mode=mode, cval=cval, origin=origin)
        elif conv_or_corr in ['corr', 'correlate', 'correlation']:
            return ndimage.filters.correlate(image, weights=kernel, mode=mode, cval=cval, origin=origin)
    elif image.ndim == 3:
        channels = list()
        for idx in xrange(image.shape[2]):
            if conv_or_corr in ['conv', 'convolve', 'convolution']:
                channels.append(ndimage.filters.convolve(image[:, :, idx], weights=kernel, mode=mode, cval=cval, origin=origin))
            elif conv_or_corr in ['corr', 'correlate', 'correlation']:
                channels.append(ndimage.filters.correlate(image[:, :, idx], weights=kernel, mode=mode, cval=cval, origin=origin))
        return np.stack(channels, axis=2)



def conv_process(img, pad=0, stride=4, filter_size=4, dim_ordering='tf'):
    # print("Warning. You are using conv_process.")
    # print("Try to use scipy.ndimage.convolve for speed")
    # sys.exit(1)
    # suitable for ndarray data type
    assert img.ndim == 3
    assert stride == filter_size
    assert dim_ordering in ['th', 'tf']
    if dim_ordering == 'th':
        hy_rows = img.shape[1]
        wx_cols = img.shape[2]
        n_channel = img.shape[0]
    elif dim_ordering == 'tf':
        hy_rows = img.shape[0]
        wx_cols = img.shape[1]
        n_channel = img.shape[2]
    assert hy_rows % filter_size == 0
    assert wx_cols % filter_size == 0
    assert n_channel in [1]
    # range_y = range(0, hy_rows + 2 * pad - filter_size + 1, stride)
    # range_x = range(0, wx_cols + 2 * pad - filter_size + 1, stride)
    # range_y = range(0 + 4, hy_rows - 4 + 2 * pad - filter_size + 1, stride)
    range_y = range(0, hy_rows + 2 * pad - filter_size + 1, stride)
    # print(range_y)
    # range_x = range(0 + 4, wx_cols - 4 + 2 * pad - filter_size + 1, stride)
    range_x = range(0, wx_cols + 2 * pad - filter_size + 1, stride)
    # print(range_x)
    output_rows = len(range_y)
    output_cols = len(range_x)
    # print 'output size', output_rows, output_cols
    # new_dim = output_rows * output_cols
    # print('new size is: {} * {}'.format(output_rows, output_cols))
    # print('new dim is: {}'.format(new_dim))
    if dim_ordering == 'th':
        result = np.zeros((n_channel, output_rows, output_cols), dtype=np.single)
    elif dim_ordering == 'tf':
        result = np.zeros((output_rows, output_cols, n_channel), dtype=np.single)
    for index in range(n_channel):
        if dim_ordering == 'th':
            if pad > 0:
                new_data = np.zeros(
                    [hy_rows + 2 * pad, wx_cols + 2 * pad], dtype=np.single)
                new_data[pad:pad + hy_rows, pad:pad + wx_cols] = img[index, ...]
            else:
                new_data = img[index, ...]

            y_ind = 0
            for y in range_y:
                x_ind = 0
                for x in range_x:
                    # print new_data_mat[y:y + filter_size, x:x + filter_size]
                    # print x_ind, y_ind
                    result[index, y_ind, x_ind] = new_data[y:y + filter_size, x:x + filter_size].sum()
                    x_ind += 1
                y_ind += 1
        elif dim_ordering == 'tf':
            if pad > 0:
                new_data = np.zeros(
                    [hy_rows + 2 * pad, wx_cols + 2 * pad], dtype=np.single)
                new_data[pad:pad + hy_rows, pad:pad + wx_cols] = img[..., index]
            else:
                new_data = img[..., index]

            y_ind = 0
            for y in range_y:
                x_ind = 0
                for x in range_x:
                    # print new_data_mat[y:y + filter_size, x:x + filter_size]
                    # print x_ind, y_ind
                    result[y_ind, x_ind, index] = new_data[y:y + filter_size, x:x + filter_size].sum()
                    x_ind += 1
                y_ind += 1
    return result


def conv_process_batch(images, pad=0, stride=4, filter_size=4, dim_ordering='tf'):
    list_images = list()
    for idx in xrange(images.shape[0]):
        list_images.append(conv_process(images[idx], pad=pad, stride=stride, filter_size=filter_size, dim_ordering=dim_ordering))
    result = np.asarray(list_images)
    assert result.ndim == 4
    assert len(result) == len(images)
    return result

##################################################################################################################################
### to train the network using the whole image ###

def datagen_v3(h5file_view1, h5file_view2, h5file_view3, h5file_GP,
            batch_size=64, images_per_set=None,
            patches_per_image=1, patch_dim=(128, 128),
            density_scaler=1,
            image_shuffle=True, patch_shuffle=True,
            random_state=None, scale_number = 3):

    with h5py.File(h5file_view1[0], 'r') as f:
        images_i = f['images'].value
        density_maps_i = f['density_maps'].value
        dmp_h = density_maps_i.shape[1]
        dmp_w = density_maps_i.shape[2]
        img_h = images_i.shape[1]
        img_w = images_i.shape[2]

    with h5py.File(h5file_GP[0], 'r') as f:
        density_maps_i = f['density_maps'].value
        gdmp_h = density_maps_i.shape[1]
        gdmp_w = density_maps_i.shape[2]

    # read images and density maps
    density_maps1 = np.zeros([1, dmp_h, dmp_w, 1])
    images1 = np.zeros([1, img_h, img_w, 1])
    for i in h5file_view1:
        h5file_view1_i = i
        with h5py.File(h5file_view1_i, 'r') as f:
            images_i = f['images'].value
            density_maps_i = f['density_maps'].value
        density_maps1 = np.concatenate([density_maps1, density_maps_i], 0)
        images1 = np.concatenate([images1, images_i], 0)
    density_maps1 = density_maps1[1:, :, :, :]
    images1 = images1[1:, :, :, :]

    density_maps2 = np.zeros([1, dmp_h, dmp_w, 1])
    images2 = np.zeros([1, img_h, img_w, 1])
    for i in h5file_view2:
        h5file_view2_i = i
        with h5py.File(h5file_view2_i, 'r') as f:
            images_i = f['images'].value
            density_maps_i = f['density_maps'].value
        density_maps2 = np.concatenate([density_maps2, density_maps_i], 0)
        images2 = np.concatenate([images2, images_i], 0)
    density_maps2 = density_maps2[1:, :, :, :]
    images2 = images2[1:, :, :, :]

    density_maps3 = np.zeros([1, dmp_h, dmp_w, 1])
    images3 = np.zeros([1, img_h, img_w, 1])
    for i in h5file_view3:
        h5file_view3_i = i
        with h5py.File(h5file_view3_i, 'r') as f:
            images_i = f['images'].value
            density_maps_i = f['density_maps'].value
        density_maps3 = np.concatenate([density_maps3, density_maps_i], 0)
        images3 = np.concatenate([images3, images_i], 0)
    density_maps3 = density_maps3[1:, :, :, :]
    images3 = images3[1:, :, :, :]

    density_maps4 = np.zeros([1, gdmp_h, gdmp_w, 1])
    # images4 = np.asarray([])
    for i in h5file_GP:
        h5file_GP_i = i
        with h5py.File(h5file_GP_i, 'r') as f:
            #images_i = f['images'].value
            density_maps_i = f['density_maps'].value
        density_maps4 = np.concatenate([density_maps4, density_maps_i], 0)
        #images3 = np.concatenate([images3, images_i], 0)
    density_maps4 = density_maps4[1:, :, :, :]

    nb_images = len(images1)  # number of images

    # depth ratio maps
    # view 1
    scale_range = range(scale_number)
    scale_size = 2 * 4
    scale_zoom = 0.5 #0.75

    # view 1
    view1_image_depth = np.load('coords_correspondence/view_depth_image/'
                                'v1_1_depth_image_halfHeight.npz')
    view1_image_depth = view1_image_depth.f.arr_0
    h = view1_image_depth.shape[0]
    w = view1_image_depth.shape[1]
    h_scale = h / scale_size
    w_scale = w / scale_size
    view1_image_depth_resized = cv2.resize(view1_image_depth, (w_scale, h_scale))

    # set the center's scale of the image view1 as median of the all scales
    scale_center = np.median(scale_range)
    depth_center = view1_image_depth_resized[h_scale / 2, w_scale / 2]
    view1_image_depth_resized_log2 = np.log2(view1_image_depth_resized / depth_center)
    # view_image_depth_resized_log2 = view1_image_depth_resized_log2
    # plt.figure()
    # plt.imshow(view1_image_depth_resized_log2)
    # plt.show()
    view1_image_depth_resized_log2 = np.expand_dims(view1_image_depth_resized_log2, axis=2)
    view1_image_depth_resized_log2 = np.expand_dims(view1_image_depth_resized_log2, axis=0)

    # view 2
    view2_image_depth = np.load('coords_correspondence/view_depth_image/'
                                'v1_2_depth_image_halfHeight.npz')
    view2_image_depth = view2_image_depth.f.arr_0
    view2_image_depth_resized = cv2.resize(view2_image_depth, (w_scale, h_scale))
    view2_image_depth_resized_log2 = np.log2(view2_image_depth_resized / depth_center)
    # view_image_depth_resized_log2 = view2_image_depth_resized_log2
    # plt.figure()
    # plt.imshow(view2_image_depth_resized_log2)
    # plt.show()
    view2_image_depth_resized_log2 = np.expand_dims(view2_image_depth_resized_log2, axis=2)
    view2_image_depth_resized_log2 = np.expand_dims(view2_image_depth_resized_log2, axis=0)

    # view 3
    view3_image_depth = np.load('coords_correspondence/view_depth_image/'
                                'v1_3_depth_image_halfHeight.npz')
    view3_image_depth = view3_image_depth.f.arr_0
    view3_image_depth_resized = cv2.resize(view3_image_depth, (w_scale, h_scale))
    view3_image_depth_resized_log2 = np.log2(view3_image_depth_resized / depth_center) #/np.log2(4/3.0)
    # view_image_depth_resized_log2 = view3_image_depth_resized_log2
    # plt.figure()
    # plt.imshow(view3_image_depth_resized_log2)
    # plt.show()
    view3_image_depth_resized_log2 = np.expand_dims(view3_image_depth_resized_log2, axis=2)
    view3_image_depth_resized_log2 = np.expand_dims(view3_image_depth_resized_log2, axis=0)

    ############### random ###########################
    # nb_patch_used = 0
    nb_patch_used = 0
    np.random.seed(14)
    Nall = np.random.permutation(nb_images)
    patches_per_set = nb_images


    # codes and only codes inside while will be executed during every call of this generator
    while 1:
        # only extract more patches if the patches from the current image set (whole dataset so far) are exhausted
        if nb_patch_used + batch_size < patches_per_set + 1: #or (ind_image == 0 and nb_patch_used == 0):

            X_list = []
            Y_list = []

            n = Nall[nb_patch_used]

            img1_s0 = images1[n, :, :, 0]
            img1_s1 = cv2.resize(img1_s0, (img1_s0.shape[1]/2, img1_s0.shape[0]/2))
            img1_s2 = cv2.resize(img1_s1, (img1_s1.shape[1]/2, img1_s1.shape[0]/2))

            img1_s0 = np.expand_dims(img1_s0, axis=0)
            img1_s0 = np.expand_dims(img1_s0, axis=3)
            img1_s1 = np.expand_dims(img1_s1, axis=0)
            img1_s1 = np.expand_dims(img1_s1, axis=3)
            img1_s2 = np.expand_dims(img1_s2, axis=0)
            img1_s2 = np.expand_dims(img1_s2, axis=3)

            X_list.append(img1_s0)
            X_list.append(img1_s1)
            X_list.append(img1_s2)

            # img0 = np.zeros((img.shape[0], img.shape[1]))
            # img0[i*stepsize:i*stepsize+patchsize, j*stepsize:j*stepsize+patchsize] = img[i*stepsize:i*stepsize+patchsize, j*stepsize:j*stepsize+patchsize]
            # img_patch = img

            img2_s0 = images2[n, :, :, 0]
            img2_s1 = cv2.resize(img2_s0, (img2_s0.shape[1]/2, img2_s0.shape[0]/2))
            img2_s2 = cv2.resize(img2_s1, (img2_s1.shape[1]/2, img2_s1.shape[0]/2))

            img2_s0 = np.expand_dims(img2_s0, axis=0)
            img2_s0 = np.expand_dims(img2_s0, axis=3)
            img2_s1 = np.expand_dims(img2_s1, axis=0)
            img2_s1 = np.expand_dims(img2_s1, axis=3)
            img2_s2 = np.expand_dims(img2_s2, axis=0)
            img2_s2 = np.expand_dims(img2_s2, axis=3)

            X_list.append(img2_s0)
            X_list.append(img2_s1)
            X_list.append(img2_s2)

            #img0 = np.zeros((img2.shape[0], img2.shape[1]))
            #img0[i*stepsize:i*stepsize+patchsize, j*stepsize:j*stepsize+patchsize] = img2[i*stepsize:i*stepsize+patchsize, j*stepsize:j*stepsize+patchsize]
            # img_patch2 = img2
            
            img3_s0 = images3[n, :, :, 0]
            img3_s1 = cv2.resize(img3_s0, (img3_s0.shape[1]/2, img3_s0.shape[0]/2))
            img3_s2 = cv2.resize(img3_s1, (img3_s1.shape[1]/2, img3_s1.shape[0]/2))

            img3_s0 = np.expand_dims(img3_s0, axis=0)
            img3_s0 = np.expand_dims(img3_s0, axis=3)
            img3_s1 = np.expand_dims(img3_s1, axis=0)
            img3_s1 = np.expand_dims(img3_s1, axis=3)
            img3_s2 = np.expand_dims(img3_s2, axis=0)
            img3_s2 = np.expand_dims(img3_s2, axis=3)

            X_list.append(img3_s0)
            X_list.append(img3_s1)
            X_list.append(img3_s2)

            # depth ratio map
            X_list.append(view1_image_depth_resized_log2)
            X_list.append(view2_image_depth_resized_log2)
            X_list.append(view3_image_depth_resized_log2)


            ### get the dmap using projection ####
            dmap = density_maps1[n:n+1, :, :, :]
            dmap2 = density_maps2[n:n+1, :, :, :]
            dmap3 = density_maps3[n:n+1, :, :, :]
            dmap4 = density_maps4[n:n+1, :708, :608, :]
            #dmap_patch = dmap[i * stepsize:i * stepsize + patchsize, j * stepsize:j * stepsize + patchsize]

            dmap  = conv_process_batch(dmap,  pad=0, stride=4, filter_size=4)
            dmap2 = conv_process_batch(dmap2, pad=0, stride=4, filter_size=4)
            dmap3 = conv_process_batch(dmap3, pad=0, stride=4, filter_size=4)
            dmap4 = conv_process_batch(dmap4, pad=0, stride=4, filter_size=4)*1000

            # Y_list.append(dmap)
            # Y_list.append(dmap2)
            # Y_list.append(dmap3)
            Y_list.append(dmap4)

            # print(sum(dmap.flatten()))
            # print(sum(dmap2.flatten()))
            # print(sum(dmap3.flatten()))
            # print(sum(dmap4.flatten()))

            # yield ([img, img2, img3], [dmap4])
            yield (X_list, Y_list)
            X_list = []
            Y_list = []

            nb_patch_used = nb_patch_used + batch_size

        else:
            nb_patch_used = 0