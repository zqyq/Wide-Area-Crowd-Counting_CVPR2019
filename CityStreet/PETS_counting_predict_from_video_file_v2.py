from __future__ import print_function
import os
import sys
import re
import glob
import h5py
import skvideo.io
import numpy as np

np.set_printoptions(precision=6)
fixed_seed = 999
np.random.seed(fixed_seed)  # Set seed for reproducibility
import tensorflow as tf

tf.set_random_seed(fixed_seed)
import keras

print("Using keras {}".format(keras.__version__))
assert keras.__version__.startswith('2.')
from keras.optimizers import SGD
# from datagen import datagen

# from net_def_longer_fusionLayers import build_model_FCN_model_api as build_FCNN
from net_def_VGG import build_model_FCN_model_api as build_FCNN

# from net_multi_def_net_merge_270_135 import build_model_FCN_model_api as build_FCNN
# from net_multi_recept_field_def import build_model_FCN_model_api as build_FCNN
# from net_merge_540_270_7_7 import build_model_FCN_model_api as build_FCNN


# from net_multi_def_net_merge_540_270 import build_model_FCN_model_api as build_FCNN
# from sklearn import feature_extraction

import cv2
# assert cv2.__version__.startswith('3.4')

# import im_patch
import matplotlib.pyplot as plt

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# set enough GPU memory as needed(default, all GPU memory is used)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)



def rgb2gray(rgb):
    r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    return gray



def check_folder(dir_path):
    files = os.listdir(dir_path)
    xls_count = 0
    avi_count = 0
    mp4_count = 0
    # print("all {} files: {}".format(len(files), files))
    for file in files:
        if file.endswith('.xls'):
            xls_count += 1
            # print("found one label file: {}".format(file))
        elif file.endswith('.avi'):
            avi_count += 1
            # print("found one data file: {}".format(file))
        elif file.endswith('.mp4'):
            mp4_count += 1

        elif file.endswith(('.h5', '.hdf5')):
            print("found one label file (h5): {}".format(file))
        else:
            print("Unknown file type: {}".format(file))
            sys.exit(1)
    assert avi_count == xls_count
    if xls_count > 1 or avi_count > 1:
        print("more than one data file: {}".format(files))
        return False
    else:
        return True


def build_model_load_weights(image_dim, model_dir, model_name):
    opt = SGD(lr=0.001)
    model = build_FCNN(
        batch_size=1,
        patch_size=image_dim,
        optimizer=opt,
        output_ROI_mask=False,
    )
    # weight_files = os.listdir(model_dir)
    # weight_files.sort()
    # weight_files = weight_files[::-1]
    #
    # pattern = re.compile(model_name)               # 14-5.8780
    # #07-1067.8649.h5 29-5.8353 07-1067.8649 04-1067.5609-better.h5
    #
    # for file in weight_files:
    #     has_match = pattern.search(file)
    #     if has_match:
    #         break
    # best_model = has_match.group(0)
    # print(">>>> best_model: {}".format(best_model))
    # best_model = weight_files[-1]
    model.load_weights(filepath=os.path.join(model_dir, model_name), by_name=True)
    return model





def main(exp_name):
    # save_dir = os.path.join("/opt/visal/di/models/models_keras2/model_cell", exp_name)
    save_dir = os.path.join("models/", exp_name)
    print("save_dir: {}".format(save_dir))
    scaler_stability_factor = 1000

    stage = 'test_pred'
    print(stage)

    model_name = '07-30.2190.h5'
    counting_results_name = 'counting_results/' + model_name
    h5_savename = counting_results_name + '/counting_num_' + stage

    if os.path.isdir(counting_results_name):
        os.rmdir(counting_results_name)
    os.mkdir(counting_results_name)

    model = build_model_load_weights(image_dim=(380, 676, 3),
                                     model_dir='models/Street_all_1output_VGG/',
                                     model_name=model_name)  # projection/
    print(model_name)
    #################################################################

    # train
    train_path0 = '/opt/visal/home/qzhang364/Multi_view/Datasets/Street/'
    # train_path0 = '/media/zq/16A4D077A4D05B37/0_Ubuntu/datasets/Multi_view_shenshuipo/Street/'

    train_view1_1 = train_path0 + 'dmaps/train/Street_view1_dmap_10.h5'
    train_view2_1 = train_path0 + 'dmaps/train/Street_view2_dmap_10.h5'
    train_view3_1 = train_path0 + 'dmaps/train/Street_view3_dmap_10.h5'
    train_GP_1 = train_path0 + 'GP_dmaps/train/Street_groundplane_train_dmaps_10.h5'

    h5file_train_view1 = [train_view1_1]
    h5file_train_view2 = [train_view2_1]
    h5file_train_view3 = [train_view3_1]
    h5file_train_GP = [train_GP_1]

    # test
    # test_path0 = '/opt/visal/home/qzhang364/Multi_view/Datasets/PETS_2009/dmaps/'
    # test_path0 = '/media/zq/16A4D077A4D05B37/0_Ubuntu/datasets/PETS_2009/dmaps/'

    test_view1_1 = train_path0 + 'dmaps/test/Street_view1_dmap_10.h5'
    test_view2_1 = train_path0 + 'dmaps/test/Street_view2_dmap_10.h5'
    test_view3_1 = train_path0 + 'dmaps/test/Street_view3_dmap_10.h5'
    test_GP_1 = train_path0 + 'GP_dmaps/test/Street_groundplane_test_dmaps_10.h5'

    h5file_test_GP = [test_GP_1]

    h5file_test_view1 = [test_view1_1]
    h5file_test_view2 = [test_view2_1]
    h5file_test_view3 = [test_view3_1]

    if stage == 'train':
        h5file_view1 = h5file_train_view1
        h5file_view2 = h5file_train_view2
        h5file_view3 = h5file_train_view3
        h5file_GP = h5file_train_GP
    else:
        h5file_view1 = h5file_test_view1
        h5file_view2 = h5file_test_view2
        h5file_view3 = h5file_test_view3
        h5file_GP = h5file_test_GP

    # load the train or test data
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

    # predi

    count_view1_roi_GP = []
    count_view2_roi_GP = []
    count_view3_roi_GP = []
    count_gplane = []
    pred_dmap_gplane = []

    for j in range(1):

        # view 1
        density_maps1 = np.zeros([1, dmp_h, dmp_w, 1])
        images1 = np.zeros([1, img_h, img_w, 1])

        h5file_view1_i = h5file_view1[j]
        with h5py.File(h5file_view1_i, 'r') as f:
            images_i = f['images'].value
            density_maps_i = f['density_maps'].value
        density_maps1 = np.concatenate([density_maps1, density_maps_i], 0)
        images1 = np.concatenate([images1, images_i], 0)

        density_maps1 = density_maps1[1:, :, :, :]
        images1 = images1[1:, :, :, :]
        h1_test = images1
        h1_dmaps_test = density_maps1

        # view 2
        density_maps2 = np.zeros([1, dmp_h, dmp_w, 1])
        images2 = np.zeros([1, img_h, img_w, 1])

        h5file_view2_i = h5file_view2[j]
        with h5py.File(h5file_view2_i, 'r') as f:
            images_i = f['images'].value
            density_maps_i = f['density_maps'].value
        density_maps2 = np.concatenate([density_maps2, density_maps_i], 0)
        images2 = np.concatenate([images2, images_i], 0)

        density_maps2 = density_maps2[1:, :, :, :]
        images2 = images2[1:, :, :, :]
        h2_test = images2
        h2_dmaps_test = density_maps2

        # view 3
        density_maps3 = np.zeros([1, dmp_h, dmp_w, 1])
        images3 = np.zeros([1, img_h, img_w, 1])
        h5file_view3_i = h5file_view3[j]
        with h5py.File(h5file_view3_i, 'r') as f:
            images_i = f['images'].value
            density_maps_i = f['density_maps'].value
        density_maps3 = np.concatenate([density_maps3, density_maps_i], 0)
        images3 = np.concatenate([images3, images_i], 0)
        density_maps3 = density_maps3[1:, :, :, :]
        images3 = images3[1:, :, :, :]
        h3_test = images3
        h3_dmaps_test = density_maps3

        # GP
        density_maps4 = np.zeros([1, gdmp_h, gdmp_w, 1])
        # images4 = np.asarray([])
        h5file_GP_i = h5file_GP[j]
        with h5py.File(h5file_GP_i, 'r') as f:
            # images_i = f['images'].value
            density_maps_i = f['density_maps'].value
        density_maps4 = np.concatenate([density_maps4, density_maps_i], 0)
        # images3 = np.concatenate([images3, images_i], 0)
        density_maps4 = density_maps4[1:, :, :, :]
        h4_dmaps_test = density_maps4

        # depth ratio maps input
        # view 1
        scale_number = 3
        scale_range = range(scale_number)
        scale_size = 4
        # view 1
        view1_image_depth = np.load('coords_correspondence_Street/view_depth_image/'
                                    'v1_1_depth_image_avgHeight.npz')
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
        view1_image_depth_resized_log2 = np.expand_dims(view1_image_depth_resized_log2, axis=2)
        view1_image_depth_resized_log2 = np.expand_dims(view1_image_depth_resized_log2, axis=0)

        # view 2
        view2_image_depth = np.load('coords_correspondence_Street/view_depth_image/'
                                    'v1_2_depth_image_avgHeight.npz')
        view2_image_depth = view2_image_depth.f.arr_0
        view2_image_depth_resized = cv2.resize(view2_image_depth, (w_scale, h_scale))
        view2_image_depth_resized_log2 = np.log2(view2_image_depth_resized / depth_center)
        # view_image_depth_resized_log2 = view2_image_depth_resized_log2
        view2_image_depth_resized_log2 = np.expand_dims(view2_image_depth_resized_log2, axis=2)
        view2_image_depth_resized_log2 = np.expand_dims(view2_image_depth_resized_log2, axis=0)

        # view 3
        view3_image_depth = np.load('coords_correspondence_Street/view_depth_image/'
                                    'v1_3_depth_image_avgHeight.npz')
        view3_image_depth = view3_image_depth.f.arr_0
        view3_image_depth_resized = cv2.resize(view3_image_depth, (w_scale, h_scale))
        view3_image_depth_resized_log2 = np.log2(view3_image_depth_resized / depth_center)
        # view_image_depth_resized_log2 = view3_image_depth_resized_log2
        view3_image_depth_resized_log2 = np.expand_dims(view3_image_depth_resized_log2, axis=2)
        view3_image_depth_resized_log2 = np.expand_dims(view3_image_depth_resized_log2, axis=0)

        # GP mask:
        view1_gp_mask = np.load('coords_correspondence_Street/mask/view1_GP_mask.npz')
        view1_gp_mask = view1_gp_mask.f.arr_0
        view2_gp_mask = np.load('coords_correspondence_Street/mask/view2_GP_mask.npz')
        view2_gp_mask = view2_gp_mask.f.arr_0
        view3_gp_mask = np.load('coords_correspondence_Street/mask/view3_GP_mask.npz')
        view3_gp_mask = view3_gp_mask.f.arr_0

        view_gp_mask = view1_gp_mask + view2_gp_mask + view3_gp_mask
        view_gp_mask = np.clip(view_gp_mask, 0, 1)
        # plt.imshow(view_gp_mask)
        view1_gp_mask2 = cv2.resize(view1_gp_mask, (640, 768))
        view2_gp_mask2 = cv2.resize(view2_gp_mask, (640, 768))
        view3_gp_mask2 = cv2.resize(view3_gp_mask, (640, 768))
        view_gp_mask = cv2.resize(view_gp_mask, (640 / 4, 768 / 4))

        count1 = []
        count2 = []
        count3 = []

        list_pred = list()
        pred_dmaps_list = []
        image_dim = None
        f_count = 0

        #plt.figure()

        for i in range(h3_test.shape[0]):
            # i = 121

            frame1_s0 = h1_test[i:i + 1]
            frame1 = frame1_s0[0, :, :, 0]

            frame1_s1_0 = cv2.resize(frame1, (frame1.shape[1] / 2, frame1.shape[0] / 2))
            frame1_s1 = np.expand_dims(frame1_s1_0, axis=0)
            frame1_s1 = np.expand_dims(frame1_s1, axis=3)

            frame1_s2_0 = cv2.resize(frame1_s1_0, (frame1_s1_0.shape[1] / 2, frame1_s1_0.shape[0] / 2))
            frame1_s2 = np.expand_dims(frame1_s2_0, axis=0)
            frame1_s2 = np.expand_dims(frame1_s2, axis=3)

            frame2_s0 = h2_test[i:i + 1]
            frame2 = frame2_s0[0, :, :, 0]

            frame2_s1_0 = cv2.resize(frame2, (frame2.shape[1] / 2, frame2.shape[0] / 2))
            frame2_s1 = np.expand_dims(frame2_s1_0, axis=0)
            frame2_s1 = np.expand_dims(frame2_s1, axis=3)

            frame2_s2_0 = cv2.resize(frame2_s1_0, (frame2_s1_0.shape[1] / 2, frame2_s1_0.shape[0] / 2))
            frame2_s2 = np.expand_dims(frame2_s2_0, axis=0)
            frame2_s2 = np.expand_dims(frame2_s2, axis=3)

            frame3_s0 = h3_test[i:i + 1]
            frame3 = frame3_s0[0, :, :, 0]

            frame3_s1_0 = cv2.resize(frame3, (frame3.shape[1] / 2, frame3.shape[0] / 2))
            frame3_s1 = np.expand_dims(frame3_s1_0, axis=0)
            frame3_s1 = np.expand_dims(frame3_s1, axis=3)

            frame3_s2_0 = cv2.resize(frame3_s1_0, (frame3_s1_0.shape[1] / 2, frame3_s1_0.shape[0] / 2))
            frame3_s2 = np.expand_dims(frame3_s2_0, axis=0)
            frame3_s2 = np.expand_dims(frame3_s2, axis=3)

            dmap1 = h1_dmaps_test[i:i + 1]
            dmap2 = h2_dmaps_test[i:i + 1]
            dmap3 = h3_dmaps_test[i:i + 1]
            dmap4 = h4_dmaps_test[i:i + 1]

            count1_gt_i = np.sum(np.sum(dmap1[0, :, :, 0]))
            count2_gt_i = np.sum(np.sum(dmap2[0, :, :, 0]))
            count3_gt_i = np.sum(np.sum(dmap3[0, :, :, 0]))
            count4_gt_i = np.sum(np.sum(dmap4[0, :, :, 0]))

            # print(count1_gt_i-count4_gt_i)
            # # output layers:
            # from keras.models import Model
            # Xaug = [frame1_s0, frame1_s1, frame1_s2,
            #         frame2_s0, frame2_s1, frame2_s2,
            #         frame3_s0, frame3_s1, frame3_s2,
            #         view1_image_depth_resized_log2,
            #         view2_image_depth_resized_log2,
            #         view3_image_depth_resized_log2]
            # XX = model.input
            # YY_166 = model.layers[166].output
            # new_model_166 = Model(XX, YY_166)
            # Xresult_166 = new_model_166.predict(Xaug)
            #
            # YY_168 = model.layers[168].output
            # new_model_168 = Model(XX, YY_168)
            # Xresult_168 = new_model_168.predict(Xaug)
            #
            # YY_170 = model.layers[170].output
            # new_model_170 = Model(XX, YY_170)
            # Xresult_170 = new_model_170.predict(Xaug)
            #
            # fig = plt.figure()
            # fig.patch.set_facecolor('white')
            # plt.subplot(3, 3, 1)
            # plt.imshow(Xresult_166[0, :, :, 0], 'viridis')
            # plt.axis('off')
            # plt.colorbar()
            # plt.clim(0, 1)
            #
            # plt.subplot(3, 3, 2)
            # plt.imshow(Xresult_166[0, :, :, 1], 'viridis')
            # plt.axis('off')
            # plt.colorbar()
            # plt.clim(0, 1)
            #
            # plt.subplot(3, 3, 3)
            # plt.imshow(Xresult_166[0, :, :, 2], 'viridis')
            # plt.axis('off')
            # plt.colorbar()
            # plt.clim(0, 1)
            #
            #
            # plt.subplot(3, 3, 4)
            # plt.imshow(Xresult_168[0, :, :, 0], 'viridis')
            # plt.axis('off')
            # plt.colorbar()
            # plt.clim(0, 1)
            #
            # plt.subplot(3, 3, 5)
            # plt.imshow(Xresult_168[0, :, :, 1], 'viridis')
            # plt.axis('off')
            # plt.colorbar()
            # plt.clim(0, 1)
            #
            # plt.subplot(3, 3, 6)
            # plt.imshow(Xresult_168[0, :, :, 2], 'viridis')
            # plt.axis('off')
            # plt.colorbar()
            # plt.clim(0, 1)
            #
            #
            # plt.subplot(3, 3, 7)
            # plt.imshow(Xresult_170[0, :, :, 0], 'viridis')
            # plt.axis('off')
            # plt.colorbar()
            # plt.clim(0, 1)
            #
            # plt.subplot(3, 3, 8)
            # plt.imshow(Xresult_170[0, :, :, 1], 'viridis')
            # plt.axis('off')
            # plt.colorbar()
            # plt.clim(0, 1)
            #
            # plt.subplot(3, 3, 9)
            # plt.imshow(Xresult_170[0, :, :, 2], 'viridis')
            # plt.axis('off')
            # plt.colorbar()
            # plt.clim(0, 1)

            frame1_s0 = np.tile(frame1_s0, (1, 1, 1, 3))
            frame2_s0 = np.tile(frame2_s0, (1, 1, 1, 3))
            frame3_s0 = np.tile(frame3_s0, (1, 1, 1, 3))

            frame1_s1 = np.tile(frame1_s1, (1, 1, 1, 3))
            frame2_s1 = np.tile(frame2_s1, (1, 1, 1, 3))
            frame3_s1 = np.tile(frame3_s1, (1, 1, 1, 3))

            frame1_s2 = np.tile(frame1_s2, (1, 1, 1, 3))
            frame2_s2 = np.tile(frame2_s2, (1, 1, 1, 3))
            frame3_s2 = np.tile(frame3_s2, (1, 1, 1, 3))

            pred_dmap = model.predict_on_batch([frame1_s0, frame1_s1, frame1_s2,
                                                frame2_s0, frame2_s1, frame2_s2,
                                                frame3_s0, frame3_s1, frame3_s2,
                                                view1_image_depth_resized_log2,
                                                view2_image_depth_resized_log2,
                                                view3_image_depth_resized_log2])

            # count1_pred_i = np.sum(pred_dmap[0].flatten())/1000
            # count2_pred_i = np.sum(pred_dmap[1].flatten())/1000
            # count3_pred_i = np.sum(pred_dmap[2].flatten())/1000

            pred_dmap_0 = pred_dmap#[3]
            # pred_dmap_0 = pred_dmap_0*view_gp_mask
            count4_pred_i = np.sum(pred_dmap_0.flatten()) / 1000
            pred_dmap_gplane.append(pred_dmap_0)

            # fig = plt.figure()
            # fig.patch.set_facecolor('white')
            # plt.subplot(231)
            # plt.imshow(frame1_s0[0, :, :, 0], cmap='gray')
            # plt.axis('off')
            # plt.subplot(232)
            # plt.imshow(frame2_s0[0, :, :, 0], cmap='gray')
            # plt.axis('off')
            # plt.subplot(233)
            # plt.imshow(frame3_s0[0, :, :, 0], cmap='gray')
            # plt.axis('off')
            #
            # plt.subplot(234)
            # plt.imshow(dmap4[0, :, :, 0], cmap='viridis')
            # plt.axis('off')
            # plt.subplot(235)
            # plt.imshow(pred_dmap_0[0, :, :, 0] / 1000, cmap='viridis')
            # plt.axis('off')

            # count1.append([count1_gt_i, count1_pred_i])
            # count2.append([count2_gt_i, count2_pred_i])
            # count3.append([count3_gt_i, count3_pred_i])
            count_gplane.append([count1_gt_i, count2_gt_i, count3_gt_i, count4_gt_i, count4_pred_i])

            # roi GP pred
            count_view1_roi_GP_i = np.sum(np.sum(pred_dmap_0[0, :, :, 0] * view1_gp_mask)) / 1000
            count_view2_roi_GP_i = np.sum(np.sum(pred_dmap_0[0, :, :, 0] * view2_gp_mask)) / 1000
            count_view3_roi_GP_i = np.sum(np.sum(pred_dmap_0[0, :, :, 0] * view3_gp_mask)) / 1000
            # roi GP gt
            count_view1_roi_GP_gt_i = np.sum(np.sum(dmap4[0, :, :, 0] * view1_gp_mask2))
            count_view2_roi_GP_gt_i = np.sum(np.sum(dmap4[0, :, :, 0] * view2_gp_mask2))
            count_view3_roi_GP_gt_i = np.sum(np.sum(dmap4[0, :, :, 0] * view3_gp_mask2))
            count_view1_roi_GP.append([count_view1_roi_GP_gt_i, count_view1_roi_GP_i])
            count_view2_roi_GP.append([count_view2_roi_GP_gt_i, count_view2_roi_GP_i])
            count_view3_roi_GP.append([count_view3_roi_GP_gt_i, count_view3_roi_GP_i])


    # mae1 = np.asarray(count1)[:, 0] - np.asarray(count1)[:, 1]
    # mae1 = np.mean(np.abs(mae1))
    # print(mae1)
    # mae2 = np.asarray(count2)[:, 0] - np.asarray(count2)[:, 1]
    # mae2 = np.mean(np.abs(mae2))
    # print(mae2)
    # mae3 = np.asarray(count3)[:, 0] - np.asarray(count3)[:, 1]
    # mae3 = np.mean(np.abs(mae3))
    # print(mae3)

    # GP
    mae_GP = np.asarray(count_gplane)[:, 4] - np.asarray(count_gplane)[:, 3]
    mae_GP = np.mean(np.abs(mae_GP))
    print(mae_GP)

    # GP roi / GP
    dif_view1_GP = np.asarray(count_view1_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    mae_view1_GP = np.mean(np.abs(dif_view1_GP))
    print(mae_view1_GP)
    dif_view2_GP = np.asarray(count_view2_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    mae_view2_GP = np.mean(np.abs(dif_view2_GP))
    print(mae_view2_GP)
    dif_view3_GP = np.asarray(count_view3_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    mae_view3_GP = np.mean(np.abs(dif_view3_GP))
    print(mae_view3_GP)

    # GP roi / GP roi
    mae_view1_GProi = np.asarray(count_view1_roi_GP)[:, 1] - np.asarray(count_view1_roi_GP)[:, 0]
    mae_view1_GProi = np.mean(np.abs(mae_view1_GProi))
    print(mae_view1_GProi)
    mae_view2_GProi = np.asarray(count_view2_roi_GP)[:, 1] - np.asarray(count_view2_roi_GP)[:, 0]
    mae_view2_GProi = np.mean(np.abs(mae_view2_GProi))
    print(mae_view2_GProi)
    mae_view3_GProi = np.asarray(count_view3_roi_GP)[:, 1] - np.asarray(count_view3_roi_GP)[:, 0]
    mae_view3_GProi = np.mean(np.abs(mae_view3_GProi))
    print(mae_view3_GProi)

    # GP roi/view
    dif_view1 = np.asarray(count_view1_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 0]
    mae_view1 = np.mean(np.abs(dif_view1))
    print(mae_view1)
    dif_view2 = np.asarray(count_view2_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 1]
    mae_view2 = np.mean(np.abs(dif_view2))
    print(mae_view2)
    dif_view3 = np.asarray(count_view3_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 2]
    mae_view3 = np.mean(np.abs(dif_view3))
    print(mae_view3)

    with h5py.File(h5_savename, 'w') as f:
        f.create_dataset("count1_GProi", data=count_view1_roi_GP)
        f.create_dataset("count2_GProi", data=count_view2_roi_GP)
        f.create_dataset("count3_GProi", data=count_view3_roi_GP)
        f.create_dataset("count_gplane", data=count_gplane)
        f.create_dataset("mae_GP", data=mae_GP)

        f.create_dataset("pred_dmap_gplane", data=pred_dmap_gplane)

        f.create_dataset("mae_view1_GP", data=mae_view1_GP)
        f.create_dataset("mae_view2_GP", data=mae_view2_GP)
        f.create_dataset("mae_view3_GP", data=mae_view3_GP)

        f.create_dataset("mae_view1", data=mae_view1)
        f.create_dataset("mae_view2", data=mae_view2)
        f.create_dataset("mae_view3", data=mae_view3)

        f.create_dataset("mae_view1_GProi", data=mae_view1_GProi)
        f.create_dataset("mae_view2_GProi", data=mae_view2_GProi)
        f.create_dataset("mae_view3_GProi", data=mae_view3_GProi)




if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f", "--file",
        type=str,
        default='cell',
        action="store")
    parser.add_argument(
        "-e", "--exp_name",
        type=str,
        default='FCNN-sgd-whole-raw',
        action="store")
    args = parser.parse_args()
    main(exp_name=args.exp_name)
