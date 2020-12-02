from __future__ import print_function
import os
import sys
import re
import glob
import h5py
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

#from net_def_longer_fusionLayers import build_model_FCN_model_api as build_FCNN
# from net_def_getWeights import build_model_FCN_model_api as build_FCNN
from net_def import build_model_FCN_model_api as build_FCNN

# from net_multi_def_net_merge_270_135 import build_model_FCN_model_api as build_FCNN
#from net_multi_recept_field_def import build_model_FCN_model_api as build_FCNN
#from net_merge_540_270_7_7 import build_model_FCN_model_api as build_FCNN


#from net_multi_def_net_merge_540_270 import build_model_FCN_model_api as build_FCNN
# from sklearn import feature_extraction

import cv2
# assert cv2.__version__.startswith('3.4')

#import im_patch
# import matplotlib.pyplot as plt

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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
    weight_files = os.listdir(model_dir)
    weight_files.sort()
    weight_files = weight_files[::-1]

    pattern = re.compile(model_name)               # 14-5.8780
    #07-1067.8649.h5 29-5.8353 07-1067.8649 04-1067.5609-better.h5

    for file in weight_files:
        has_match = pattern.search(file)
        if has_match:
            break
    best_model = has_match.group(0)
    print(">>>> best_model: {}".format(best_model))
    # best_model = weight_files[-1]
    model.load_weights(filepath=os.path.join(model_dir, model_name), by_name=True)

    # model.save_weights('load_weights/feature_extraction/MVMS_feature_extraction.h5')

    return model


def main(exp_name):
    # data_root = "/opt/visal/di/data/Cell_Tracking"
    # data_root = "/home/zq/codes/MTR/MTR_counting/videos_predict/raw/" # rawd
    # data_root = "/home/zq/codes/multi-view/TaskA_late_fusion_whole_rawData/PETS09_2/Data/"

    # data_root = "/media/zq/16A4D077A4D05B37/0_Ubuntu/datasets/PETS_2009/data/PETS_data_10/"
    # data_root = "/media/zq/16A4D077A4D05B37/0_Ubuntu/datasets/PETS_2009/dmaps/S2L3/14_41/"

    #save_dir = os.path.join("/opt/visal/di/models/models_keras2/model_cell", exp_name)
    save_dir = os.path.join("models/", exp_name)
    print("save_dir: {}".format(save_dir))
    scaler_stability_factor = 1000

    stage = 'test_pred_nae'
    print(stage)

    model_name = '07-10.5414-better.h5'
    counting_results_name = 'counting_results/S12_all_1output_newlr2/' + model_name
    h5_savename = counting_results_name + '/counting_num_' + stage

    if os.path.isdir(counting_results_name):
        os.rmdir(counting_results_name)
    os.mkdir( counting_results_name )

    model = build_model_load_weights(image_dim=(288, 384, 1),
                                     model_dir='models/S12_all_1output_newlr2/',
                                     model_name = model_name)  # projection/

    # model = build_model_load_weights(image_dim=(288, 384, 1),
    #                                  model_dir='Cross_scene/Street_models_FCN7/MVMS',
    #                                  model_name=model_name)

    #################################################################

    # train
    train_test_path0 = '/opt/visal/home/qzhang364/Multi_view/Datasets/PETS_2009/dmaps/'
    # train_test_path0 = '/media/zq/16A4D077A4D05B37/0_Ubuntu/datasets/PETS_2009/dmaps/'

    # train_test_path0 = '/media/zq/16A4D077A4D05B37/0_Ubuntu/datasets/PETS_2009/dmaps/' \
    #                    'S1L1/13_57/train_test/'

    train_view1_1 = train_test_path0 + 'S1L3/14_17/train_test/PETS_S1L3_1_view1_train_test_10.h5'
    train_view1_2 = train_test_path0 + 'S1L3/14_33/train_test/PETS_S1L3_2_view1_train_test_10.h5'
    train_view1_3 = train_test_path0 + 'S2L2/14_55/train_test/10_10/PETS_S2L2_1_view1_train_test_10.h5'
    train_view1_4 = train_test_path0 + 'S2L3/14_41/train_test/10_10/PETS_S2L3_1_view1_train_test_10.h5'

    train_view2_1 = train_test_path0 + 'S1L3/14_17/train_test/PETS_S1L3_1_view2_train_test_10.h5'
    train_view2_2 = train_test_path0 + 'S1L3/14_33/train_test/PETS_S1L3_2_view2_train_test_10.h5'
    train_view2_3 = train_test_path0 + 'S2L2/14_55/train_test/10_10/PETS_S2L2_1_view2_train_test_10.h5'
    train_view2_4 = train_test_path0 + 'S2L3/14_41/train_test/10_10/PETS_S2L3_1_view2_train_test_10.h5'

    train_view3_1 = train_test_path0 + 'S1L3/14_17/train_test/PETS_S1L3_1_view3_train_test_10.h5'
    train_view3_2 = train_test_path0 + 'S1L3/14_33/train_test/PETS_S1L3_2_view3_train_test_10.h5'
    train_view3_3 = train_test_path0 + 'S2L2/14_55/train_test/10_10/PETS_S2L2_1_view3_train_test_10.h5'
    train_view3_4 = train_test_path0 + 'S2L3/14_41/train_test/10_10/PETS_S2L3_1_view3_train_test_10.h5'

    train_GP_1 = train_test_path0 + 'S1L3/14_17/GP_maps/PETS_S1L3_1_groundplane_dmaps_10.h5'
    train_GP_2 = train_test_path0 + 'S1L3/14_33/GP_maps/PETS_S1L3_2_groundplane_dmaps_10.h5'
    train_GP_3 = train_test_path0 + 'S2L2/14_55/GP_maps/PETS_S2L2_1_groundplane_dmaps_10.h5'
    train_GP_4 = train_test_path0 + 'S2L3/14_41/GP_maps/PETS_S2L3_1_groundplane_dmaps_10.h5'

    h5file_train_view1 = [train_view1_1, train_view1_2, train_view1_3, train_view1_4]
    h5file_train_view2 = [train_view2_1, train_view2_2, train_view2_3, train_view2_4]
    h5file_train_view3 = [train_view3_1, train_view3_2, train_view3_3, train_view3_4]
    h5file_train_GP = [train_GP_1, train_GP_2, train_GP_3, train_GP_4]

    # test
    # test_path0 = '/opt/visal/home/qzhang364/Multi_view/Datasets/PETS_2009/dmaps/'
    # test_path0 = '/media/zq/16A4D077A4D05B37/0_Ubuntu/datasets/PETS_2009/dmaps/'

    test_view1_1 = train_test_path0 + 'S1L1/13_57/train_test/PETS_S1L1_1_view1_train_test_10.h5'
    test_view1_2 = train_test_path0 + 'S1L1/13_59/train_test/PETS_S1L1_2_view1_train_test_10.h5'
    test_view1_3 = train_test_path0 + 'S1L2/14_06/train_test/PETS_S1L2_1_view1_train_test_10.h5'
    test_view1_4 = train_test_path0 + 'S1L2/14_31/train_test/PETS_S1L2_2_view1_train_test_10.h5'

    test_view2_1 = train_test_path0 + 'S1L1/13_57/train_test/PETS_S1L1_1_view2_train_test_10.h5'
    test_view2_2 = train_test_path0 + 'S1L1/13_59/train_test/PETS_S1L1_2_view2_train_test_10.h5'
    test_view2_3 = train_test_path0 + 'S1L2/14_06/train_test/PETS_S1L2_1_view2_train_test_10.h5'
    test_view2_4 = train_test_path0 + 'S1L2/14_31/train_test/PETS_S1L2_2_view2_train_test_10.h5'

    test_view3_1 = train_test_path0 + 'S1L1/13_57/train_test/PETS_S1L1_1_view3_train_test_10.h5'
    test_view3_2 = train_test_path0 + 'S1L1/13_59/train_test/PETS_S1L1_2_view3_train_test_10.h5'
    test_view3_3 = train_test_path0 + 'S1L2/14_06/train_test/PETS_S1L2_1_view3_train_test_10.h5'
    test_view3_4 = train_test_path0 + 'S1L2/14_31/train_test/PETS_S1L2_2_view3_train_test_10.h5'

    test_GP_1 = train_test_path0 + 'S1L1/13_57/GP_maps/PETS_S1L1_1_groundplane_dmaps_10.h5'
    test_GP_2 = train_test_path0 + 'S1L1/13_59/GP_maps/PETS_S1L1_2_groundplane_dmaps_10.h5'
    test_GP_3 = train_test_path0 + 'S1L2/14_06/GP_maps/PETS_S1L2_1_groundplane_dmaps_10.h5'
    test_GP_4 = train_test_path0 + 'S1L2/14_31/GP_maps/PETS_S1L2_2_groundplane_dmaps_10.h5'

    h5file_test_GP = [test_GP_1, test_GP_2, test_GP_3, test_GP_4]
    h5file_test_view1 = [test_view1_1, test_view1_2, test_view1_3, test_view1_4]
    h5file_test_view2 = [test_view2_1, test_view2_2, test_view2_3, test_view2_4]
    h5file_test_view3 = [test_view3_1, test_view3_2, test_view3_3, test_view3_4]

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

    for j in range(4):

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

        print(h4_dmaps_test.shape)

        # depth ratio maps input
        # view 1
        scale_number = 3
        scale_range = range(scale_number)
        scale_size = 2 * 4
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
        view1_image_depth_resized_log2 = np.expand_dims(view1_image_depth_resized_log2, axis=2)
        view1_image_depth_resized_log2 = np.expand_dims(view1_image_depth_resized_log2, axis=0)

        # view 2
        view2_image_depth = np.load('coords_correspondence/view_depth_image/'
                                    'v1_2_depth_image_halfHeight.npz')
        view2_image_depth = view2_image_depth.f.arr_0
        view2_image_depth_resized = cv2.resize(view2_image_depth, (w_scale, h_scale))
        view2_image_depth_resized_log2 = np.log2(view2_image_depth_resized / depth_center)
        # view_image_depth_resized_log2 = view2_image_depth_resized_log2
        view2_image_depth_resized_log2 = np.expand_dims(view2_image_depth_resized_log2, axis=2)
        view2_image_depth_resized_log2 = np.expand_dims(view2_image_depth_resized_log2, axis=0)

        # view 3
        view3_image_depth = np.load('coords_correspondence/view_depth_image/'
                                    'v1_3_depth_image_halfHeight.npz')
        view3_image_depth = view3_image_depth.f.arr_0
        view3_image_depth_resized = cv2.resize(view3_image_depth, (w_scale, h_scale))
        view3_image_depth_resized_log2 = np.log2(view3_image_depth_resized / depth_center)
        # view_image_depth_resized_log2 = view3_image_depth_resized_log2
        view3_image_depth_resized_log2 = np.expand_dims(view3_image_depth_resized_log2, axis=2)
        view3_image_depth_resized_log2 = np.expand_dims(view3_image_depth_resized_log2, axis=0)

        # GP mask:
        view1_gp_mask = np.load('coords_correspondence/mask/view1_gp_mask.npz')
        view1_gp_mask = view1_gp_mask.f.arr_0
        view2_gp_mask = np.load('coords_correspondence/mask/view2_gp_mask.npz')
        view2_gp_mask = view2_gp_mask.f.arr_0
        view3_gp_mask = np.load('coords_correspondence/mask/view3_gp_mask.npz')
        view3_gp_mask = view3_gp_mask.f.arr_0

        view_gp_mask = view1_gp_mask + view2_gp_mask + view3_gp_mask
        view_gp_mask = np.clip(view_gp_mask, 0, 1)
        # plt.imshow(view_gp_mask)
        view1_gp_mask2 = cv2.resize(view1_gp_mask, (610 / 4, 710 / 4))
        view2_gp_mask2 = cv2.resize(view2_gp_mask, (610 / 4, 710 / 4))
        view3_gp_mask2 = cv2.resize(view3_gp_mask, (610 / 4, 710 / 4))
        view_gp_mask = cv2.resize(view_gp_mask, (610 / 4, 710 / 4))

        count1 = []
        count2 = []
        count3 = []

        list_pred = list()
        pred_dmaps_list = []
        image_dim = None
        f_count = 0

        print(h3_test.shape[0])

        for i in range(h3_test.shape[0]):
            # i = 97

            frame1_s0 = h1_test[i:i+1]
            frame1 = frame1_s0[0, :, :, 0]

            frame1_s1_0 = cv2.resize(frame1,(frame1.shape[1]/2,frame1.shape[0]/2))
            frame1_s1 = np.expand_dims(frame1_s1_0, axis=0)
            frame1_s1 = np.expand_dims(frame1_s1, axis=3)

            frame1_s2_0 = cv2.resize(frame1_s1_0,(frame1_s1_0.shape[1]/2, frame1_s1_0.shape[0]/2))
            frame1_s2 = np.expand_dims(frame1_s2_0, axis=0)
            frame1_s2 = np.expand_dims(frame1_s2, axis=3)


            frame2_s0 = h2_test[i:i+1]
            frame2 = frame2_s0[0, :, :, 0]

            frame2_s1_0 = cv2.resize(frame2,(frame2.shape[1]/2,frame2.shape[0]/2))
            frame2_s1 = np.expand_dims(frame2_s1_0, axis=0)
            frame2_s1 = np.expand_dims(frame2_s1, axis=3)

            frame2_s2_0 = cv2.resize(frame2_s1_0,(frame2_s1_0.shape[1]/2, frame2_s1_0.shape[0]/2))
            frame2_s2 = np.expand_dims(frame2_s2_0, axis=0)
            frame2_s2 = np.expand_dims(frame2_s2, axis=3)


            frame3_s0 = h3_test[i:i+1]
            frame3 = frame3_s0[0, :, :, 0]

            frame3_s1_0 = cv2.resize(frame3,(frame3.shape[1]/2,frame3.shape[0]/2))
            frame3_s1 = np.expand_dims(frame3_s1_0, axis=0)
            frame3_s1 = np.expand_dims(frame3_s1, axis=3)

            frame3_s2_0 = cv2.resize(frame3_s1_0,(frame3_s1_0.shape[1]/2, frame3_s1_0.shape[0]/2))
            frame3_s2 = np.expand_dims(frame3_s2_0, axis=0)
            frame3_s2 = np.expand_dims(frame3_s2, axis=3)


            dmap1 = h1_dmaps_test[i:i+1]
            dmap2 = h2_dmaps_test[i:i+1]
            dmap3 = h3_dmaps_test[i:i+1]
            dmap4 = h4_dmaps_test[i:i + 1]

            count1_gt_i = np.sum(np.sum(dmap1[0, :, :, 0]))/1000
            count2_gt_i = np.sum(np.sum(dmap2[0, :, :, 0]))/1000
            count3_gt_i = np.sum(np.sum(dmap3[0, :, :, 0]))/1000
            count4_gt_i = np.sum(np.sum(dmap4[0, :, :, 0]))


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

            pred_dmap = model.predict_on_batch([frame1_s0, frame1_s1, frame1_s2,
                                                frame2_s0, frame2_s1, frame2_s2,
                                                frame3_s0, frame3_s1, frame3_s2,
                                                view1_image_depth_resized_log2,
                                                view2_image_depth_resized_log2,
                                                view3_image_depth_resized_log2])

            # count1_pred_i = np.sum(pred_dmap[0].flatten())/1000
            # count2_pred_i = np.sum(pred_dmap[1].flatten())/1000
            # count3_pred_i = np.sum(pred_dmap[2].flatten())/1000

            pred_dmap_0 = pred_dmap
            #pred_dmap_0 = pred_dmap_0*view_gp_mask
            count4_pred_i = np.sum(pred_dmap_0.flatten())/1000

            pred_dmap_gplane.append(pred_dmap)

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
            # plt.subplot(234)
            # plt.imshow(dmap4[0, :, :, 0], cmap='viridis')
            # plt.axis('off')
            # plt.subplot(235)
            # plt.imshow(pred_dmap[0, :, :, 0] / 1000, cmap='viridis')
            # plt.axis('off')
            # plt.show()


            # count1.append([count1_gt_i, count1_pred_i])
            # count2.append([count2_gt_i, count2_pred_i])
            # count3.append([count3_gt_i, count3_pred_i])
            count_gplane.append([count1_gt_i, count2_gt_i, count3_gt_i, count4_gt_i, count4_pred_i])

            # roi GP pred
            count_view1_roi_GP_i = np.sum(np.sum(pred_dmap_0[0, :, :, 0]  * view1_gp_mask2))/1000
            count_view2_roi_GP_i = np.sum(np.sum(pred_dmap_0[0, :, :, 0]  * view2_gp_mask2))/1000
            count_view3_roi_GP_i = np.sum(np.sum(pred_dmap_0[0, :, :, 0]  * view3_gp_mask2))/1000
            # roi GP gt
            count_view1_roi_GP_gt_i = np.sum(np.sum(dmap4[0, :, :, 0] * view1_gp_mask))
            count_view2_roi_GP_gt_i = np.sum(np.sum(dmap4[0, :, :, 0] * view2_gp_mask))
            count_view3_roi_GP_gt_i = np.sum(np.sum(dmap4[0, :, :, 0] * view3_gp_mask))
            count_view1_roi_GP.append([count_view1_roi_GP_gt_i, count_view1_roi_GP_i])
            count_view2_roi_GP.append([count_view2_roi_GP_gt_i, count_view2_roi_GP_i])
            count_view3_roi_GP.append([count_view3_roi_GP_gt_i, count_view3_roi_GP_i])

    # GP
    mae_GP = np.asarray(count_gplane)[:, 4]-np.asarray(count_gplane)[:, 3]
    mae_GP = np.mean(np.abs(mae_GP))
    print(mae_GP)

    # GP_nae
    mae_GP_nae = np.asarray(count_gplane)[:, 4]-np.asarray(count_gplane)[:, 3]
    nae_GP_i = np.abs(mae_GP_nae)/np.abs(np.asarray(count_gplane)[:, 3])
    nae_GP = np.mean(nae_GP_i)
    print(nae_GP)



    # GP roi/GP
    dif_view1_GP = np.asarray(count_view1_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    mae_view1_GP = np.mean(np.abs(dif_view1_GP))
    print(mae_view1_GP)
    dif_view2_GP = np.asarray(count_view2_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    mae_view2_GP = np.mean(np.abs(dif_view2_GP))
    print(mae_view2_GP)
    dif_view3_GP = np.asarray(count_view3_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    mae_view3_GP = np.mean(np.abs(dif_view3_GP))
    print(mae_view3_GP)

    # GP roi / GP_nae
    print('-----------GP_roi_GP_nae-----------')
    dif_view1_GP = np.asarray(count_view1_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    nae_view1_GP = np.mean(np.abs(dif_view1_GP)/np.abs(np.asarray(count_gplane)[:, 3]))
    print(nae_view1_GP)
    dif_view2_GP = np.asarray(count_view2_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    nae_view2_GP = np.mean(np.abs(dif_view2_GP)/np.abs(np.asarray(count_gplane)[:, 3]))
    print(nae_view2_GP)
    dif_view3_GP = np.asarray(count_view3_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 3]
    nae_view3_GP = np.mean(np.abs(dif_view3_GP)/np.abs(np.asarray(count_gplane)[:, 3]))
    print(nae_view3_GP)



    # GP roi
    mae_view1_GProi = np.asarray(count_view1_roi_GP)[:, 1]-np.asarray(count_view1_roi_GP)[:, 0]
    mae_view1_GProi = np.mean(np.abs(mae_view1_GProi))
    print(mae_view1_GProi)
    mae_view2_GProi = np.asarray(count_view2_roi_GP)[:, 1]-np.asarray(count_view2_roi_GP)[:, 0]
    mae_view2_GProi = np.mean(np.abs(mae_view2_GProi))
    print(mae_view2_GProi)
    mae_view3_GProi = np.asarray(count_view3_roi_GP)[:, 1]-np.asarray(count_view3_roi_GP)[:, 0]
    mae_view3_GProi = np.mean(np.abs(mae_view3_GProi))
    print(mae_view3_GProi)

    # GP roi / GP roi_nae
    print('-----------GP_roi_GP_roi_nae-----------')
    nae_view1_GProi = np.asarray(count_view1_roi_GP)[:, 1]-np.asarray(count_view1_roi_GP)[:, 0]
    nae_view1_GProi = np.mean(np.abs(nae_view1_GProi)/np.abs(np.asarray(count_view1_roi_GP)[:, 0]))
    print(nae_view1_GProi)
    nae_view2_GProi = np.asarray(count_view2_roi_GP)[:, 1]-np.asarray(count_view2_roi_GP)[:, 0]
    nae_view2_GProi = np.mean(np.abs(nae_view2_GProi)/np.abs(np.asarray(count_view2_roi_GP)[:, 0]))
    print(nae_view2_GProi)
    nae_view3_GProi = np.asarray(count_view3_roi_GP)[:, 1]-np.asarray(count_view3_roi_GP)[:, 0]
    nae_view3_GProi = np.mean(np.abs(nae_view3_GProi)/np.abs(np.asarray(count_view3_roi_GP)[:, 0]))
    print(nae_view3_GProi)



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

    # GP roi/view_nae
    print('-----------GP_roi_view_nae-----------')
    dif_view1 = np.asarray(count_view1_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 0]
    nae_view1 = np.mean(np.abs(dif_view1)/np.abs(np.asarray(count_gplane)[:, 0]))
    print(nae_view1)
    dif_view2 = np.asarray(count_view2_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 1]
    nae_view2 = np.mean(np.abs(dif_view2)/np.abs(np.asarray(count_gplane)[:, 1]))
    print(nae_view2)
    dif_view3 = np.asarray(count_view3_roi_GP)[:, 1] - np.asarray(count_gplane)[:, 2]
    nae_view3 = np.mean(np.abs(dif_view3)/np.abs(np.asarray(count_gplane)[:, 2]))
    print(nae_view3)



    with h5py.File(h5_savename, 'w') as f:
        f.create_dataset("count1_GProi", data=count_view1_roi_GP)
        f.create_dataset("count2_GProi", data=count_view2_roi_GP)
        f.create_dataset("count3_GProi", data=count_view3_roi_GP)
        f.create_dataset("count_gplane", data=count_gplane)

        f.create_dataset("mae_GP", data=mae_GP)
        f.create_dataset("nae_GP_i", data=nae_GP_i)
        f.create_dataset("nae_GP", data=nae_GP)
        # f.create_dataset("GAME", data=GAME)

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

        f.create_dataset("nae_view1_GP", data=nae_view1_GP)
        f.create_dataset("nae_view2_GP", data=nae_view2_GP)
        f.create_dataset("nae_view3_GP", data=nae_view3_GP)

        f.create_dataset("nae_view1", data=nae_view1)
        f.create_dataset("nae_view2", data=nae_view2)
        f.create_dataset("nae_view3", data=nae_view3)

        f.create_dataset("nae_view1_GProi", data=nae_view1_GProi)
        f.create_dataset("nae_view2_GProi", data=nae_view2_GProi)
        f.create_dataset("nae_view3_GProi", data=nae_view3_GProi)




        # pred_dmap1 = pred_dmap[0];
        # print(np.sum(np.sum(pred_dmap1[0, :, :, 0]))/1000)
        #
        # pred_dmap2 = pred_dmap[1];
        # print(np.sum(np.sum(pred_dmap2[0, :, :, 0]))/1000)
        #
        # pred_dmap3 = pred_dmap[2];
        # print(np.sum(np.sum(pred_dmap3[0, :, :, 0]))/1000)
        #
        # pred_dmap_gplane = pred_dmap[3];
        # print(np.sum(np.sum(pred_dmap_gplane[0, :, :, 0]))/1000)
        #
        # plt.figure()
        # plt.imshow(pred_dmap_gplane[0, :, :, 0])
        # plt.show()
        #
        #
        # from keras.models import Model
        # XX = model.input
        # YY_60 = model.layers[60].output
        # YY_61 = model.layers[61].output
        # YY_62 = model.layers[62].output
        #
        # YY_63 = model.layers[63].output
        # YY_64 = model.layers[64].output
        # YY_65 = model.layers[65].output
        #
        # YY_66 = model.layers[66].output
        #
        # YY_67 = model.layers[67].output
        # YY_68 = model.layers[68].output
        # YY_69 = model.layers[69].output
        # YY_70 = model.layers[70].output
        #
        # YY_71 = model.layers[71].output
        # #
        #
        # new_model_60 = Model(XX, YY_60)
        # new_model_61 = Model(XX, YY_61)
        # new_model_62 = Model(XX, YY_62)
        #
        # new_model_63 = Model(XX, YY_63)
        # new_model_64 = Model(XX, YY_64)
        # new_model_65 = Model(XX, YY_65)
        #
        # new_model_66 = Model(XX, YY_66)
        #
        # new_model_67 = Model(XX, YY_67)
        # new_model_68 = Model(XX, YY_68)
        # new_model_69 = Model(XX, YY_69)
        # new_model_70 = Model(XX, YY_70)
        #
        # new_model_71 = Model(XX, YY_71)
        # #new_model_72 = Model(XX, YY_72)
        #
        # Xaug = [frame1, frame2, frame3]
        # Xresult_60 = new_model_60.predict(Xaug)
        # Xresult_61 = new_model_61.predict(Xaug)
        # Xresult_62 = new_model_62.predict(Xaug)
        #
        # Xresult_63 = new_model_63.predict(Xaug)
        # Xresult_64 = new_model_64.predict(Xaug)
        # Xresult_65 = new_model_65.predict(Xaug)
        #
        # Xresult_66 = new_model_66.predict(Xaug)
        #
        # Xresult_67 = new_model_67.predict(Xaug)
        # Xresult_68 = new_model_68.predict(Xaug)
        # Xresult_69 = new_model_69.predict(Xaug)
        # Xresult_70 = new_model_70.predict(Xaug)
        #
        # Xresult_71 = new_model_71.predict(Xaug)
        #
        #
        #
        # # YY_75 = model.layers[75].output
        # # new_model_75 = Model(XX, YY_75)
        # # Xresult_75 = new_model_75.predict(Xaug)
        # # print('layer60')
        # # print(sum(sum(Xresult_75[0,:,:,0])))
        # # #print('layer60 shape')
        # # print(Xresult_75.shape)
        # # plt.figure()
        # # plt.imshow(Xresult_75[0,:,:,0])
        #
        #
        #
        # print('layer60')
        # print(sum(sum(Xresult_60[0,:,:,0])))
        # #print('layer60 shape')
        # print(Xresult_60.shape)
        # plt.figure()
        # plt.imshow(Xresult_60[0,:,:,0])
        # #plt.show()
        #
        # print('\n')
        # print('layer61')
        # print(sum(sum(Xresult_61[0,:,:,0])))
        # #print('layer63 shape')
        # plt.figure()
        # print(Xresult_61.shape)
        # plt.imshow(Xresult_61[0,:,:,0])
        # #plt.show()
        #
        # print('\n')
        # print('layer62')
        # print(sum(sum(Xresult_62[0,:,:,0])))
        # #print('layer63 shape')
        # plt.figure()
        # print(Xresult_62.shape)
        # plt.imshow(Xresult_62[0,:,:,0])
        # #plt.show()
        #
        # print('\n')
        # print('layer63')
        # print(sum(sum(Xresult_63[0,:,:,0])))
        # #print('layer63 shape')
        # plt.figure()
        # print(Xresult_63.shape)
        # plt.imshow(Xresult_63[0,:,:,0])
        # #plt.show()
        #
        # print('\n')
        # print('layer64')
        # print(sum(sum(Xresult_64[0,:,:,0])))
        # #print('layer63 shape')
        # plt.figure()
        # print(Xresult_64.shape)
        # plt.imshow(Xresult_64[0,:,:,0])
        # #plt.show()
        #
        # print('\n')
        # print('layer65')
        # print(sum(sum(Xresult_65[0,:,:,0])))
        # #print('layer63 shape')
        # plt.figure()
        # print(Xresult_65.shape)
        # plt.imshow(Xresult_65[0,:,:,0])
        # #plt.show()
        #
        # print('\n')
        # print('layer66')
        # print(sum(sum(Xresult_66[0,:,:,0])))
        # #print('layer66 shape')
        # plt.figure()
        # print(Xresult_66.shape)
        # plt.imshow(Xresult_66[0,:,:,0])
        # # plt.show()
        #
        # plt.figure()
        # print(Xresult_66.shape)
        # plt.imshow(Xresult_66[0,:,:,0]+Xresult_66[0,:,:,1]+Xresult_66[0,:,:,2])
        # # plt.show()
        #
        # print('\n')
        # print('layer67')
        # print(sum(sum(Xresult_67[0,:,:,0])))
        # #print('layer67 shape')
        #
        # #fig1 = plt.figure()
        # print(Xresult_67.shape)
        #
        # # fig1 = plt.figure()
        # # for i in range(0, 64):
        # #     img = Xresult_67[0,:,:,i]
        # #     fig1.add_subplot(8, 8, i+1)
        # #     plt.imshow(img)
        #
        # #plt.show()
        # # fig1.add_subplot(3, 3, 1)
        # # plt.imshow(Xresult_67[0,:,:,0])
        # # fig1.add_subplot(3, 3, 2)
        # # plt.imshow(Xresult_67[0, :, :, 1])
        # # fig1.add_subplot(3, 3, 3)
        # # plt.imshow(Xresult_67[0, :, :, 2])
        # # fig1.add_subplot(3, 3, 4)
        # # plt.imshow(Xresult_67[0, :, :, 3])
        # # fig1.add_subplot(3, 3, 5)
        # # plt.imshow(Xresult_67[0, :, :, 4])
        # # fig1.add_subplot(3, 3, 6)
        # # plt.imshow(Xresult_67[0, :, :, 5])
        # # fig1.add_subplot(3, 3, 7)
        # # plt.imshow(Xresult_67[0, :, :, 6])
        # # fig1.add_subplot(3, 3, 8)
        # # plt.imshow(Xresult_67[0, :, :, 7])
        # # fig1.add_subplot(3, 3, 9)
        # # plt.imshow(Xresult_67[0, :, :, 8])
        # # plt.show()
        #
        # print('\n')
        # print('layer68: activation')
        # print(sum(sum(Xresult_68[0,:,:,0])))
        # #print('layer68 shape')
        # #plt.figure()
        # print(Xresult_68.shape)
        # fig2 = plt.figure()
        # for i in range(0, 64):
        #     img = Xresult_68[0,:,:,i]
        #     fig2.add_subplot(8, 8, i+1)
        #     plt.imshow(img)
        # #plt.show()
        #
        # print('\n')
        # print('layer69: conv2')
        # print(sum(sum(Xresult_69[0,:,:,0])))
        # #print('layer69 shape')
        # #plt.figure()
        # print(Xresult_69.shape)
        # fig3 = plt.figure()
        # for i in range(0, 32):
        #     img = Xresult_69[0,:,:,i]
        #     fig3.add_subplot(4, 8, i+1)
        #     plt.imshow(img)
        # #plt.show()
        #
        # print('\n')
        # print('layer70: conv2-activation')
        # print(sum(sum(Xresult_70[0,:,:,0])))
        # #print('layer70 shape')
        # #plt.figure()
        # print(Xresult_70.shape)
        # fig4 = plt.figure()
        # for i in range(0, 32):
        #     img = Xresult_70[0,:,:,i]
        #     fig4.add_subplot(4, 8, i+1)
        #     plt.imshow(img)
        # #plt.show()
        #
        # print('\n')
        # print('layer71')
        # print(sum(sum(Xresult_71[0,:,:,0])))
        # #print('layer66 shape')
        # #plt.figure()
        # print(Xresult_71.shape)
        # fig5 = plt.figure()
        # for i in range(0, 1):
        #     img = Xresult_71[0,:,:,i]
        #     fig5.add_subplot(1, 1, i+1)
        #     plt.imshow(img)
        # #plt.show()
        #
        #
        #
        # YY_72 = model.layers[72].output
        # new_model_72 = Model(XX, YY_72)
        # Xresult_72 = new_model_72.predict(Xaug)
        #
        # print('\n')
        # print('layer72')
        # print(sum(sum(Xresult_72[0,:,:,0])))
        # #print('layer66 shape')
        # #plt.figure()
        # print(Xresult_72.shape)
        # fig6 = plt.figure()
        # for i in range(0, 1):
        #     img = Xresult_72[0,:,:,i]
        #     fig6.add_subplot(1, 1, i+1)
        #     plt.imshow(img)
        # plt.show()


        # patch_pred_list.append(patch_pred_dmap[:, :, np.newaxis])

        # count0 = np.sum(np.sum(pred_dmaps))
        # count.append(count0)
                
        # # for frame in videogen:
        # # print(frame.shape)
        # # f_count += 1
        # # assert frame.dtype == np.uint8

        # # plt.imshow(frame)
        # # plt.show()

        # #if f_count in range(1, 3302, 20):  #3302
        # print(f_count)
        # if image_dim is None:
        #     image_dim = frame.shape[:2]
        # frame_gray = rgb2gray(frame) #frame[:,:,0]  #frame[:,:,1] #rgb2gray(frame)
        # # downsized:
        # frame_gray = cv2.resize(frame_gray, (0,0), fx = 0.25, fy = 0.25)


        # # frame_gray = np.resize(frame_gray, (frame_gray.shape[0]/4, frame_gray.shape[1]/4))
        # # plt.imshow(frame_gray)
        # # patches = feature_extraction.image.extract_patches_2d(frame_rgb, (128,128))
        # # imPatch, h, w = im_patch.im2patch(image=frame_gray, patchsize = 128, stepsize = 32 )
        # # cur_input = rgb2gray(frame)[:, :, np.newaxis]
        # # imPatch = np.asarray(imPatch)
        # # cur_input = imPatch[:, :, :]
        # patch_pred_list = []
        # for i in range(len(imPatch)):
        #     patch = imPatch[i]
        #     patch = patch[:,:,np.newaxis]
        #     cur_input = patch[np.newaxis, :, :, :]

        #     patch2 = cv2.resize(patch, (0,0), fx = 0.5, fy = 0.5)
        #     patch2 = patch2[:, :, np.newaxis]
        #     cur_input2 = patch2[np.newaxis, :, :, :]

        #     # predict the patch
        #     patch_pred_dmap = model.predict_on_batch([cur_input, cur_input2]).squeeze() / scaler_stability_factor
        #     #patch_pred_dmap = model.predict_on_batch(cur_input).squeeze() / scaler_stability_factor

        #     #print(np.sum(np.sum(pred_dmap)))
        #     patch_pred_list.append(patch_pred_dmap[:, :, np.newaxis])
        # # reconstruct the image
        # im_pred = im_patch.patch2im(patch_pred_list, frame_gray, 32/4, h, w)
        # pred_dmaps = np.asarray(im_pred)

        # count0 = np.sum(np.sum(pred_dmaps))
        # count.append(count0)
        # print(np.sum(np.sum(pred_dmaps)))
        # # do not predict the region out of interest
        # pred_dmaps[:(500/4/4), :] = 0
        # #plt.imshow(pred_dmaps)
        # #plt.pause(0.2)
        # count0 = np.sum(np.sum(pred_dmaps))
        # count1.append(count0)

        # print(np.sum(np.sum(pred_dmaps)))

        # pred_dmaps_list.append(pred_dmaps)

        # # f0 = h5py.File('Count_num.h5', 'r')
        # # count_gtruth = f0["Count_Num"].value
        # # mae = np.sum(np.abs(count_gtruth-count))

        # f1 = h5py.File('count_num_pred_recept_field_540.h5', 'w')
        # down_set1 = f1.create_dataset('count_pred', data=count)
        # down_set2 = f1.create_dataset('count_pred_ROI', data=count1)
        # f1.close()


        # pred_dmaps_list = np.float32(np.asarray(pred_dmaps_list))
        # # plt.imshow(pred_dmaps_list[0])
        # # plt.show()
        # # plt.imshow(pred_dmaps_list[1])
        # # plt.show()
        # # assert pred_dmaps.dtype == np.float32, "dtype of pred_dmaps: {}".format(pred_dmaps.dtype)
        # # assert pred_dmaps.shape[0] == f_count, "shape of pred_dmaps: {}".format(pred_dmaps.shape)
        # # assert pred_dmaps.shape[3] == 1, "shape of pred_dmaps: {}".format(pred_dmaps.shape)
        # # assert pred_dmaps.shape[1] * 4 == image_dim[0], "shape of pred_dmaps: {}".format(pred_dmaps.shape)
        # # assert pred_dmaps.shape[2] * 4 == image_dim[1], "shape of pred_dmaps: {}".format(pred_dmaps.shape)
        # # mae /= nb_images
        # # print("MAE: {:.3f}".format(mae))
        # print("saving predicted density maps into\n    >>>> {}".format(h5_savename))
        # with h5py.File(os.path.join(save_dir, 'pred_{}'.format(h5_savename)), 'w') as f:
        #     f.create_dataset("pre_density_maps", data=pred_dmaps_list, shape=pred_dmaps_list.shape, dtype=np.float32,
        #                      chunks=True, compression='gzip', compression_opts=4, fletcher32=True)

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