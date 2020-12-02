from keras.layers.core import Layer
import tensorflow as tf
import numpy as np
import cv2

import matplotlib.pyplot as plt

from keras.engine import InputSpec


class feature_scale_fusion_layer(Layer):

    def __init__(self,
                 scale = 1,
                 scale_number=3,
                 view = 2,
                 **kwargs):
        self.scale = scale
        self.scale_number = scale_number
        self.view = view
        super(feature_scale_fusion_layer, self).__init__(**kwargs)


    def build(self, input_shape):

        super(feature_scale_fusion_layer, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        scale_number = self.scale_number
        return (int(input_shape[0]),
                int(input_shape[1]),
                int(input_shape[2]),
                int(input_shape[3]/scale_number))

    def call(self, x):
        view = self.view

        batch_size = x.shape[0].value
        height = x.shape[1]
        width  = x.shape[2]
        num_channels = x.shape[3].value

        scale_number = self.scale_number
        num_channels_single = num_channels/scale_number

        scale = self.scale
        scale_range = range(scale_number)

###################################################################################
        # load depth ratio map (masked)
        view1_2_depth_ratio = np.load('coords_correspondence/view_Depth_halfHeight/'
                                      'v1_2_depth_ratio_halfHeight.npz')
        view1_2_depth_ratio = view1_2_depth_ratio.f.arr_0
        view1_2_depth_ratio_log2 = np.log2(view1_2_depth_ratio)
        depth_H = view1_2_depth_ratio.shape[0]
        depth_W = view1_2_depth_ratio.shape[1]

        view1_3_depth_ratio = np.load('coords_correspondence/view_Depth_halfHeight/'
                                      'v1_3_depth_ratio_halfHeight.npz')
        view1_3_depth_ratio = view1_3_depth_ratio.f.arr_0
        view1_3_depth_ratio_log2 = np.log2(view1_3_depth_ratio)


        # resize the depth ration map according to the scale parameter
        scale_size = 1*4
        depth_H_scale = depth_H/scale_size
        depth_W_scale = depth_W/scale_size
        # the 2 pooling layers
        view1_2_depth_ratio_log2_resized = cv2.resize(view1_2_depth_ratio_log2,
                                                      (depth_W_scale, depth_H_scale))
        view1_3_depth_ratio_log2_resized = cv2.resize(view1_3_depth_ratio_log2,
                                                      (depth_W_scale, depth_H_scale))

        # build the scale-selection map:
        scale_map_v2 = scale - np.round(view1_2_depth_ratio_log2_resized)
        scale_map_v2 = np.clip(scale_map_v2, min(scale_range), max(scale_range))
        scale_map_v3 = scale - np.round(view1_3_depth_ratio_log2_resized)
        scale_map_v3 = np.clip(scale_map_v3, min(scale_range), max(scale_range))

        scale_selection_map_v2 = np.zeros([depth_H_scale, depth_W_scale, scale_number])
        scale_selection_map_v3 = np.zeros([depth_H_scale, depth_W_scale, scale_number])

        for i in range(depth_H_scale):
            for j in range(depth_W_scale):
                scale_sel2 = int(scale_map_v2[i, j])
                scale_selection_map_v2[i, j, scale_sel2] = 1

                scale_sel3 = int(scale_map_v3[i, j])
                scale_selection_map_v3[i, j, scale_sel3] = 1

        if view == 2:
            scale_selection_map = scale_selection_map_v2
        if view == 3:
            scale_selection_map = scale_selection_map_v3

        # plt.figure()
        # plt.imshow(scale_map_v2)
        # plt.show()
        #
        # plt.figure()
        # plt.imshow(scale_map_v3)
        # plt.show()
        #
        # fig1 = plt.figure()
        # plt.subplot(1, 3, 1)
        # plt.title('scale 0')
        # plt.imshow(scale_selection_map_v2[:,:,0])
        #
        # plt.subplot(1, 3, 2)
        # plt.title('scale 1')
        # plt.imshow(scale_selection_map_v2[:,:,1])
        #
        # plt.subplot(1, 3, 3)
        # plt.title('scale 2')
        # plt.imshow(scale_selection_map_v2[:,:,2])
        # plt.show()
        #
        #
        # fig2 = plt.figure()
        # plt.subplot(1, 3, 1)
        # plt.title('scale 0')
        # plt.imshow(scale_selection_map_v3[:, :, 0])
        #
        # plt.subplot(1, 3, 2)
        # plt.title('scale 1')
        # plt.imshow(scale_selection_map_v3[:, :, 1])
        #
        # plt.subplot(1, 3, 3)
        # plt.title('scale 2')
        # plt.imshow(scale_selection_map_v3[:, :, 2])
        # plt.show()

###################################################################################
        # input for convolution

        v_conv_add = tf.zeros([batch_size, height, width, num_channels_single])

        for s in scale_range:
            x2_s = x[:, :, :, (s + 0) * num_channels_single:(s + 1) * num_channels_single]

            v2_s = scale_selection_map[:, :, s]
            v2_s = tf.cast(v2_s, 'float32')
            v2_s = tf.expand_dims(v2_s, axis=0)
            v2_s = tf.expand_dims(v2_s, axis=3)
            v2_s = tf.tile(v2_s, [batch_size, 1, 1, num_channels_single])

            kx2_s_mul = tf.multiply(x2_s, v2_s)
            v_conv_add = v_conv_add + kx2_s_mul

        output = v_conv_add
        return output