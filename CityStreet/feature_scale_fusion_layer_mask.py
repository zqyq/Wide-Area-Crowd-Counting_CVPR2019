from keras.layers.core import Layer
import tensorflow as tf
import numpy as np
import cv2

import matplotlib.pyplot as plt

from keras.engine import InputSpec


class feature_scale_fusion_layer_mask(Layer):

    def __init__(self,
                 scale_number=3,
                 view = 0,
                 output_shape = [1, 1, 1, 1],
                 **kwargs):
        # self.scale = scale
        self.scale_number = scale_number
        self.view = view
        self.o_shape = output_shape

        super(feature_scale_fusion_layer_mask, self).__init__(**kwargs)


    def build(self, input_shape):
        print('No trainable weights for mask layer.')

    def compute_output_shape(self, input_shape):
        scale_number = self.scale_number
        output_shape = self.o_shape
        return ((output_shape.shape[0],
                 output_shape.shape[1],
                 output_shape.shape[2],
                 output_shape.shape[2]))

    def call(self):
        view = self.view
        scale_number = self.scale_number
        output_shape = self.o_shape
        num_channels = output_shape.shape[3]
        num_channels_single = num_channels/scale_number

        scale = self.scale
        scale_range = range(scale_number)

        if view==1:
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
            view_image_depth_resized_log2 = view1_image_depth_resized_log2
        if view==2:
            scale_range = range(scale_number)
            scale_size = 2 * 4
            view1_image_depth = np.load('coords_correspondence/view_depth_image/'
                                        'v1_1_depth_image_halfHeight.npz')
            view1_image_depth = view1_image_depth.f.arr_0
            h = view1_image_depth.shape[0]
            w = view1_image_depth.shape[1]
            h_scale = h / scale_size
            w_scale = w / scale_size
            view1_image_depth_resized = cv2.resize(view1_image_depth, (w_scale, h_scale))

            # view 2
            view2_image_depth = np.load('coords_correspondence/view_depth_image/'
                                        'v1_2_depth_image_halfHeight.npz')
            view2_image_depth = view2_image_depth.f.arr_0
            # plt.figure()
            # plt.imshow(view1_image_depth)
            # plt.show()
            view2_image_depth_resized = cv2.resize(view2_image_depth, (w_scale, h_scale))

            # set the center's scale of the image view1 as median of the all scales
            scale_center = np.median(scale_range)
            depth_center = view1_image_depth_resized[h_scale / 2, w_scale / 2]
            view2_image_depth_resized_log2 = np.log2(view2_image_depth_resized / depth_center)
            view_image_depth_resized_log2 = view2_image_depth_resized_log2
        if view==3:
            scale_range = range(scale_number)
            scale_size = 2 * 4
            view1_image_depth = np.load('coords_correspondence/view_depth_image/'
                                        'v1_1_depth_image_halfHeight.npz')
            view1_image_depth = view1_image_depth.f.arr_0
            h = view1_image_depth.shape[0]
            w = view1_image_depth.shape[1]
            h_scale = h / scale_size
            w_scale = w / scale_size
            view1_image_depth_resized = cv2.resize(view1_image_depth, (w_scale, h_scale))

            # view 3
            view3_image_depth = np.load('coords_correspondence/view_depth_image/'
                                        'v1_3_depth_image_halfHeight.npz')
            view3_image_depth = view3_image_depth.f.arr_0
            # plt.figure()
            # plt.imshow(view1_image_depth)
            # plt.show()

            view3_image_depth_resized = cv2.resize(view3_image_depth, (w_scale, h_scale))

            # set the center's scale of the image view1 as median of the all scales
            scale_center = np.median(scale_range)
            depth_center = view1_image_depth_resized[h_scale / 2, w_scale / 2]
            view3_image_depth_resized_log2 = np.log2(view3_image_depth_resized / depth_center)
            view_image_depth_resized_log2 = view3_image_depth_resized_log2

        view_image_depth_resized_log2 = tf.constant(view_image_depth_resized_log2, dtype='float32')
        view_image_depth_resized_log2 = tf.expand_dims(view_image_depth_resized_log2, axis=2)
        view_image_depth_resized_log2 = tf.expand_dims(view_image_depth_resized_log2, axis=0)
        return view_image_depth_resized_log2

