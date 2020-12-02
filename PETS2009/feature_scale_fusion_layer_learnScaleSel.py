from keras.layers.core import Layer
import tensorflow as tf
import numpy as np
import cv2

# import matplotlib.pyplot as plt

from keras.engine import InputSpec


class feature_scale_fusion_layer(Layer):

    def __init__(self,
                 scale_number=3,
                 **kwargs):
        #self.scale = scale
        self.scale_number = scale_number
        #self.view = view
        super(feature_scale_fusion_layer, self).__init__(**kwargs)


    def build(self, input_shape):
        super(feature_scale_fusion_layer, self).build(input_shape)  # Be sure to call this at the end

    def compute_output_shape(self, input_shape):
        scale_number = self.scale_number
        feature = input_shape[0]
        return (int(feature[0]),
                int(feature[1]),
                int(feature[2]),
                int(feature[3]/scale_number))

    def call(self, x):
        scale_number = self.scale_number

        feature = x[0]
        mask = x[1]

        n_channels_single = int(feature.shape[-1].value / scale_number)
        batch_size = feature.shape[0].value
        height = feature.shape[1].value
        width = feature.shape[2].value
        num_channels = feature.shape[3].value
        feature_mask = tf.zeros([batch_size, height, width, n_channels_single])

        for i in range(scale_number):
            feature_i = feature[:, :, :, int(i * n_channels_single):int((i + 1) * n_channels_single)]
            mask_i = mask[:, :, :, i:i + 1]
            mask_i_n = tf.tile(mask_i, [1, 1, 1, n_channels_single])
            feature_mask_i = tf.multiply(feature_i, mask_i_n)
            feature_mask = feature_mask + feature_mask_i
        return feature_mask


