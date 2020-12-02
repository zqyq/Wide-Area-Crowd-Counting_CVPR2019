from keras.layers.core import Layer
import tensorflow as tf
import numpy as np
import cv2

# import matplotlib.pyplot as plt

from keras.engine import InputSpec


class feature_scale_fusion_layer_rbm(Layer):

    def __init__(self,
                 scale_number=3,
                 #view = 0,
                 **kwargs):
        # self.scale = scale
        self.scale_number = scale_number
        #self.view = view

        super(feature_scale_fusion_layer_rbm, self).__init__(**kwargs)


    def build(self, input_shape):
        print('No trainable weights for mask layer.')

    def compute_output_shape(self, input_shape):
        scale_number = self.scale_number
        return ((input_shape[0],
                 input_shape[1],
                 input_shape[2],
                 input_shape[3]*scale_number))

    def call(self, x):
        # view = self.view
        scale_number = self.scale_number
        scale_range = range(scale_number)

        batch_size = x.shape[0].value
        height = x.shape[1].value
        width = x.shape[2].value
        num_channels = x.shape[3].value*scale_number

        output_mask = tf.zeros([batch_size, height, width, 1])
        for i in scale_range:
            scale_i = tf.constant(scale_range[i], 'float32')
            output_mask_i = -(tf.square(x-scale_i))
            output_mask = tf.concat([output_mask, output_mask_i], axis = 3)
        output_mask = output_mask[:,:,:,1:]

        output_mask = tf.nn.softmax(output_mask)

        # output_mask_max = tf.reduce_max(output_mask, axis=3)
        # output_mask_max = tf.expand_dims(output_mask_max, axis=3)
        # output_mask_max = tf.tile(output_mask_max, [1,1,1,num_channels])
        #
        # output_mask = output_mask - output_mask_max
        # output_mask = tf.sign(output_mask) + 1
        return output_mask

