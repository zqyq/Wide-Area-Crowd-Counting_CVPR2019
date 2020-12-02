from keras.layers.core import Layer
import tensorflow as tf
import numpy as np
import cv2

class SpatialTransformer(Layer):
    """Spatial Transformer Layer
    Implements a spatial transformer layer as described in [1]_.
    Borrowed from [2]_:
    downsample_fator : float
        A value of 1 will keep the orignal size of the image.
        Values larger than 1 will down sample the image. Values below 1 will
        upsample the image.
        example image: height= 100, width = 200
        downsample_factor = 2
        output image will then be 50, 100
    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    .. [3]  https://github.com/EderSantana/seya/blob/keras1/seya/layers/attention.py
    """

    def __init__(self,
                 view,
                 output_size,
                 **kwargs):
        #self.locnet = localization_net
        self.view = view
        self.output_size = output_size
        super(SpatialTransformer, self).__init__(**kwargs)


    # def build(self, input_shape):
    #    super(SpatialTransformer, self).build(input_shape)
        # self.locnet.build(input_shape)
        # self.trainable_weights = self.locnet.trainable_weights
        # self.regularizers = self.locnet.regularizers #//NOT SUER ABOUT THIS, THERE IS NO MORE SUCH PARAMETR AT self.locnet
        # self.constraints = self.locnet.constraints

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        return (int(input_shape[0]),
                int(output_size[0]),
                int(output_size[1]),
                int(input_shape[-1]))

    def call(self, inputs, mask=None):
        view = self.view
        output = self._transform(view, inputs, self.output_size)
        return output

    def _repeat(self, x, num_repeats):
        ones = tf.ones((1, num_repeats), dtype='int32')
        x = tf.reshape(x, shape=(-1,1))
        x = tf.matmul(x, ones)
        return tf.reshape(x, [-1])

    def _interpolate(self, image, x, y, output_size):
        batch_size = tf.shape(image)[0]
        height = tf.shape(image)[1]
        width = tf.shape(image)[2]
        num_channels = tf.shape(image)[3]
        batch_size = image.shape[0].value
        # height = image.shape[1].value
        # width = image.shape[2].value
        num_channels = image.shape[3].value

        x = tf.cast(x , dtype='float32')
        y = tf.cast(y , dtype='float32')

        height_float = tf.cast(height, dtype='float32')
        width_float = tf.cast(width, dtype='float32')

        output_height = output_size[0]
        output_width  = output_size[1]

        x = .5*(x + 1.0)*(width_float)
        y = .5*(y + 1.0)*(height_float)

        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        max_y = tf.cast(height - 1, dtype='int32')
        max_x = tf.cast(width - 1,  dtype='int32')
        zero = tf.zeros([], dtype='int32')

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        flat_image_dimensions = width*height
        pixels_batch = tf.range(batch_size)*flat_image_dimensions
        flat_output_dimensions = output_height*output_width
        base = self._repeat(pixels_batch, flat_output_dimensions)
        base_y0 = base + y0*width
        base_y1 = base + y1*width
        indices_a = base_y0 + x0
        indices_b = base_y1 + x0
        indices_c = base_y0 + x1
        indices_d = base_y1 + x1

        flat_image = tf.reshape(image, shape=(-1, num_channels))
        flat_image = tf.cast(flat_image, dtype='float32')
        pixel_values_a = tf.gather(flat_image, indices_a)
        pixel_values_b = tf.gather(flat_image, indices_b)
        pixel_values_c = tf.gather(flat_image, indices_c)
        pixel_values_d = tf.gather(flat_image, indices_d)

        x0 = tf.cast(x0, 'float32')
        x1 = tf.cast(x1, 'float32')
        y0 = tf.cast(y0, 'float32')
        y1 = tf.cast(y1, 'float32')

        area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
        area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
        area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
        area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
        output = tf.add_n([area_a*pixel_values_a,
                           area_b*pixel_values_b,
                           area_c*pixel_values_c,
                           area_d*pixel_values_d])
        return output

    def _meshgrid(self, height, width):
        x_linspace = tf.linspace(-1., 1., width)
        y_linspace = tf.linspace(-1., 1., height)

        # x_linspace = tf.linspace(0., width-1, width)
        # y_linspace = tf.linspace(0., height-1, height)

        x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
        x_coordinates = tf.reshape(x_coordinates, [-1])
        y_coordinates = tf.reshape(y_coordinates, [-1])
        ones = tf.ones_like(x_coordinates)
        indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
        return indices_grid

    def _transform(self, view, input_shape, output_size):
        batch_size = tf.shape(input_shape)[0]
        height = tf.shape(input_shape)[1]
        width = tf.shape(input_shape)[2]
        num_channels = tf.shape(input_shape)[3]

        batch_size = input_shape.shape[0].value
        # height = input_shape.shape[1].value
        # width = input_shape.shape[2].value
        num_channels = input_shape.shape[3].value

        # B = tf.shape(input_shape)[0]
        # H = tf.shape(input_shape)[1]
        # W = tf.shape(input_shape)[2]
        # C = tf.shape(input_shape)[3]
        # n_fc = 6
        # W_fc1 = tf.Variable(tf.zeros([H * W * C, n_fc]), name='W_fc1')
        # b_fc1 = tf.Variable(initial_value = affine_transformation, name='b_fc1')
        # affine_transformation = tf.matmul(tf.zeros([B, H * W * C]), W_fc1) + b_fc1

        # affine_transformation = tf.reshape(affine_transformation, shape=(batch_size,2,3))
        # affine_transformation = tf.reshape(affine_transformation, (-1, 2, 3))
        # affine_transformation = tf.cast(affine_transformation, 'float32')

        width = tf.cast(width, dtype='float32')
        height = tf.cast(height, dtype='float32')
        output_height = output_size[0]
        output_width = output_size[1]
        indices_grid = self._meshgrid(output_height, output_width)
        indices_grid = tf.expand_dims(indices_grid, 0)
        indices_grid = tf.reshape(indices_grid, [-1]) # flatten?

        indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
        indices_grid = tf.reshape(indices_grid, (batch_size, 3, -1))

        # transformed_grid = tf.matmul(affine_transformation, indices_grid)
        if(view == 1):
            view1_ic = np.load('coords_correspondence/projection_forth/view1_correspondence_forth.npz')
            view1_ic = view1_ic.f.arr_0
            view1_ic = tf.cast(view1_ic, 'float32')
            view1_ic = tf.expand_dims(view1_ic, axis=0)
            view1_ic = tf.tile(view1_ic, [batch_size, 1, 1])

            transformed_grid = tf.cast(view1_ic, 'float32')

            view1_gp_mask = np.load('coords_correspondence/mask/view1_gp_mask.npz')
            view1_gp_mask = view1_gp_mask.f.arr_0
            view1_gp_mask = cv2.resize(view1_gp_mask, (int(610 / 4), int(710 / 4)))
            view1_gp_mask = tf.cast(view1_gp_mask, 'float32')
            view1_gp_mask = tf.expand_dims(view1_gp_mask, axis=0)
            view1_gp_mask = tf.expand_dims(view1_gp_mask, axis=3)
            view1_gp_mask = tf.tile(view1_gp_mask, [batch_size, 1, 1, num_channels])
            view_gp_mask = tf.cast(view1_gp_mask, 'float32')

            view_norm_mask = np.load('coords_correspondence/norm/view_Wld_normalization_forth_mask.npz')
            view1_norm_mask= view_norm_mask.f.arr_0[0]
            view1_norm_mask = tf.cast(view1_norm_mask, 'float32')
            view1_norm_mask = tf.expand_dims(view1_norm_mask, axis=0)
            view1_norm_mask = tf.expand_dims(view1_norm_mask, axis=3)
            view1_norm_mask = tf.tile(view1_norm_mask, [batch_size, 1, 1, num_channels])
            view_norm_mask = tf.cast(view1_norm_mask, 'float32')

        elif (view == 2):
            view2_ic = np.load('coords_correspondence/projection_forth/view2_correspondence_forth.npz')
            view2_ic = view2_ic.f.arr_0
            view2_ic = tf.cast(view2_ic, 'float32')
            view2_ic = tf.expand_dims(view2_ic, axis=0)
            view2_ic = tf.tile(view2_ic, [batch_size, 1, 1])

            transformed_grid = tf.cast(view2_ic, 'float32')

            view2_gp_mask = np.load('coords_correspondence/mask/view2_gp_mask.npz')
            view2_gp_mask = view2_gp_mask.f.arr_0
            view2_gp_mask = cv2.resize(view2_gp_mask, (int(610 / 4), int(710 / 4)))
            view2_gp_mask = tf.cast(view2_gp_mask, 'float32')
            view2_gp_mask = tf.expand_dims(view2_gp_mask, axis=0)
            view2_gp_mask = tf.expand_dims(view2_gp_mask, axis=3)
            view2_gp_mask = tf.tile(view2_gp_mask, [batch_size, 1, 1, num_channels])
            view_gp_mask = tf.cast(view2_gp_mask, 'float32')

            view_norm_mask = np.load('coords_correspondence/norm/view_Wld_normalization_forth_mask.npz')
            view2_norm_mask= view_norm_mask.f.arr_0[1]
            view2_norm_mask = tf.cast(view2_norm_mask, 'float32')
            view2_norm_mask = tf.expand_dims(view2_norm_mask, axis=0)
            view2_norm_mask = tf.expand_dims(view2_norm_mask, axis=3)
            view2_norm_mask = tf.tile(view2_norm_mask, [batch_size, 1, 1, num_channels])
            view_norm_mask = tf.cast(view2_norm_mask, 'float32')

        elif (view == 3):
            view3_ic = np.load('coords_correspondence/projection_forth/view3_correspondence_forth.npz')
            view3_ic = view3_ic.f.arr_0
            view3_ic = tf.cast(view3_ic, 'float32')
            view3_ic = tf.expand_dims(view3_ic, axis=0)
            view3_ic = tf.tile(view3_ic, [batch_size, 1, 1])

            transformed_grid = tf.cast(view3_ic, 'float32')

            view3_gp_mask = np.load('coords_correspondence/mask/view3_gp_mask.npz')
            view3_gp_mask = view3_gp_mask.f.arr_0
            view3_gp_mask = cv2.resize(view3_gp_mask, (int(610 / 4), int(710 / 4)))
            view3_gp_mask = tf.cast(view3_gp_mask, 'float32')
            view3_gp_mask = tf.expand_dims(view3_gp_mask, axis=0)
            view3_gp_mask = tf.expand_dims(view3_gp_mask, axis=3)
            view3_gp_mask = tf.tile(view3_gp_mask, [batch_size, 1, 1, num_channels])
            view_gp_mask = tf.cast(view3_gp_mask, 'float32')

            view_norm_mask = np.load('coords_correspondence/norm/view_Wld_normalization_forth_mask.npz')
            view3_norm_mask= view_norm_mask.f.arr_0[2]
            view3_norm_mask = tf.cast(view3_norm_mask, 'float32')
            view3_norm_mask = tf.expand_dims(view3_norm_mask, axis=0)
            view3_norm_mask = tf.expand_dims(view3_norm_mask, axis=3)
            view3_norm_mask = tf.tile(view3_norm_mask, [batch_size, 1, 1, num_channels])
            view_norm_mask = tf.cast(view3_norm_mask, 'float32')


        x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
        x_s_flatten = tf.reshape(x_s, [-1])
        y_s_flatten = tf.reshape(y_s, [-1])

        transformed_image = self._interpolate(input_shape,
                                                x_s_flatten,
                                                y_s_flatten,
                                                output_size)

        transformed_image = tf.reshape(transformed_image, shape=(batch_size,
                                                                output_height,
                                                                output_width,
                                                                num_channels))

        transformed_image = tf.multiply(transformed_image, view_gp_mask)
        # transformed_image = tf.multiply(transformed_image, view_norm_mask)

        # # normalization:
        # # get the sum of each channel/each image
        # input_sum = tf.reduce_sum(input_shape, [1, 2])
        # input_sum = tf.expand_dims(input_sum, axis=1)
        # input_sum = tf.expand_dims(input_sum, axis=1)
        #
        # output_sum = tf.reduce_sum(transformed_image, [1, 2])
        # # output_sum = tf.multiply(tf.ones(output_sum.shape), 0.01) + output_sum
        # output_sum = tf.expand_dims(output_sum, axis=1)
        # output_sum = tf.expand_dims(output_sum, axis=1)
        #
        # amplify_times = tf.divide(input_sum, output_sum)
        # mul_times = tf.constant([1, output_height, output_width, 1])
        # amplify_times = tf.tile(amplify_times, mul_times)
        #
        # # transformed_image = tf.image.resize_images(transformed_image,
        # #                                            [output_height/4, output_width/4])
        #
        # transformed_image_sum = tf.multiply(transformed_image, amplify_times)

        return transformed_image