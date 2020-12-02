from keras.layers.core import Layer
import tensorflow as tf
import numpy as np
import cv2

import camera_proj_Duke as proj


class SpatialTransformer_2DTo2D_real(Layer):
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
                 patch_num=1,
                 **kwargs):
        #self.locnet = localization_net
        self.view = view
        self.output_size = output_size
        self.patch_num = patch_num

        super(SpatialTransformer_2DTo2D_real, self).__init__(**kwargs)


    # def build(self, input_shape):
    #    super(SpatialTransformer, self).build(input_shape)
        # self.locnet.build(input_shape)
        # self.trainable_weights = self.locnet.trainable_weights
        # self.regularizers = self.locnet.regularizers #//NOT SUER ABOUT THIS, THERE IS NO MORE SUCH PARAMETR AT self.locnet
        # self.constraints = self.locnet.constraints

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        patch_num = self.patch_num

        return (int(input_shape[0]/patch_num),
                patch_num,
                int(output_size[0]),
                int(output_size[1]),
                int(input_shape[-1]) # , 33
                )  # add the depth too.



    def call(self, inputs, mask=None):
        # view = self.view
        # num_channels = self.num_channels
        #
        # output_size = self.output_size
        # output_height = int(output_size[0])
        # output_width = int(output_size[1])
        #
        # batch_size = inputs.shape[0].value
        # height = inputs.shape[1].value
        # width = inputs.shape[2].value
        # num_channels0 = inputs.shape[3].value
        #
        # D = int(36 / 4) * 4
        # height_range = np.linspace(0, D - 1, D)
        #
        # output = tf.zeros([batch_size, output_height, output_width, 1])
        # for i in range(len(height_range)):
        #     hi = height_range[i]
        #     input_i = inputs[:, :, :, i:i + 1]
        #     self.proj_2DTo2D(view, input_i, hi)  # inputs
        #     output_i = self.proj_splat(input_i)  # inputs
        #     output = tf.concat([output, output_i], axis=-1)
        # output = output[:, :, :, 1:]
        #
        # output = tf.expand_dims(output, axis=-1)
        # num_channels = num_channels
        # output = tf.tile(output, [batch_size, 1, 1, 1, num_channels])
        #
        # return output

        view = self.view
        output = tf.placeholder(dtype=tf.float32)
        for i in range(len(view)):
            view_i = view[i]
            self.proj_2DTo2D(view_i, inputs[i:i+1, ...])
            output_i = self.proj_splat(inputs[i:i+1, ...]) #inputs

            if i==0:
                output = output_i
            else:
                output = tf.concat([output, output_i], axis = 0)
        return output



    # util functions
    def tf_static_shape(self, T):
        return T.get_shape().as_list()

    def Image2World(self, view, imgcoords):
        N = imgcoords.shape[0]
        wld_coords = []
        for i in range(N):
            imgcoords_i = imgcoords[i, :]

            Xi = imgcoords_i[0]
            Yi = imgcoords_i[1]
            Zw = imgcoords_i[2]

            XYw = proj.Image2World(view, Xi, Yi, Zw)
            wld_coords.append(XYw)
        wld_coords = np.asarray(wld_coords)
        return wld_coords

    def World2Image(self, view, wldcoords):
        N = wldcoords.shape[0]
        imgcoords = []
        for i in range(N):
            wldcoords_i = wldcoords[i, :]

            Xw = wldcoords_i[0]
            Yw = wldcoords_i[1]
            Zw = wldcoords_i[2]

            XYi = proj.World2Image(view, Xw, Yw, Zw)
            imgcoords.append(XYi)
        imgcoords = np.asarray(imgcoords)
        return imgcoords

    def proj_2DTo2D(self, view, inputs):
        w = 640
        h = 360
        W = int(640/4)
        H = int(480/4)

        # D = hi # 36/4*4

        bbox = [50, 50]  # assuming half-man plane/ 1.75/2*1000
        # bbox = [-21, 19, -25, 15]

        image_size = [h/4, w/4]
        resolution_scaler = 4  # control the resolution of the project ROI mask (ground plane density map).
        ph = 1.75  # average height of a person in millimeters

        # nR, fh, fw, fdim = self.tf_static_shape(inputs)

        step_height = 100
        step_height = ph * 1000


        nR, fh, fw, fdim = inputs.get_shape().as_list()
        # self.batch_size, self.gp_x, self.gp_y, self.gp_z = nR, W, H, 1 # D = 1
        self.batch_size, self.gp_x, self.gp_y = nR, W, H # D = 1


        rsz_h = float(fh) / h/3
        rsz_w = float(fw) / w/3

        # Create voxel grid
        grid_rangeX = np.linspace(0, W - 1, W)
        grid_rangeY = np.linspace(0, H - 1, H)
        # grid_rangeZ = hi #np.linspace(0, D - 1, D)
        # grid_rangeX, grid_rangeY, grid_rangeZ = np.meshgrid(grid_rangeX, grid_rangeY, grid_rangeZ)
        grid_rangeX, grid_rangeY = np.meshgrid(grid_rangeX, grid_rangeY)

        grid_rangeX = np.reshape(grid_rangeX, [-1])
        grid_rangeY = np.reshape(grid_rangeY, [-1])
        # grid_rangeZ = np.reshape(grid_rangeZ, [-1])

        grid_rangeX = grid_rangeX * 4 / resolution_scaler - bbox[0]
        grid_rangeX = grid_rangeX
        grid_rangeX = np.expand_dims(grid_rangeX, 1)

        grid_rangeY = grid_rangeY * 4 / resolution_scaler - bbox[1]
        grid_rangeY = grid_rangeY
        grid_rangeY = np.expand_dims(grid_rangeY, 1)

        # grid_rangeZ = grid_rangeZ * step_height/1000 * np.ones(grid_rangeX.shape)
        grid_rangeZ = step_height/1000 * np.ones(grid_rangeX.shape)

        # grid_rangeZ = np.expand_dims(grid_rangeZ, 1)

        wldcoords = np.concatenate(([grid_rangeX, grid_rangeY, grid_rangeZ]), axis=1)

        if view==1:
            view = 'view1'
            view_gp_mask = np.load('coords_correspondence_DukeMTMC/mask/view1_GP_mask.npz')
        if view==2:
            view = 'view2'
            view_gp_mask = np.load('coords_correspondence_DukeMTMC/mask/view2_GP_mask.npz')
        if view==3:
            view = 'view3'
            view_gp_mask = np.load('coords_correspondence_DukeMTMC/mask/view3_GP_mask.npz')
        if view==4:
            view = 'view4'
            view_gp_mask = np.load('coords_correspondence_DukeMTMC/mask/view4_GP_mask.npz')
        #
        # # gp view mask:
        # view_gp_mask = view_gp_mask.f.arr_0
        # view_gp_mask = cv2.resize(view_gp_mask, (640 / 4, 480 / 4))
        # view_gp_mask = tf.cast(view_gp_mask, 'float32')
        # view_gp_mask = tf.expand_dims(view_gp_mask, axis=0)
        # view_gp_mask = tf.expand_dims(view_gp_mask, axis=-1)
        # view_gp_mask = tf.expand_dims(view_gp_mask, axis=-1)
        # batch_size = nR
        # num_channels = fdim ###### no to add the depth dim
        # view_gp_mask = tf.tile(view_gp_mask, [batch_size, 1, 1, self.gp_z, num_channels])
        # view_gp_mask = tf.cast(view_gp_mask, 'float32')
        # self.view_gp_mask = view_gp_mask

        # view1_ic = self.World2Image('view1', wldcoords)
        # view2_ic = self.World2Image('view2', wldcoords)
        # view3_ic = self.World2Image('view3', wldcoords)
        view_ic = self.World2Image(view, wldcoords)

        # view1_ic = np.transpose(view1_ic)
        # view2_ic = np.transpose(view2_ic)
        # view3_ic = np.transpose(view3_ic)
        view_ic = np.transpose(view_ic)


        # # normalization:
        # view1_ic[0:1, :] = view1_ic[0:1, :] * rsz_w
        # view1_ic[1:2, :] = view1_ic[1:2, :] * rsz_h
        #
        # view2_ic[0:1, :] = view2_ic[0:1, :] * rsz_w
        # view2_ic[1:2, :] = view2_ic[1:2, :] * rsz_h
        #
        # view3_ic[0:1, :] = view3_ic[0:1, :] * rsz_w
        # view3_ic[1:2, :] = view3_ic[1:2, :] * rsz_h

        view_ic[0:1, :] = view_ic[0:1, :] * rsz_w
        view_ic[1:2, :] = view_ic[1:2, :] * rsz_h
        view_ic[2:3, :] = view_ic[2:3, :] /(step_height/1000.0)

        # net.proj_view = np.concatenate(
        #     [view1_ic[0:1, :], view2_ic[0:1, :], view3_ic[0:1, :],
        #      view1_ic[1:2, :], view2_ic[1:2, :], view3_ic[1:2, :],
        #      view1_ic[2:3, :], view2_ic[2:3, :], view3_ic[2:3, :]],
        #     axis=0)
        self.proj_view = np.concatenate(
            [view_ic[0:1, :], view_ic[1:2, :], view_ic[2:3, :]],axis=0)

    def proj_splat(self,  inputs):
        with tf.variable_scope('ProjSplat'):
            nR, fh, fw, fdim = self.tf_static_shape(inputs)

            nV = self.proj_view.shape[1]

            im_p = self.proj_view
            im_x, im_y, im_z = im_p[::3, :], im_p[1::3, :], im_p[2::3, :]

            im_x = tf.constant(im_x, dtype='float32')
            im_y = tf.constant(im_y, dtype='float32')
            im_z = tf.constant(im_z, dtype='float32')
            self.im_p, self.im_x, self.im_y, self.im_z = im_p, im_x, im_y, im_z

            # im_p = tf.constant(self.proj_view, dtype='float32')
            # im_x, im_y, im_z = self[::3, :], im_p[1::3, :], im_p[2::3, :]
            # self.im_p, self.im_x, self.im_y, self.im_z = im_p, im_x, im_y, im_z

            # Bilinear interpolation
            with tf.name_scope('BilinearInterp'):
                im_x = tf.clip_by_value(im_x, 0, fw - 1)
                im_y = tf.clip_by_value(im_y, 0, fh - 1)

                im_x0 = tf.cast(tf.floor(im_x), 'int32')
                im_x1 = im_x0 + 1
                im_x1 = tf.clip_by_value(im_x1, 0, fw - 1)

                im_y0 = tf.cast(tf.floor(im_y), 'int32')
                im_y1 = im_y0 + 1
                im_y1 = tf.clip_by_value(im_y1, 0, fh - 1)

                im_x0_f, im_x1_f = tf.to_float(im_x0), tf.to_float(im_x1)
                im_y0_f, im_y1_f = tf.to_float(im_y0), tf.to_float(im_y1)

                ind_grid = tf.range(0, nR)
                ind_grid = tf.expand_dims(ind_grid, 1)
                im_ind = tf.tile(ind_grid, [1, nV])

                def _get_gather_inds(x, y):
                    return tf.reshape(tf.stack([im_ind, y, x], axis=2), [-1, 3])

                # Gather  values
                Ia = tf.gather_nd(inputs, _get_gather_inds(im_x0, im_y0))
                Ib = tf.gather_nd(inputs, _get_gather_inds(im_x0, im_y1))
                Ic = tf.gather_nd(inputs, _get_gather_inds(im_x1, im_y0))
                Id = tf.gather_nd(inputs, _get_gather_inds(im_x1, im_y1))

                # Calculate bilinear weights
                wa = (im_x1_f - im_x) * (im_y1_f - im_y)
                wb = (im_x1_f - im_x) * (im_y - im_y0_f)
                wc = (im_x - im_x0_f) * (im_y1_f - im_y)
                wd = (im_x - im_x0_f) * (im_y - im_y0_f)
                wa, wb = tf.reshape(wa, [-1, 1]), tf.reshape(wb, [-1, 1])
                wc, wd = tf.reshape(wc, [-1, 1]), tf.reshape(wd, [-1, 1])
                self.wa, self.wb, self.wc, self.wd = wa, wb, wc, wd
                self.Ia, self.Ib, self.Ic, self.Id = Ia, Ib, Ic, Id
                Ibilin = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

            with tf.name_scope('AppendDepth'):
                # Concatenate depth value along ray to feature
                # Ibilin = tf.concat(
                #     [Ibilin, tf.reshape(im_z, [nV * nR, 1])], axis=1)
                # fdim = Ibilin.get_shape().as_list()[-1]
                Ibilin = tf.reshape(Ibilin, [int(self.batch_size/self.patch_num), self.patch_num,
                                             self.gp_y, self.gp_x, fdim]) # fdim

                self.Ibilin = Ibilin

                # add a mask:
                # self.Ibilin = tf.multiply(Ibilin, self.view_gp_mask)

                # self.Ibilin = tf.transpose(self.Ibilin, [0, 2, 1, 3, 4])
        return self.Ibilin















































    #
    # def _repeat(self, x, num_repeats):
    #     ones = tf.ones((1, num_repeats), dtype='int32')
    #     x = tf.reshape(x, shape=(-1,1))
    #     x = tf.matmul(x, ones)
    #     return tf.reshape(x, [-1])
    #
    # def _interpolate(self, image, x, y, output_size):
    #     batch_size = tf.shape(image)[0]
    #     height = tf.shape(image)[1]
    #     width = tf.shape(image)[2]
    #     num_channels = tf.shape(image)[3]
    #     # batch_size = image.shape[0].value
    #     # height = image.shape[1].value
    #     # width = image.shape[2].value
    #     # num_channels = image.shape[3].value
    #
    #     x = tf.cast(x , dtype='float32')
    #     y = tf.cast(y , dtype='float32')
    #
    #     height_float = tf.cast(height, dtype='float32')
    #     width_float = tf.cast(width, dtype='float32')
    #
    #     output_height = output_size[0]
    #     output_width  = output_size[1]
    #
    #     x = .5*(x + 1.0)*(width_float)
    #     y = .5*(y + 1.0)*(height_float)
    #
    #     x0 = tf.cast(tf.floor(x), 'int32')
    #     x1 = x0 + 1
    #     y0 = tf.cast(tf.floor(y), 'int32')
    #     y1 = y0 + 1
    #
    #     max_y = tf.cast(height - 1, dtype='int32')
    #     max_x = tf.cast(width - 1,  dtype='int32')
    #     zero = tf.zeros([], dtype='int32')
    #
    #     x0 = tf.clip_by_value(x0, zero, max_x)
    #     x1 = tf.clip_by_value(x1, zero, max_x)
    #     y0 = tf.clip_by_value(y0, zero, max_y)
    #     y1 = tf.clip_by_value(y1, zero, max_y)
    #
    #     flat_image_dimensions = width*height
    #     pixels_batch = tf.range(batch_size)*flat_image_dimensions
    #     flat_output_dimensions = output_height*output_width
    #     base = self._repeat(pixels_batch, flat_output_dimensions)
    #     base_y0 = base + y0*width
    #     base_y1 = base + y1*width
    #     indices_a = base_y0 + x0
    #     indices_b = base_y1 + x0
    #     indices_c = base_y0 + x1
    #     indices_d = base_y1 + x1
    #
    #     flat_image = tf.reshape(image, shape=(-1, num_channels))
    #     flat_image = tf.cast(flat_image, dtype='float32')
    #     pixel_values_a = tf.gather(flat_image, indices_a)
    #     pixel_values_b = tf.gather(flat_image, indices_b)
    #     pixel_values_c = tf.gather(flat_image, indices_c)
    #     pixel_values_d = tf.gather(flat_image, indices_d)
    #
    #     x0 = tf.cast(x0, 'float32')
    #     x1 = tf.cast(x1, 'float32')
    #     y0 = tf.cast(y0, 'float32')
    #     y1 = tf.cast(y1, 'float32')
    #
    #     area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
    #     area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
    #     area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
    #     area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
    #     output = tf.add_n([area_a*pixel_values_a,
    #                        area_b*pixel_values_b,
    #                        area_c*pixel_values_c,
    #                        area_d*pixel_values_d])
    #     return output
    #
    # def _meshgrid(self, height, width):
    #     x_linspace = tf.linspace(-1., 1., width)
    #     y_linspace = tf.linspace(-1., 1., height)
    #
    #     # x_linspace = tf.linspace(0., width-1, width)
    #     # y_linspace = tf.linspace(0., height-1, height)
    #
    #     x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
    #     x_coordinates = tf.reshape(x_coordinates, [-1])
    #     y_coordinates = tf.reshape(y_coordinates, [-1])
    #     ones = tf.ones_like(x_coordinates)
    #     indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
    #     return indices_grid
    #
    # def _transform(self, view, input_shape, output_size):
    #     batch_size = tf.shape(input_shape)[0]
    #     height = tf.shape(input_shape)[1]
    #     width = tf.shape(input_shape)[2]
    #     num_channels = tf.shape(input_shape)[3]
    #
    #     # batch_size = input_shape.shape[0].value
    #     # height = input_shape.shape[1].value
    #     # width = input_shape.shape[2].value
    #     # num_channels = input_shape.shape[3].value
    #
    #     # B = tf.shape(input_shape)[0]
    #     # H = tf.shape(input_shape)[1]
    #     # W = tf.shape(input_shape)[2]
    #     # C = tf.shape(input_shape)[3]
    #     # n_fc = 6
    #     # W_fc1 = tf.Variable(tf.zeros([H * W * C, n_fc]), name='W_fc1')
    #     # b_fc1 = tf.Variable(initial_value = affine_transformation, name='b_fc1')
    #     # affine_transformation = tf.matmul(tf.zeros([B, H * W * C]), W_fc1) + b_fc1
    #
    #     # affine_transformation = tf.reshape(affine_transformation, shape=(batch_size,2,3))
    #     # affine_transformation = tf.reshape(affine_transformation, (-1, 2, 3))
    #     # affine_transformation = tf.cast(affine_transformation, 'float32')
    #
    #     width = tf.cast(width, dtype='float32')
    #     height = tf.cast(height, dtype='float32')
    #     output_height = output_size[0]
    #     output_width = output_size[1]
    #     indices_grid = self._meshgrid(output_height, output_width)
    #     indices_grid = tf.expand_dims(indices_grid, 0)
    #     indices_grid = tf.reshape(indices_grid, [-1]) # flatten?
    #
    #     indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
    #     indices_grid = tf.reshape(indices_grid, (batch_size, 3, -1))
    #
    #     # transformed_grid = tf.matmul(affine_transformation, indices_grid)
    #     if(view == 1):
    #         view1_ic = np.load('coords_correspondence/projection_forth/view1_correspondence_forth.npz')
    #         view1_ic = view1_ic.f.arr_0
    #         view1_ic = tf.cast(view1_ic, 'float32')
    #         view1_ic = tf.expand_dims(view1_ic, axis=0)
    #         view1_ic = tf.tile(view1_ic, [batch_size, 1, 1])
    #
    #         transformed_grid = tf.cast(view1_ic, 'float32')
    #
    #         view1_gp_mask = np.load('coords_correspondence/mask/view1_gp_mask.npz')
    #         view1_gp_mask = view1_gp_mask.f.arr_0
    #         view1_gp_mask = cv2.resize(view1_gp_mask, (610 / 4, 710 / 4))
    #         view1_gp_mask = tf.cast(view1_gp_mask, 'float32')
    #         view1_gp_mask = tf.expand_dims(view1_gp_mask, axis=0)
    #         view1_gp_mask = tf.expand_dims(view1_gp_mask, axis=3)
    #         view1_gp_mask = tf.tile(view1_gp_mask, [batch_size, 1, 1, num_channels])
    #         view_gp_mask = tf.cast(view1_gp_mask, 'float32')
    #
    #         view_norm_mask = np.load('coords_correspondence/norm/view_Wld_normalization_forth_mask.npz')
    #         view1_norm_mask= view_norm_mask.f.arr_0[0]
    #         view1_norm_mask = tf.cast(view1_norm_mask, 'float32')
    #         view1_norm_mask = tf.expand_dims(view1_norm_mask, axis=0)
    #         view1_norm_mask = tf.expand_dims(view1_norm_mask, axis=3)
    #         view1_norm_mask = tf.tile(view1_norm_mask, [batch_size, 1, 1, num_channels])
    #         view_norm_mask = tf.cast(view1_norm_mask, 'float32')
    #
    #     elif (view == 2):
    #         view2_ic = np.load('coords_correspondence/projection_forth/view2_correspondence_forth.npz')
    #         view2_ic = view2_ic.f.arr_0
    #         view2_ic = tf.cast(view2_ic, 'float32')
    #         view2_ic = tf.expand_dims(view2_ic, axis=0)
    #         view2_ic = tf.tile(view2_ic, [batch_size, 1, 1])
    #
    #         transformed_grid = tf.cast(view2_ic, 'float32')
    #
    #         view2_gp_mask = np.load('coords_correspondence/mask/view2_gp_mask.npz')
    #         view2_gp_mask = view2_gp_mask.f.arr_0
    #         view2_gp_mask = cv2.resize(view2_gp_mask, (610 / 4, 710 / 4))
    #         view2_gp_mask = tf.cast(view2_gp_mask, 'float32')
    #         view2_gp_mask = tf.expand_dims(view2_gp_mask, axis=0)
    #         view2_gp_mask = tf.expand_dims(view2_gp_mask, axis=3)
    #         view2_gp_mask = tf.tile(view2_gp_mask, [batch_size, 1, 1, num_channels])
    #         view_gp_mask = tf.cast(view2_gp_mask, 'float32')
    #
    #         view_norm_mask = np.load('coords_correspondence/norm/view_Wld_normalization_forth_mask.npz')
    #         view2_norm_mask= view_norm_mask.f.arr_0[1]
    #         view2_norm_mask = tf.cast(view2_norm_mask, 'float32')
    #         view2_norm_mask = tf.expand_dims(view2_norm_mask, axis=0)
    #         view2_norm_mask = tf.expand_dims(view2_norm_mask, axis=3)
    #         view2_norm_mask = tf.tile(view2_norm_mask, [batch_size, 1, 1, num_channels])
    #         view_norm_mask = tf.cast(view2_norm_mask, 'float32')
    #
    #     elif (view == 3):
    #         view3_ic = np.load('coords_correspondence/projection_forth/view3_correspondence_forth.npz')
    #         view3_ic = view3_ic.f.arr_0
    #         view3_ic = tf.cast(view3_ic, 'float32')
    #         view3_ic = tf.expand_dims(view3_ic, axis=0)
    #         view3_ic = tf.tile(view3_ic, [batch_size, 1, 1])
    #
    #         transformed_grid = tf.cast(view3_ic, 'float32')
    #
    #         view3_gp_mask = np.load('coords_correspondence/mask/view3_gp_mask.npz')
    #         view3_gp_mask = view3_gp_mask.f.arr_0
    #         view3_gp_mask = cv2.resize(view3_gp_mask, (610 / 4, 710 / 4))
    #         view3_gp_mask = tf.cast(view3_gp_mask, 'float32')
    #         view3_gp_mask = tf.expand_dims(view3_gp_mask, axis=0)
    #         view3_gp_mask = tf.expand_dims(view3_gp_mask, axis=3)
    #         view3_gp_mask = tf.tile(view3_gp_mask, [batch_size, 1, 1, num_channels])
    #         view_gp_mask = tf.cast(view3_gp_mask, 'float32')
    #
    #         view_norm_mask = np.load('coords_correspondence/norm/view_Wld_normalization_forth_mask.npz')
    #         view3_norm_mask= view_norm_mask.f.arr_0[2]
    #         view3_norm_mask = tf.cast(view3_norm_mask, 'float32')
    #         view3_norm_mask = tf.expand_dims(view3_norm_mask, axis=0)
    #         view3_norm_mask = tf.expand_dims(view3_norm_mask, axis=3)
    #         view3_norm_mask = tf.tile(view3_norm_mask, [batch_size, 1, 1, num_channels])
    #         view_norm_mask = tf.cast(view3_norm_mask, 'float32')
    #
    #
    #     x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
    #     y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
    #     x_s_flatten = tf.reshape(x_s, [-1])
    #     y_s_flatten = tf.reshape(y_s, [-1])
    #
    #     transformed_image = self._interpolate(input_shape,
    #                                             x_s_flatten,
    #                                             y_s_flatten,
    #                                             output_size)
    #
    #     transformed_image = tf.reshape(transformed_image, shape=(batch_size,
    #                                                             output_height,
    #                                                             output_width,
    #                                                             num_channels))
    #
    #     #transformed_image = tf.multiply(transformed_image, view_gp_mask)
    #     transformed_image = tf.multiply(transformed_image, view_norm_mask)
    #
    #     # # normalization:
    #     # get the sum of each channel/each image
    #     input_sum = tf.reduce_sum(input_shape, [1, 2])
    #     input_sum = tf.expand_dims(input_sum, axis=1)
    #     input_sum = tf.expand_dims(input_sum, axis=1)
    #
    #     output_sum = tf.reduce_sum(transformed_image, [1, 2])
    #     output_sum = tf.expand_dims(output_sum, axis=1)
    #     output_sum = tf.expand_dims(output_sum, axis=1)
    #
    #     amplify_times = tf.divide(input_sum, output_sum)
    #     mul_times = tf.constant([1, output_height, output_width, 1])
    #     amplify_times = tf.tile(amplify_times, mul_times)
    #
    #     # transformed_image = tf.image.resize_images(transformed_image,
    #     #                                            [output_height/4, output_width/4])
    #
    #     transformed_image_sum = tf.multiply(transformed_image, amplify_times)
    #
    #     return transformed_image_sum




class SpatialTransformer_2DTo2D_real2(Layer):
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
                 patch_num=1,
                 **kwargs):
        #self.locnet = localization_net
        self.view = view
        self.output_size = output_size
        self.patch_num = patch_num

        super(SpatialTransformer_2DTo2D_real2, self).__init__(**kwargs)


    # def build(self, input_shape):
    #    super(SpatialTransformer, self).build(input_shape)
        # self.locnet.build(input_shape)
        # self.trainable_weights = self.locnet.trainable_weights
        # self.regularizers = self.locnet.regularizers #//NOT SUER ABOUT THIS, THERE IS NO MORE SUCH PARAMETR AT self.locnet
        # self.constraints = self.locnet.constraints

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        patch_num = self.patch_num

        return (int(input_shape[0]/patch_num),
                patch_num,
                int(output_size[0]),
                int(output_size[1]),
                int(input_shape[-1]) # , 33
                )  # add the depth too.



    def call(self, inputs, mask=None):
        # view = self.view
        # num_channels = self.num_channels
        #
        # output_size = self.output_size
        # output_height = int(output_size[0])
        # output_width = int(output_size[1])
        #
        # batch_size = inputs.shape[0].value
        # height = inputs.shape[1].value
        # width = inputs.shape[2].value
        # num_channels0 = inputs.shape[3].value
        #
        # D = int(36 / 4) * 4
        # height_range = np.linspace(0, D - 1, D)
        #
        # output = tf.zeros([batch_size, output_height, output_width, 1])
        # for i in range(len(height_range)):
        #     hi = height_range[i]
        #     input_i = inputs[:, :, :, i:i + 1]
        #     self.proj_2DTo2D(view, input_i, hi)  # inputs
        #     output_i = self.proj_splat(input_i)  # inputs
        #     output = tf.concat([output, output_i], axis=-1)
        # output = output[:, :, :, 1:]
        #
        # output = tf.expand_dims(output, axis=-1)
        # num_channels = num_channels
        # output = tf.tile(output, [batch_size, 1, 1, 1, num_channels])
        #
        # return output

        view = self.view
        output = tf.placeholder(dtype=tf.float32)
        for i in range(len(view)):
            view_i = view[i]
            self.proj_2DTo2D(view_i, inputs[i:i+1, :, :, :])
            output_i = self.proj_splat(inputs[i:i+1, :, :, :]) #inputs

            if i==0:
                output = output_i
            else:
                output = tf.concat([output, output_i], axis = 0) #0
        return output



    # util functions
    def tf_static_shape(self, T):
        return T.get_shape().as_list()

    def Image2World(self, view, imgcoords):
        N = imgcoords.shape[0]
        wld_coords = []
        for i in range(N):
            imgcoords_i = imgcoords[i, :]

            Xi = imgcoords_i[0]
            Yi = imgcoords_i[1]
            Zw = imgcoords_i[2]

            XYw = proj.Image2World(view, Xi, Yi, Zw)
            wld_coords.append(XYw)
        wld_coords = np.asarray(wld_coords)
        return wld_coords

    def World2Image(self, view, wldcoords):
        N = wldcoords.shape[0]
        imgcoords = []
        for i in range(N):
            wldcoords_i = wldcoords[i, :]

            Xw = wldcoords_i[0]
            Yw = wldcoords_i[1]
            Zw = wldcoords_i[2]

            XYi = proj.World2Image(view, Xw, Yw, Zw)
            imgcoords.append(XYi)
        imgcoords = np.asarray(imgcoords)
        return imgcoords

    def proj_2DTo2D(self, view, inputs):
        w = 640
        h = 360
        W = int(640/4)
        H = int(480/4)

        # D = hi # 36/4*4

        bbox = [50, 50]  # assuming half-man plane/ 1.75/2*1000
        # bbox = [-21, 19, -25, 15]

        image_size = [h/4, w/4]

        resolution_scaler = 4 # 4 #4  # control the resolution of the project ROI mask (ground plane density map).

        ph = 1.75  # average height of a person in millimeters

        # nR, fh, fw, fdim = self.tf_static_shape(inputs)

        step_height = 100
        step_height = ph * 1000


        nR, fh, fw, fdim = inputs.get_shape().as_list()
        # self.batch_size, self.gp_x, self.gp_y, self.gp_z = nR, W, H, 1 # D = 1
        self.batch_size, self.gp_x, self.gp_y = nR, W, H # D = 1


        rsz_h = float(fh) / h/3.0
        rsz_w = float(fw) / w/3.0

        # Create voxel grid
        grid_rangeX = np.linspace(0, W - 1, W)
        grid_rangeY = np.linspace(0, H - 1, H)
        # grid_rangeZ = hi #np.linspace(0, D - 1, D)
        # grid_rangeX, grid_rangeY, grid_rangeZ = np.meshgrid(grid_rangeX, grid_rangeY, grid_rangeZ)
        grid_rangeX, grid_rangeY = np.meshgrid(grid_rangeX, grid_rangeY)

        grid_rangeX = np.reshape(grid_rangeX, [-1])
        grid_rangeY = np.reshape(grid_rangeY, [-1])
        # grid_rangeZ = np.reshape(grid_rangeZ, [-1])

        grid_rangeX = (grid_rangeX * 4 - bbox[0]*4)/resolution_scaler # 4
        # grid_rangeX = grid_rangeX * 2/resolution_scaler - bbox[0] # 4

        grid_rangeX = grid_rangeX
        grid_rangeX = np.expand_dims(grid_rangeX, 1)

        grid_rangeY = (grid_rangeY * 4 - bbox[1]*4)/resolution_scaler # 4
        # grid_rangeY = grid_rangeY * 2/resolution_scaler - bbox[1] # 4

        grid_rangeY = grid_rangeY
        grid_rangeY = np.expand_dims(grid_rangeY, 1)

        # grid_rangeZ = grid_rangeZ * step_height/1000 * np.ones(grid_rangeX.shape)
        grid_rangeZ = step_height/1000 * np.ones(grid_rangeX.shape)

        # grid_rangeZ = np.expand_dims(grid_rangeZ, 1)

        wldcoords = np.concatenate(([grid_rangeX, grid_rangeY, grid_rangeZ]), axis=1)

        if view==1:
            view = 'view1'
            view_gp_mask = np.load('coords_correspondence_DukeMTMC/mask/view1_GP_mask.npz')
            view_gp_mask = view_gp_mask.f.arr_0
        if view==2:
            view = 'view2'
            view_gp_mask = np.load('coords_correspondence_DukeMTMC/mask/view2_GP_mask.npz')
            view_gp_mask = view_gp_mask.f.arr_0
            # view_gp_mask[75:, :] = 0
            # view_gp_mask[:, 75:] = 0
        if view==3:
            view = 'view3'
            view_gp_mask = np.load('coords_correspondence_DukeMTMC/mask/view3_GP_mask.npz')
            view_gp_mask = view_gp_mask.f.arr_0
        if view==4:
            view = 'view4'
            view_gp_mask = np.load('coords_correspondence_DukeMTMC/mask/view4_GP_mask.npz')
            view_gp_mask = view_gp_mask.f.arr_0
            # view_gp_mask[:50, :] = 0
            # view_gp_mask[:, 75:] = 0

        # gp view mask:
        # view_gp_mask = view_gp_mask.f.arr_0
        view_gp_mask = cv2.resize(view_gp_mask, (W, H))
        view_gp_mask = tf.cast(view_gp_mask, 'float32')
        view_gp_mask = tf.expand_dims(view_gp_mask, axis=0)
        view_gp_mask = tf.expand_dims(view_gp_mask, axis=1)
        view_gp_mask = tf.expand_dims(view_gp_mask, axis=-1)
        batch_size = nR
        num_channels = fdim  ###### no to add the depth dim
        view_gp_mask = tf.tile(view_gp_mask, [int(self.batch_size / self.patch_num),
                                              self.patch_num, 1, 1, num_channels])
        view_gp_mask = tf.cast(view_gp_mask, 'float32')
        self.view_gp_mask  = view_gp_mask

        # view1_ic = self.World2Image('view1', wldcoords)
        # view2_ic = self.World2Image('view2', wldcoords)
        # view3_ic = self.World2Image('view3', wldcoords)
        view_ic = self.World2Image(view, wldcoords)

        # view1_ic = np.transpose(view1_ic)
        # view2_ic = np.transpose(view2_ic)
        # view3_ic = np.transpose(view3_ic)
        view_ic = np.transpose(view_ic)


        # # normalization:
        # view1_ic[0:1, :] = view1_ic[0:1, :] * rsz_w
        # view1_ic[1:2, :] = view1_ic[1:2, :] * rsz_h
        #
        # view2_ic[0:1, :] = view2_ic[0:1, :] * rsz_w
        # view2_ic[1:2, :] = view2_ic[1:2, :] * rsz_h
        #
        # view3_ic[0:1, :] = view3_ic[0:1, :] * rsz_w
        # view3_ic[1:2, :] = view3_ic[1:2, :] * rsz_h

        view_ic[0:1, :] = view_ic[0:1, :] * rsz_w
        view_ic[1:2, :] = view_ic[1:2, :] * rsz_h
        view_ic[2:3, :] = view_ic[2:3, :] /(step_height/1000.0)

        # net.proj_view = np.concatenate(
        #     [view1_ic[0:1, :], view2_ic[0:1, :], view3_ic[0:1, :],
        #      view1_ic[1:2, :], view2_ic[1:2, :], view3_ic[1:2, :],
        #      view1_ic[2:3, :], view2_ic[2:3, :], view3_ic[2:3, :]],
        #     axis=0)
        self.proj_view = np.concatenate(
            [view_ic[0:1, :], view_ic[1:2, :], view_ic[2:3, :]],axis=0)

    def proj_splat(self,  inputs):
        with tf.variable_scope('ProjSplat'):
            nR, fh, fw, fdim = self.tf_static_shape(inputs)

            nV = self.proj_view.shape[1]

            im_p = self.proj_view
            im_x, im_y, im_z = im_p[::3, :], im_p[1::3, :], im_p[2::3, :]

            im_x = tf.constant(im_x, dtype='float32')
            im_y = tf.constant(im_y, dtype='float32')
            im_z = tf.constant(im_z, dtype='float32')
            self.im_p, self.im_x, self.im_y, self.im_z = im_p, im_x, im_y, im_z

            # im_p = tf.constant(self.proj_view, dtype='float32')
            # im_x, im_y, im_z = self[::3, :], im_p[1::3, :], im_p[2::3, :]
            # self.im_p, self.im_x, self.im_y, self.im_z = im_p, im_x, im_y, im_z

            # Bilinear interpolation
            with tf.name_scope('BilinearInterp'):
                im_x = tf.clip_by_value(im_x, 0, fw - 1)
                im_y = tf.clip_by_value(im_y, 0, fh - 1)

                im_x0 = tf.cast(tf.floor(im_x), 'int32')
                im_x1 = im_x0 + 1
                im_x1 = tf.clip_by_value(im_x1, 0, fw - 1)

                im_y0 = tf.cast(tf.floor(im_y), 'int32')
                im_y1 = im_y0 + 1
                im_y1 = tf.clip_by_value(im_y1, 0, fh - 1)

                im_x0_f, im_x1_f = tf.to_float(im_x0), tf.to_float(im_x1)
                im_y0_f, im_y1_f = tf.to_float(im_y0), tf.to_float(im_y1)

                ind_grid = tf.range(0, nR)
                ind_grid = tf.expand_dims(ind_grid, 1)
                im_ind = tf.tile(ind_grid, [1, nV])

                def _get_gather_inds(x, y):
                    return tf.reshape(tf.stack([im_ind, y, x], axis=2), [-1, 3])

                # Gather  values
                Ia = tf.gather_nd(inputs, _get_gather_inds(im_x0, im_y0))
                Ib = tf.gather_nd(inputs, _get_gather_inds(im_x0, im_y1))
                Ic = tf.gather_nd(inputs, _get_gather_inds(im_x1, im_y0))
                Id = tf.gather_nd(inputs, _get_gather_inds(im_x1, im_y1))

                # Calculate bilinear weights
                wa = (im_x1_f - im_x) * (im_y1_f - im_y)
                wb = (im_x1_f - im_x) * (im_y - im_y0_f)
                wc = (im_x - im_x0_f) * (im_y1_f - im_y)
                wd = (im_x - im_x0_f) * (im_y - im_y0_f)
                wa, wb = tf.reshape(wa, [-1, 1]), tf.reshape(wb, [-1, 1])
                wc, wd = tf.reshape(wc, [-1, 1]), tf.reshape(wd, [-1, 1])
                self.wa, self.wb, self.wc, self.wd = wa, wb, wc, wd
                self.Ia, self.Ib, self.Ic, self.Id = Ia, Ib, Ic, Id
                Ibilin = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

            with tf.name_scope('AppendDepth'):
                # Concatenate depth value along ray to feature
                # Ibilin = tf.concat(
                #     [Ibilin, tf.reshape(im_z, [nV * nR, 1])], axis=1)
                # fdim = Ibilin.get_shape().as_list()[-1]
                Ibilin = tf.reshape(Ibilin, [int(self.batch_size/self.patch_num), self.patch_num,
                                             self.gp_y, self.gp_x, fdim]) # fdim

                # self.Ibilin = Ibilin

                # add a mask:
                self.Ibilin = tf.multiply(Ibilin, self.view_gp_mask)

                # self.Ibilin = tf.transpose(self.Ibilin, [0, 2, 1, 3, 4])
        return self.Ibilin

    #
    # def _repeat(self, x, num_repeats):
    #     ones = tf.ones((1, num_repeats), dtype='int32')
    #     x = tf.reshape(x, shape=(-1,1))
    #     x = tf.matmul(x, ones)
    #     return tf.reshape(x, [-1])
    #
    # def _interpolate(self, image, x, y, output_size):
    #     batch_size = tf.shape(image)[0]
    #     height = tf.shape(image)[1]
    #     width = tf.shape(image)[2]
    #     num_channels = tf.shape(image)[3]
    #     # batch_size = image.shape[0].value
    #     # height = image.shape[1].value
    #     # width = image.shape[2].value
    #     # num_channels = image.shape[3].value
    #
    #     x = tf.cast(x , dtype='float32')
    #     y = tf.cast(y , dtype='float32')
    #
    #     height_float = tf.cast(height, dtype='float32')
    #     width_float = tf.cast(width, dtype='float32')
    #
    #     output_height = output_size[0]
    #     output_width  = output_size[1]
    #
    #     x = .5*(x + 1.0)*(width_float)
    #     y = .5*(y + 1.0)*(height_float)
    #
    #     x0 = tf.cast(tf.floor(x), 'int32')
    #     x1 = x0 + 1
    #     y0 = tf.cast(tf.floor(y), 'int32')
    #     y1 = y0 + 1
    #
    #     max_y = tf.cast(height - 1, dtype='int32')
    #     max_x = tf.cast(width - 1,  dtype='int32')
    #     zero = tf.zeros([], dtype='int32')
    #
    #     x0 = tf.clip_by_value(x0, zero, max_x)
    #     x1 = tf.clip_by_value(x1, zero, max_x)
    #     y0 = tf.clip_by_value(y0, zero, max_y)
    #     y1 = tf.clip_by_value(y1, zero, max_y)
    #
    #     flat_image_dimensions = width*height
    #     pixels_batch = tf.range(batch_size)*flat_image_dimensions
    #     flat_output_dimensions = output_height*output_width
    #     base = self._repeat(pixels_batch, flat_output_dimensions)
    #     base_y0 = base + y0*width
    #     base_y1 = base + y1*width
    #     indices_a = base_y0 + x0
    #     indices_b = base_y1 + x0
    #     indices_c = base_y0 + x1
    #     indices_d = base_y1 + x1
    #
    #     flat_image = tf.reshape(image, shape=(-1, num_channels))
    #     flat_image = tf.cast(flat_image, dtype='float32')
    #     pixel_values_a = tf.gather(flat_image, indices_a)
    #     pixel_values_b = tf.gather(flat_image, indices_b)
    #     pixel_values_c = tf.gather(flat_image, indices_c)
    #     pixel_values_d = tf.gather(flat_image, indices_d)
    #
    #     x0 = tf.cast(x0, 'float32')
    #     x1 = tf.cast(x1, 'float32')
    #     y0 = tf.cast(y0, 'float32')
    #     y1 = tf.cast(y1, 'float32')
    #
    #     area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
    #     area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
    #     area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
    #     area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
    #     output = tf.add_n([area_a*pixel_values_a,
    #                        area_b*pixel_values_b,
    #                        area_c*pixel_values_c,
    #                        area_d*pixel_values_d])
    #     return output
    #
    # def _meshgrid(self, height, width):
    #     x_linspace = tf.linspace(-1., 1., width)
    #     y_linspace = tf.linspace(-1., 1., height)
    #
    #     # x_linspace = tf.linspace(0., width-1, width)
    #     # y_linspace = tf.linspace(0., height-1, height)
    #
    #     x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
    #     x_coordinates = tf.reshape(x_coordinates, [-1])
    #     y_coordinates = tf.reshape(y_coordinates, [-1])
    #     ones = tf.ones_like(x_coordinates)
    #     indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
    #     return indices_grid
    #
    # def _transform(self, view, input_shape, output_size):
    #     batch_size = tf.shape(input_shape)[0]
    #     height = tf.shape(input_shape)[1]
    #     width = tf.shape(input_shape)[2]
    #     num_channels = tf.shape(input_shape)[3]
    #
    #     # batch_size = input_shape.shape[0].value
    #     # height = input_shape.shape[1].value
    #     # width = input_shape.shape[2].value
    #     # num_channels = input_shape.shape[3].value
    #
    #     # B = tf.shape(input_shape)[0]
    #     # H = tf.shape(input_shape)[1]
    #     # W = tf.shape(input_shape)[2]
    #     # C = tf.shape(input_shape)[3]
    #     # n_fc = 6
    #     # W_fc1 = tf.Variable(tf.zeros([H * W * C, n_fc]), name='W_fc1')
    #     # b_fc1 = tf.Variable(initial_value = affine_transformation, name='b_fc1')
    #     # affine_transformation = tf.matmul(tf.zeros([B, H * W * C]), W_fc1) + b_fc1
    #
    #     # affine_transformation = tf.reshape(affine_transformation, shape=(batch_size,2,3))
    #     # affine_transformation = tf.reshape(affine_transformation, (-1, 2, 3))
    #     # affine_transformation = tf.cast(affine_transformation, 'float32')
    #
    #     width = tf.cast(width, dtype='float32')
    #     height = tf.cast(height, dtype='float32')
    #     output_height = output_size[0]
    #     output_width = output_size[1]
    #     indices_grid = self._meshgrid(output_height, output_width)
    #     indices_grid = tf.expand_dims(indices_grid, 0)
    #     indices_grid = tf.reshape(indices_grid, [-1]) # flatten?
    #
    #     indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
    #     indices_grid = tf.reshape(indices_grid, (batch_size, 3, -1))
    #
    #     # transformed_grid = tf.matmul(affine_transformation, indices_grid)
    #     if(view == 1):
    #         view1_ic = np.load('coords_correspondence/projection_forth/view1_correspondence_forth.npz')
    #         view1_ic = view1_ic.f.arr_0
    #         view1_ic = tf.cast(view1_ic, 'float32')
    #         view1_ic = tf.expand_dims(view1_ic, axis=0)
    #         view1_ic = tf.tile(view1_ic, [batch_size, 1, 1])
    #
    #         transformed_grid = tf.cast(view1_ic, 'float32')
    #
    #         view1_gp_mask = np.load('coords_correspondence/mask/view1_gp_mask.npz')
    #         view1_gp_mask = view1_gp_mask.f.arr_0
    #         view1_gp_mask = cv2.resize(view1_gp_mask, (610 / 4, 710 / 4))
    #         view1_gp_mask = tf.cast(view1_gp_mask, 'float32')
    #         view1_gp_mask = tf.expand_dims(view1_gp_mask, axis=0)
    #         view1_gp_mask = tf.expand_dims(view1_gp_mask, axis=3)
    #         view1_gp_mask = tf.tile(view1_gp_mask, [batch_size, 1, 1, num_channels])
    #         view_gp_mask = tf.cast(view1_gp_mask, 'float32')
    #
    #         view_norm_mask = np.load('coords_correspondence/norm/view_Wld_normalization_forth_mask.npz')
    #         view1_norm_mask= view_norm_mask.f.arr_0[0]
    #         view1_norm_mask = tf.cast(view1_norm_mask, 'float32')
    #         view1_norm_mask = tf.expand_dims(view1_norm_mask, axis=0)
    #         view1_norm_mask = tf.expand_dims(view1_norm_mask, axis=3)
    #         view1_norm_mask = tf.tile(view1_norm_mask, [batch_size, 1, 1, num_channels])
    #         view_norm_mask = tf.cast(view1_norm_mask, 'float32')
    #
    #     elif (view == 2):
    #         view2_ic = np.load('coords_correspondence/projection_forth/view2_correspondence_forth.npz')
    #         view2_ic = view2_ic.f.arr_0
    #         view2_ic = tf.cast(view2_ic, 'float32')
    #         view2_ic = tf.expand_dims(view2_ic, axis=0)
    #         view2_ic = tf.tile(view2_ic, [batch_size, 1, 1])
    #
    #         transformed_grid = tf.cast(view2_ic, 'float32')
    #
    #         view2_gp_mask = np.load('coords_correspondence/mask/view2_gp_mask.npz')
    #         view2_gp_mask = view2_gp_mask.f.arr_0
    #         view2_gp_mask = cv2.resize(view2_gp_mask, (610 / 4, 710 / 4))
    #         view2_gp_mask = tf.cast(view2_gp_mask, 'float32')
    #         view2_gp_mask = tf.expand_dims(view2_gp_mask, axis=0)
    #         view2_gp_mask = tf.expand_dims(view2_gp_mask, axis=3)
    #         view2_gp_mask = tf.tile(view2_gp_mask, [batch_size, 1, 1, num_channels])
    #         view_gp_mask = tf.cast(view2_gp_mask, 'float32')
    #
    #         view_norm_mask = np.load('coords_correspondence/norm/view_Wld_normalization_forth_mask.npz')
    #         view2_norm_mask= view_norm_mask.f.arr_0[1]
    #         view2_norm_mask = tf.cast(view2_norm_mask, 'float32')
    #         view2_norm_mask = tf.expand_dims(view2_norm_mask, axis=0)
    #         view2_norm_mask = tf.expand_dims(view2_norm_mask, axis=3)
    #         view2_norm_mask = tf.tile(view2_norm_mask, [batch_size, 1, 1, num_channels])
    #         view_norm_mask = tf.cast(view2_norm_mask, 'float32')
    #
    #     elif (view == 3):
    #         view3_ic = np.load('coords_correspondence/projection_forth/view3_correspondence_forth.npz')
    #         view3_ic = view3_ic.f.arr_0
    #         view3_ic = tf.cast(view3_ic, 'float32')
    #         view3_ic = tf.expand_dims(view3_ic, axis=0)
    #         view3_ic = tf.tile(view3_ic, [batch_size, 1, 1])
    #
    #         transformed_grid = tf.cast(view3_ic, 'float32')
    #
    #         view3_gp_mask = np.load('coords_correspondence/mask/view3_gp_mask.npz')
    #         view3_gp_mask = view3_gp_mask.f.arr_0
    #         view3_gp_mask = cv2.resize(view3_gp_mask, (610 / 4, 710 / 4))
    #         view3_gp_mask = tf.cast(view3_gp_mask, 'float32')
    #         view3_gp_mask = tf.expand_dims(view3_gp_mask, axis=0)
    #         view3_gp_mask = tf.expand_dims(view3_gp_mask, axis=3)
    #         view3_gp_mask = tf.tile(view3_gp_mask, [batch_size, 1, 1, num_channels])
    #         view_gp_mask = tf.cast(view3_gp_mask, 'float32')
    #
    #         view_norm_mask = np.load('coords_correspondence/norm/view_Wld_normalization_forth_mask.npz')
    #         view3_norm_mask= view_norm_mask.f.arr_0[2]
    #         view3_norm_mask = tf.cast(view3_norm_mask, 'float32')
    #         view3_norm_mask = tf.expand_dims(view3_norm_mask, axis=0)
    #         view3_norm_mask = tf.expand_dims(view3_norm_mask, axis=3)
    #         view3_norm_mask = tf.tile(view3_norm_mask, [batch_size, 1, 1, num_channels])
    #         view_norm_mask = tf.cast(view3_norm_mask, 'float32')
    #
    #
    #     x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
    #     y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
    #     x_s_flatten = tf.reshape(x_s, [-1])
    #     y_s_flatten = tf.reshape(y_s, [-1])
    #
    #     transformed_image = self._interpolate(input_shape,
    #                                             x_s_flatten,
    #                                             y_s_flatten,
    #                                             output_size)
    #
    #     transformed_image = tf.reshape(transformed_image, shape=(batch_size,
    #                                                             output_height,
    #                                                             output_width,
    #                                                             num_channels))
    #
    #     #transformed_image = tf.multiply(transformed_image, view_gp_mask)
    #     transformed_image = tf.multiply(transformed_image, view_norm_mask)
    #
    #     # # normalization:
    #     # get the sum of each channel/each image
    #     input_sum = tf.reduce_sum(input_shape, [1, 2])
    #     input_sum = tf.expand_dims(input_sum, axis=1)
    #     input_sum = tf.expand_dims(input_sum, axis=1)
    #
    #     output_sum = tf.reduce_sum(transformed_image, [1, 2])
    #     output_sum = tf.expand_dims(output_sum, axis=1)
    #     output_sum = tf.expand_dims(output_sum, axis=1)
    #
    #     amplify_times = tf.divide(input_sum, output_sum)
    #     mul_times = tf.constant([1, output_height, output_width, 1])
    #     amplify_times = tf.tile(amplify_times, mul_times)
    #
    #     # transformed_image = tf.image.resize_images(transformed_image,
    #     #                                            [output_height/4, output_width/4])
    #
    #     transformed_image_sum = tf.multiply(transformed_image, amplify_times)
    #
    #     return transformed_image_sum




class SpatialTransformer_2DTo2D_real2_proj(Layer):
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
                 patch_num=1,
                 **kwargs):
        #self.locnet = localization_net
        self.view = view
        self.output_size = output_size
        self.patch_num = patch_num

        super(SpatialTransformer_2DTo2D_real2_proj, self).__init__(**kwargs)


    # def build(self, input_shape):
    #    super(SpatialTransformer, self).build(input_shape)
        # self.locnet.build(input_shape)
        # self.trainable_weights = self.locnet.trainable_weights
        # self.regularizers = self.locnet.regularizers #//NOT SUER ABOUT THIS, THERE IS NO MORE SUCH PARAMETR AT self.locnet
        # self.constraints = self.locnet.constraints

    def compute_output_shape(self, input_shape):
        output_size = self.output_size
        patch_num = self.patch_num

        return (1,
                4,
                int(output_size[0]),
                int(output_size[1]),
                int(input_shape[-1]) # , 33
                )  # add the depth too.



    def call(self, inputs, mask=None):
        # view = self.view
        # num_channels = self.num_channels
        #
        # output_size = self.output_size
        # output_height = int(output_size[0])
        # output_width = int(output_size[1])
        #
        # batch_size = inputs.shape[0].value
        # height = inputs.shape[1].value
        # width = inputs.shape[2].value
        # num_channels0 = inputs.shape[3].value
        #
        # D = int(36 / 4) * 4
        # height_range = np.linspace(0, D - 1, D)
        #
        # output = tf.zeros([batch_size, output_height, output_width, 1])
        # for i in range(len(height_range)):
        #     hi = height_range[i]
        #     input_i = inputs[:, :, :, i:i + 1]
        #     self.proj_2DTo2D(view, input_i, hi)  # inputs
        #     output_i = self.proj_splat(input_i)  # inputs
        #     output = tf.concat([output, output_i], axis=-1)
        # output = output[:, :, :, 1:]
        #
        # output = tf.expand_dims(output, axis=-1)
        # num_channels = num_channels
        # output = tf.tile(output, [batch_size, 1, 1, 1, num_channels])
        #
        # return output

        view = self.view
        output = tf.placeholder(dtype=tf.float32)
        for i in range(len(view)):
            view_i = view[i]
            self.proj_2DTo2D(view_i, inputs[i:i+1, :, :, :])
            output_i = self.proj_splat(inputs[i:i+1, :, :, :]) #inputs

            if i==0:
                output = output_i
            else:
                output = tf.concat([output, output_i], axis = 1) #0
        return output



    # util functions
    def tf_static_shape(self, T):
        return T.get_shape().as_list()

    def Image2World(self, view, imgcoords):
        N = imgcoords.shape[0]
        wld_coords = []
        for i in range(N):
            imgcoords_i = imgcoords[i, :]

            Xi = imgcoords_i[0]
            Yi = imgcoords_i[1]
            Zw = imgcoords_i[2]

            XYw = proj.Image2World(view, Xi, Yi, Zw)
            wld_coords.append(XYw)
        wld_coords = np.asarray(wld_coords)
        return wld_coords

    def World2Image(self, view, wldcoords):
        N = wldcoords.shape[0]
        imgcoords = []
        for i in range(N):
            wldcoords_i = wldcoords[i, :]

            Xw = wldcoords_i[0]
            Yw = wldcoords_i[1]
            Zw = wldcoords_i[2]

            XYi = proj.World2Image(view, Xw, Yw, Zw)
            imgcoords.append(XYi)
        imgcoords = np.asarray(imgcoords)
        return imgcoords

    def proj_2DTo2D(self, view, inputs):
        w = 640
        h = 360
        W = int(640/2)
        H = int(480/2)

        # D = hi # 36/4*4

        bbox = [50, 50]  # assuming half-man plane/ 1.75/2*1000
        # bbox = [-21, 19, -25, 15]

        image_size = [h/4, w/4]

        resolution_scaler = 4 # 4 #4  # control the resolution of the project ROI mask (ground plane density map).

        ph = 1.75  # average height of a person in millimeters

        # nR, fh, fw, fdim = self.tf_static_shape(inputs)

        step_height = 100
        step_height = ph * 1000


        nR, fh, fw, fdim = inputs.get_shape().as_list()
        # self.batch_size, self.gp_x, self.gp_y, self.gp_z = nR, W, H, 1 # D = 1
        self.batch_size, self.gp_x, self.gp_y = nR, W, H # D = 1


        rsz_h = float(fh) / (h*3)
        rsz_w = float(fw) / (w*3)

        # Create voxel grid
        grid_rangeX = np.linspace(0, W - 1, W)
        grid_rangeY = np.linspace(0, H - 1, H)
        # grid_rangeZ = hi #np.linspace(0, D - 1, D)
        # grid_rangeX, grid_rangeY, grid_rangeZ = np.meshgrid(grid_rangeX, grid_rangeY, grid_rangeZ)
        grid_rangeX, grid_rangeY = np.meshgrid(grid_rangeX, grid_rangeY)

        grid_rangeX = np.reshape(grid_rangeX, [-1])
        grid_rangeY = np.reshape(grid_rangeY, [-1])
        # grid_rangeZ = np.reshape(grid_rangeZ, [-1])

        grid_rangeX = (grid_rangeX*2  - bbox[0]*4)/resolution_scaler # 4
        grid_rangeX = grid_rangeX
        grid_rangeX = np.expand_dims(grid_rangeX, 1)

        grid_rangeY = (grid_rangeY*2  - bbox[1]*4)/resolution_scaler # 4
        grid_rangeY = grid_rangeY
        grid_rangeY = np.expand_dims(grid_rangeY, 1)

        # grid_rangeZ = grid_rangeZ * step_height/1000 * np.ones(grid_rangeX.shape)
        grid_rangeZ = step_height/1000 * np.ones(grid_rangeX.shape)

        # grid_rangeZ = np.expand_dims(grid_rangeZ, 1)

        wldcoords = np.concatenate(([grid_rangeX, grid_rangeY, grid_rangeZ]), axis=1)

        if view==1:
            view = 'view1'
            view_gp_mask = np.load('coords_correspondence_DukeMTMC/mask/view1_GP_mask.npz')
            view_gp_mask = view_gp_mask.f.arr_0
        if view==2:
            view = 'view2'
            view_gp_mask = np.load('coords_correspondence_DukeMTMC/mask/view2_GP_mask.npz')
            view_gp_mask = view_gp_mask.f.arr_0
            view_gp_mask[75:, :] = 0
            view_gp_mask[:, 75:] = 0
        if view==3:
            view = 'view3'
            view_gp_mask = np.load('coords_correspondence_DukeMTMC/mask/view3_GP_mask.npz')
            view_gp_mask = view_gp_mask.f.arr_0
        if view==4:
            view = 'view4'
            view_gp_mask = np.load('coords_correspondence_DukeMTMC/mask/view4_GP_mask.npz')
            view_gp_mask = view_gp_mask.f.arr_0
            view_gp_mask[:50, :] = 0
            view_gp_mask[:, 75:] = 0

        #
        # gp view mask:
        # view_gp_mask = view_gp_mask.f.arr_0
        view_gp_mask = cv2.resize(view_gp_mask, (W, H))
        view_gp_mask = tf.cast(view_gp_mask, 'float32')
        view_gp_mask = tf.expand_dims(view_gp_mask, axis=0)
        view_gp_mask = tf.expand_dims(view_gp_mask, axis=1)
        view_gp_mask = tf.expand_dims(view_gp_mask, axis=-1)
        batch_size = nR
        num_channels = fdim ###### no to add the depth dim
        view_gp_mask = tf.tile(view_gp_mask, [int(self.batch_size/self.patch_num),
                                              self.patch_num, 1, 1, num_channels])
        view_gp_mask = tf.cast(view_gp_mask, 'float32')
        self.view_gp_mask = view_gp_mask

        # view1_ic = self.World2Image('view1', wldcoords)
        # view2_ic = self.World2Image('view2', wldcoords)
        # view3_ic = self.World2Image('view3', wldcoords)
        view_ic = self.World2Image(view, wldcoords)

        # view1_ic = np.transpose(view1_ic)
        # view2_ic = np.transpose(view2_ic)
        # view3_ic = np.transpose(view3_ic)
        view_ic = np.transpose(view_ic)


        # # normalization:
        # view1_ic[0:1, :] = view1_ic[0:1, :] * rsz_w
        # view1_ic[1:2, :] = view1_ic[1:2, :] * rsz_h
        #
        # view2_ic[0:1, :] = view2_ic[0:1, :] * rsz_w
        # view2_ic[1:2, :] = view2_ic[1:2, :] * rsz_h
        #
        # view3_ic[0:1, :] = view3_ic[0:1, :] * rsz_w
        # view3_ic[1:2, :] = view3_ic[1:2, :] * rsz_h

        view_ic[0:1, :] = view_ic[0:1, :] * rsz_w
        view_ic[1:2, :] = view_ic[1:2, :] * rsz_h
        view_ic[2:3, :] = view_ic[2:3, :] /(step_height/1000.0)

        # net.proj_view = np.concatenate(
        #     [view1_ic[0:1, :], view2_ic[0:1, :], view3_ic[0:1, :],
        #      view1_ic[1:2, :], view2_ic[1:2, :], view3_ic[1:2, :],
        #      view1_ic[2:3, :], view2_ic[2:3, :], view3_ic[2:3, :]],
        #     axis=0)
        self.proj_view = np.concatenate(
            [view_ic[0:1, :], view_ic[1:2, :], view_ic[2:3, :]],axis=0)

    def proj_splat(self,  inputs):
        with tf.variable_scope('ProjSplat'):
            nR, fh, fw, fdim = self.tf_static_shape(inputs)

            nV = self.proj_view.shape[1]

            im_p = self.proj_view
            im_x, im_y, im_z = im_p[::3, :], im_p[1::3, :], im_p[2::3, :]

            im_x = tf.constant(im_x, dtype='float32')
            im_y = tf.constant(im_y, dtype='float32')
            im_z = tf.constant(im_z, dtype='float32')
            self.im_p, self.im_x, self.im_y, self.im_z = im_p, im_x, im_y, im_z

            # im_p = tf.constant(self.proj_view, dtype='float32')
            # im_x, im_y, im_z = self[::3, :], im_p[1::3, :], im_p[2::3, :]
            # self.im_p, self.im_x, self.im_y, self.im_z = im_p, im_x, im_y, im_z

            # Bilinear interpolation
            with tf.name_scope('BilinearInterp'):
                im_x = tf.clip_by_value(im_x, 0, fw - 1)
                im_y = tf.clip_by_value(im_y, 0, fh - 1)

                im_x0 = tf.cast(tf.floor(im_x), 'int32')
                im_x1 = im_x0 + 1
                im_x1 = tf.clip_by_value(im_x1, 0, fw - 1)

                im_y0 = tf.cast(tf.floor(im_y), 'int32')
                im_y1 = im_y0 + 1
                im_y1 = tf.clip_by_value(im_y1, 0, fh - 1)

                im_x0_f, im_x1_f = tf.to_float(im_x0), tf.to_float(im_x1)
                im_y0_f, im_y1_f = tf.to_float(im_y0), tf.to_float(im_y1)

                ind_grid = tf.range(0, nR)
                ind_grid = tf.expand_dims(ind_grid, 1)
                im_ind = tf.tile(ind_grid, [1, nV])

                def _get_gather_inds(x, y):
                    return tf.reshape(tf.stack([im_ind, y, x], axis=2), [-1, 3])

                # Gather  values
                Ia = tf.gather_nd(inputs, _get_gather_inds(im_x0, im_y0))
                Ib = tf.gather_nd(inputs, _get_gather_inds(im_x0, im_y1))
                Ic = tf.gather_nd(inputs, _get_gather_inds(im_x1, im_y0))
                Id = tf.gather_nd(inputs, _get_gather_inds(im_x1, im_y1))

                # Calculate bilinear weights
                wa = (im_x1_f - im_x) * (im_y1_f - im_y)
                wb = (im_x1_f - im_x) * (im_y - im_y0_f)
                wc = (im_x - im_x0_f) * (im_y1_f - im_y)
                wd = (im_x - im_x0_f) * (im_y - im_y0_f)
                wa, wb = tf.reshape(wa, [-1, 1]), tf.reshape(wb, [-1, 1])
                wc, wd = tf.reshape(wc, [-1, 1]), tf.reshape(wd, [-1, 1])
                self.wa, self.wb, self.wc, self.wd = wa, wb, wc, wd
                self.Ia, self.Ib, self.Ic, self.Id = Ia, Ib, Ic, Id
                Ibilin = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])

            with tf.name_scope('AppendDepth'):
                # Concatenate depth value along ray to feature
                # Ibilin = tf.concat(
                #     [Ibilin, tf.reshape(im_z, [nV * nR, 1])], axis=1)
                # fdim = Ibilin.get_shape().as_list()[-1]
                Ibilin = tf.reshape(Ibilin, [int(self.batch_size/self.patch_num), self.patch_num,
                                             self.gp_y, self.gp_x, fdim]) # fdim

                # self.Ibilin = Ibilin

                # add a mask:
                self.Ibilin = tf.multiply(Ibilin, self.view_gp_mask)

                # self.Ibilin = tf.transpose(self.Ibilin, [0, 2, 1, 3, 4])
        return self.Ibilin















































    #
    # def _repeat(self, x, num_repeats):
    #     ones = tf.ones((1, num_repeats), dtype='int32')
    #     x = tf.reshape(x, shape=(-1,1))
    #     x = tf.matmul(x, ones)
    #     return tf.reshape(x, [-1])
    #
    # def _interpolate(self, image, x, y, output_size):
    #     batch_size = tf.shape(image)[0]
    #     height = tf.shape(image)[1]
    #     width = tf.shape(image)[2]
    #     num_channels = tf.shape(image)[3]
    #     # batch_size = image.shape[0].value
    #     # height = image.shape[1].value
    #     # width = image.shape[2].value
    #     # num_channels = image.shape[3].value
    #
    #     x = tf.cast(x , dtype='float32')
    #     y = tf.cast(y , dtype='float32')
    #
    #     height_float = tf.cast(height, dtype='float32')
    #     width_float = tf.cast(width, dtype='float32')
    #
    #     output_height = output_size[0]
    #     output_width  = output_size[1]
    #
    #     x = .5*(x + 1.0)*(width_float)
    #     y = .5*(y + 1.0)*(height_float)
    #
    #     x0 = tf.cast(tf.floor(x), 'int32')
    #     x1 = x0 + 1
    #     y0 = tf.cast(tf.floor(y), 'int32')
    #     y1 = y0 + 1
    #
    #     max_y = tf.cast(height - 1, dtype='int32')
    #     max_x = tf.cast(width - 1,  dtype='int32')
    #     zero = tf.zeros([], dtype='int32')
    #
    #     x0 = tf.clip_by_value(x0, zero, max_x)
    #     x1 = tf.clip_by_value(x1, zero, max_x)
    #     y0 = tf.clip_by_value(y0, zero, max_y)
    #     y1 = tf.clip_by_value(y1, zero, max_y)
    #
    #     flat_image_dimensions = width*height
    #     pixels_batch = tf.range(batch_size)*flat_image_dimensions
    #     flat_output_dimensions = output_height*output_width
    #     base = self._repeat(pixels_batch, flat_output_dimensions)
    #     base_y0 = base + y0*width
    #     base_y1 = base + y1*width
    #     indices_a = base_y0 + x0
    #     indices_b = base_y1 + x0
    #     indices_c = base_y0 + x1
    #     indices_d = base_y1 + x1
    #
    #     flat_image = tf.reshape(image, shape=(-1, num_channels))
    #     flat_image = tf.cast(flat_image, dtype='float32')
    #     pixel_values_a = tf.gather(flat_image, indices_a)
    #     pixel_values_b = tf.gather(flat_image, indices_b)
    #     pixel_values_c = tf.gather(flat_image, indices_c)
    #     pixel_values_d = tf.gather(flat_image, indices_d)
    #
    #     x0 = tf.cast(x0, 'float32')
    #     x1 = tf.cast(x1, 'float32')
    #     y0 = tf.cast(y0, 'float32')
    #     y1 = tf.cast(y1, 'float32')
    #
    #     area_a = tf.expand_dims(((x1 - x) * (y1 - y)), 1)
    #     area_b = tf.expand_dims(((x1 - x) * (y - y0)), 1)
    #     area_c = tf.expand_dims(((x - x0) * (y1 - y)), 1)
    #     area_d = tf.expand_dims(((x - x0) * (y - y0)), 1)
    #     output = tf.add_n([area_a*pixel_values_a,
    #                        area_b*pixel_values_b,
    #                        area_c*pixel_values_c,
    #                        area_d*pixel_values_d])
    #     return output
    #
    # def _meshgrid(self, height, width):
    #     x_linspace = tf.linspace(-1., 1., width)
    #     y_linspace = tf.linspace(-1., 1., height)
    #
    #     # x_linspace = tf.linspace(0., width-1, width)
    #     # y_linspace = tf.linspace(0., height-1, height)
    #
    #     x_coordinates, y_coordinates = tf.meshgrid(x_linspace, y_linspace)
    #     x_coordinates = tf.reshape(x_coordinates, [-1])
    #     y_coordinates = tf.reshape(y_coordinates, [-1])
    #     ones = tf.ones_like(x_coordinates)
    #     indices_grid = tf.concat([x_coordinates, y_coordinates, ones], 0)
    #     return indices_grid
    #
    # def _transform(self, view, input_shape, output_size):
    #     batch_size = tf.shape(input_shape)[0]
    #     height = tf.shape(input_shape)[1]
    #     width = tf.shape(input_shape)[2]
    #     num_channels = tf.shape(input_shape)[3]
    #
    #     # batch_size = input_shape.shape[0].value
    #     # height = input_shape.shape[1].value
    #     # width = input_shape.shape[2].value
    #     # num_channels = input_shape.shape[3].value
    #
    #     # B = tf.shape(input_shape)[0]
    #     # H = tf.shape(input_shape)[1]
    #     # W = tf.shape(input_shape)[2]
    #     # C = tf.shape(input_shape)[3]
    #     # n_fc = 6
    #     # W_fc1 = tf.Variable(tf.zeros([H * W * C, n_fc]), name='W_fc1')
    #     # b_fc1 = tf.Variable(initial_value = affine_transformation, name='b_fc1')
    #     # affine_transformation = tf.matmul(tf.zeros([B, H * W * C]), W_fc1) + b_fc1
    #
    #     # affine_transformation = tf.reshape(affine_transformation, shape=(batch_size,2,3))
    #     # affine_transformation = tf.reshape(affine_transformation, (-1, 2, 3))
    #     # affine_transformation = tf.cast(affine_transformation, 'float32')
    #
    #     width = tf.cast(width, dtype='float32')
    #     height = tf.cast(height, dtype='float32')
    #     output_height = output_size[0]
    #     output_width = output_size[1]
    #     indices_grid = self._meshgrid(output_height, output_width)
    #     indices_grid = tf.expand_dims(indices_grid, 0)
    #     indices_grid = tf.reshape(indices_grid, [-1]) # flatten?
    #
    #     indices_grid = tf.tile(indices_grid, tf.stack([batch_size]))
    #     indices_grid = tf.reshape(indices_grid, (batch_size, 3, -1))
    #
    #     # transformed_grid = tf.matmul(affine_transformation, indices_grid)
    #     if(view == 1):
    #         view1_ic = np.load('coords_correspondence/projection_forth/view1_correspondence_forth.npz')
    #         view1_ic = view1_ic.f.arr_0
    #         view1_ic = tf.cast(view1_ic, 'float32')
    #         view1_ic = tf.expand_dims(view1_ic, axis=0)
    #         view1_ic = tf.tile(view1_ic, [batch_size, 1, 1])
    #
    #         transformed_grid = tf.cast(view1_ic, 'float32')
    #
    #         view1_gp_mask = np.load('coords_correspondence/mask/view1_gp_mask.npz')
    #         view1_gp_mask = view1_gp_mask.f.arr_0
    #         view1_gp_mask = cv2.resize(view1_gp_mask, (610 / 4, 710 / 4))
    #         view1_gp_mask = tf.cast(view1_gp_mask, 'float32')
    #         view1_gp_mask = tf.expand_dims(view1_gp_mask, axis=0)
    #         view1_gp_mask = tf.expand_dims(view1_gp_mask, axis=3)
    #         view1_gp_mask = tf.tile(view1_gp_mask, [batch_size, 1, 1, num_channels])
    #         view_gp_mask = tf.cast(view1_gp_mask, 'float32')
    #
    #         view_norm_mask = np.load('coords_correspondence/norm/view_Wld_normalization_forth_mask.npz')
    #         view1_norm_mask= view_norm_mask.f.arr_0[0]
    #         view1_norm_mask = tf.cast(view1_norm_mask, 'float32')
    #         view1_norm_mask = tf.expand_dims(view1_norm_mask, axis=0)
    #         view1_norm_mask = tf.expand_dims(view1_norm_mask, axis=3)
    #         view1_norm_mask = tf.tile(view1_norm_mask, [batch_size, 1, 1, num_channels])
    #         view_norm_mask = tf.cast(view1_norm_mask, 'float32')
    #
    #     elif (view == 2):
    #         view2_ic = np.load('coords_correspondence/projection_forth/view2_correspondence_forth.npz')
    #         view2_ic = view2_ic.f.arr_0
    #         view2_ic = tf.cast(view2_ic, 'float32')
    #         view2_ic = tf.expand_dims(view2_ic, axis=0)
    #         view2_ic = tf.tile(view2_ic, [batch_size, 1, 1])
    #
    #         transformed_grid = tf.cast(view2_ic, 'float32')
    #
    #         view2_gp_mask = np.load('coords_correspondence/mask/view2_gp_mask.npz')
    #         view2_gp_mask = view2_gp_mask.f.arr_0
    #         view2_gp_mask = cv2.resize(view2_gp_mask, (610 / 4, 710 / 4))
    #         view2_gp_mask = tf.cast(view2_gp_mask, 'float32')
    #         view2_gp_mask = tf.expand_dims(view2_gp_mask, axis=0)
    #         view2_gp_mask = tf.expand_dims(view2_gp_mask, axis=3)
    #         view2_gp_mask = tf.tile(view2_gp_mask, [batch_size, 1, 1, num_channels])
    #         view_gp_mask = tf.cast(view2_gp_mask, 'float32')
    #
    #         view_norm_mask = np.load('coords_correspondence/norm/view_Wld_normalization_forth_mask.npz')
    #         view2_norm_mask= view_norm_mask.f.arr_0[1]
    #         view2_norm_mask = tf.cast(view2_norm_mask, 'float32')
    #         view2_norm_mask = tf.expand_dims(view2_norm_mask, axis=0)
    #         view2_norm_mask = tf.expand_dims(view2_norm_mask, axis=3)
    #         view2_norm_mask = tf.tile(view2_norm_mask, [batch_size, 1, 1, num_channels])
    #         view_norm_mask = tf.cast(view2_norm_mask, 'float32')
    #
    #     elif (view == 3):
    #         view3_ic = np.load('coords_correspondence/projection_forth/view3_correspondence_forth.npz')
    #         view3_ic = view3_ic.f.arr_0
    #         view3_ic = tf.cast(view3_ic, 'float32')
    #         view3_ic = tf.expand_dims(view3_ic, axis=0)
    #         view3_ic = tf.tile(view3_ic, [batch_size, 1, 1])
    #
    #         transformed_grid = tf.cast(view3_ic, 'float32')
    #
    #         view3_gp_mask = np.load('coords_correspondence/mask/view3_gp_mask.npz')
    #         view3_gp_mask = view3_gp_mask.f.arr_0
    #         view3_gp_mask = cv2.resize(view3_gp_mask, (610 / 4, 710 / 4))
    #         view3_gp_mask = tf.cast(view3_gp_mask, 'float32')
    #         view3_gp_mask = tf.expand_dims(view3_gp_mask, axis=0)
    #         view3_gp_mask = tf.expand_dims(view3_gp_mask, axis=3)
    #         view3_gp_mask = tf.tile(view3_gp_mask, [batch_size, 1, 1, num_channels])
    #         view_gp_mask = tf.cast(view3_gp_mask, 'float32')
    #
    #         view_norm_mask = np.load('coords_correspondence/norm/view_Wld_normalization_forth_mask.npz')
    #         view3_norm_mask= view_norm_mask.f.arr_0[2]
    #         view3_norm_mask = tf.cast(view3_norm_mask, 'float32')
    #         view3_norm_mask = tf.expand_dims(view3_norm_mask, axis=0)
    #         view3_norm_mask = tf.expand_dims(view3_norm_mask, axis=3)
    #         view3_norm_mask = tf.tile(view3_norm_mask, [batch_size, 1, 1, num_channels])
    #         view_norm_mask = tf.cast(view3_norm_mask, 'float32')
    #
    #
    #     x_s = tf.slice(transformed_grid, [0, 0, 0], [-1, 1, -1])
    #     y_s = tf.slice(transformed_grid, [0, 1, 0], [-1, 1, -1])
    #     x_s_flatten = tf.reshape(x_s, [-1])
    #     y_s_flatten = tf.reshape(y_s, [-1])
    #
    #     transformed_image = self._interpolate(input_shape,
    #                                             x_s_flatten,
    #                                             y_s_flatten,
    #                                             output_size)
    #
    #     transformed_image = tf.reshape(transformed_image, shape=(batch_size,
    #                                                             output_height,
    #                                                             output_width,
    #                                                             num_channels))
    #
    #     #transformed_image = tf.multiply(transformed_image, view_gp_mask)
    #     transformed_image = tf.multiply(transformed_image, view_norm_mask)
    #
    #     # # normalization:
    #     # get the sum of each channel/each image
    #     input_sum = tf.reduce_sum(input_shape, [1, 2])
    #     input_sum = tf.expand_dims(input_sum, axis=1)
    #     input_sum = tf.expand_dims(input_sum, axis=1)
    #
    #     output_sum = tf.reduce_sum(transformed_image, [1, 2])
    #     output_sum = tf.expand_dims(output_sum, axis=1)
    #     output_sum = tf.expand_dims(output_sum, axis=1)
    #
    #     amplify_times = tf.divide(input_sum, output_sum)
    #     mul_times = tf.constant([1, output_height, output_width, 1])
    #     amplify_times = tf.tile(amplify_times, mul_times)
    #
    #     # transformed_image = tf.image.resize_images(transformed_image,
    #     #                                            [output_height/4, output_width/4])
    #
    #     transformed_image_sum = tf.multiply(transformed_image, amplify_times)
    #
    #     return transformed_image_sum

