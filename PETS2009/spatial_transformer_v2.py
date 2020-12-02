from keras.layers.core import Layer
import tensorflow as tf
import numpy as np
import cv2

import camera_proj_Tsai as proj


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
        #
        # output_size = self.output_size
        # output_height = int(output_size[0])
        # output_width = int(output_size[1])
        #
        # batch_size = inputs.shape[0].value
        # height = inputs.shape[1].value
        # width = inputs.shape[2].value
        # num_channels = inputs.shape[3].value
        #
        # D = int(30/4)*4 #/4
        # height_range = np.linspace(0, D - 1, D)
        #
        # output = tf.zeros([batch_size, output_height, output_width, 1])
        # for i in range(len(height_range)):
        #     hi = height_range[i]
        #     input_i = inputs[:, :, :, i:i+1]
        #     self.proj_2DTo2D(view, input_i, hi) #inputs
        #     output_i = self.proj_splat(input_i, hi) #inputs
        #     output = tf.concat([output, output_i], axis=-1)
        # output = output[:, :, :, 1:]
        #
        # output = tf.expand_dims(output, axis=-1)
        # num_channels = 33
        # output = tf.tile(output, [batch_size, 1, 1, 1, num_channels])
        
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
        w = 768
        h = 576
        W = int(610/4)
        H = int(710/4)
        # D = hi #30/4

        bbox = [-31, 29, -45, 25]
        # bbox = [-21, 19, -25, 15]

        image_size = [576/4, 768/4]

        resolution_scaler = 10 #8
        # control the resolution of the project ROI mask (ground plane density map).

        ph = 1.75 * 1000
        # average height of a person in millimeters

        # nR, fh, fw, fdim = self.tf_static_shape(inputs)

        nR, fh, fw, fdim = inputs.get_shape().as_list()
        # self.batch_size, self.gp_x, self.gp_y, self.gp_z = nR, W, H, D
        self.batch_size, self.gp_x, self.gp_y = nR, W, H


        rsz_h = float(fh) / h
        rsz_w = float(fw) / w

        # Create voxel grid
        grid_rangeX = np.linspace(0, W - 1, W)
        grid_rangeY = np.linspace(0, H - 1, H)
        # grid_rangeZ = hi #np.linspace(0, D - 1, D)
        # grid_rangeX, grid_rangeY, grid_rangeZ = np.meshgrid(grid_rangeX, grid_rangeY, grid_rangeZ)
        grid_rangeX, grid_rangeY = np.meshgrid(grid_rangeX, grid_rangeY)

        grid_rangeX = np.reshape(grid_rangeX, [-1])
        grid_rangeY = np.reshape(grid_rangeY, [-1])
        #grid_rangeZ = np.reshape(grid_rangeZ, [-1])

        grid_rangeX = grid_rangeX * 4 / resolution_scaler + bbox[0] #*1.25
        grid_rangeX = grid_rangeX * 1000
        grid_rangeX = np.expand_dims(grid_rangeX, 1)

        grid_rangeY = grid_rangeY * 4 / resolution_scaler + bbox[2] #*1.25
        grid_rangeY = grid_rangeY * 1000
        grid_rangeY = np.expand_dims(grid_rangeY, 1)

        # grid_rangeZ = grid_rangeZ * 400* np.ones(grid_rangeX.shape)
        # grid_rangeZ = np.expand_dims(grid_rangeZ, 1)
        grid_rangeZ = ph * np.ones(grid_rangeX.shape)

        wldcoords = np.concatenate(([grid_rangeX, grid_rangeY, grid_rangeZ]), axis=1)

        if view==1:
            view = 'view1'
            view_gp_mask = np.load('coords_correspondence/mask/view1_gp_mask.npz')
        if view==2:
            view = 'view2'
            view_gp_mask = np.load('coords_correspondence/mask/view2_gp_mask.npz')
        if view==3:
            view = 'view3'
            view_gp_mask = np.load('coords_correspondence/mask/view3_gp_mask.npz')

        # gp view mask:
        view_gp_mask = view_gp_mask.f.arr_0
        view_gp_mask = cv2.resize(view_gp_mask, (W, H))
        view_gp_mask = tf.cast(view_gp_mask, 'float32')
        view_gp_mask = tf.expand_dims(view_gp_mask, axis=0)
        view_gp_mask = tf.expand_dims(view_gp_mask, axis=1)
        view_gp_mask = tf.expand_dims(view_gp_mask, axis=-1)
        batch_size = nR
        num_channels = fdim ###### remember to add the depth dim

        #view_gp_mask = tf.tile(view_gp_mask, [batch_size, 1, 1, self.gp_z, num_channels])
        view_gp_mask = tf.tile(view_gp_mask, [int(self.batch_size / self.patch_num),
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
        view_ic[2:3, :] = view_ic[2:3, :] #/ 400


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

            # with tf.name_scope('AppendDepth'):
            #     # Concatenate depth value along ray to feature
            #     Ibilin = tf.concat(
            #         [Ibilin, tf.reshape(im_z, [nV * nR, 1])], axis=1)
            #     fdim = Ibilin.get_shape().as_list()[-1]
            #     Ibilin = tf.reshape(Ibilin, [self.batch_size, self.gp_y, self.gp_x, self.gp_z]) # fdim

                Ibilin = tf.reshape(Ibilin, [int(self.batch_size/self.patch_num), self.patch_num,
                                             self.gp_y, self.gp_x, fdim]) # fdim

                # add a mask:
                self.Ibilin = tf.multiply(Ibilin, self.view_gp_mask)

                # self.Ibilin = Ibilin
                # self.Ibilin = tf.transpose(self.Ibilin, [0, 2, 1, 3, 4])
        return self.Ibilin