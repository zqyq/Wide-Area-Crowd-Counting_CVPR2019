# This is the code for MVMS.
#
# ---
# Wide-Area Crowd Counting via Ground-Plane Density Maps and Multi-View Fusion CNNs
# Copyright (c) 2019
# Qi Zhang, Antoni B. Chan
# City University of Hong Kong


from __future__ import print_function
import os
import sys
import time
import h5py
from pprint import pprint, pformat
import numpy as np
np.set_printoptions(precision=6)
fixed_seed = 999
np.random.seed(fixed_seed)  # Set seed for reproducibility
import tensorflow as tf
import keras
print("Using keras {}".format(keras.__version__))
# assert keras.__version__.startswith('2.')
tf.set_random_seed(fixed_seed)

from datagen_v3 import datagen_v3

from net_def import build_model_FCN_model_api as build_FCNN

from keras.optimizers import Adam, Nadam
from keras.optimizers import SGD
# from MyOptimizer_keras2 import SGD_policy as SGD
from keras.callbacks import Callback, CSVLogger, ModelCheckpoint, EarlyStopping, TensorBoard
from keras.models import load_model

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# set enough GPU memory as needed(default, all GPU memory is used)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def logging_level(level='info'):
    import logging
    str_format = '%(asctime)s - %(levelname)s: %(message)8s'
    if level == 'debug':
        logging.basicConfig(level=logging.DEBUG, format=str_format, datefmt='%Y-%m-%d %H:%M:%S')
    elif level == 'info':
        logging.basicConfig(level=logging.INFO, format=str_format, datefmt='%Y-%m-%d %H:%M:%S')

    return logging


class batch_loss_callback(Callback):
    """Callback that streams epoch results to a plain txt file.

    Supports all values that can be represented as a string,
    including 1D iterables such as np.ndarray.

    # Arguments
        filename: filename of the csv file, e.g. 'run/log.csv'.
        append: True: append if file exists (useful for continuing
            training). False: overwrite existing file,
    """

    def __init__(self, filename, append=False):
        self.filename = filename
        self.append = append
        self.append_header = True
        self.batch_number = 0
        super(batch_loss_callback, self).__init__()

    def on_train_begin(self, logs=None):
        self.losses = list()
        # pass
        # if self.append:
        #     if os.path.exists(self.filename):
        #         with open(self.filename) as f:(
        #             self.append_header = not bool(len(f.readline()))
        #     self.textfile = open(self.filename, 'a')
        # else:
        #     self.textfile = open(self.filename, 'w')

    def on_batch_end(self, batch, logs=None):
        self.losses.append(logs.get('loss'))
        self.batch_number += 1
        if self.batch_number % 100 == 0:
            # self.textfile.write("Batch No. {:7d}\n".format(self.batch_number))
            # self.textfile.write("    {:.6f}\n".format(logs.get('loss')))
            info_print_1 = "Batch No. {:7d}".format(self.batch_number)
            info_print_2 = "    loss: {:.6f}".format(logs.get('loss'))
            # self.textfile.write("{}\n".format(info_print_1))
            # self.textfile.write("{}\n".format(info_print_2))
            logging.info(info_print_1)
            logging.info(info_print_2)
            # self.textfile.flush()

    def on_epoch_end(self, epoch, logs=None):
        temp_name, _ = self.filename.rsplit('.', 1)
        with h5py.File('{}.h5'.format(temp_name), 'w') as f:
            f['losses'] = np.asarray(self.losses, dtype=np.float32)
        # pass
        # self.textfile.close()


def main(exp_name='FCNN', verbosity=0):
    ################################################################################
    # Experiment Settings
    ################################################################################

    # Model + Generator hyper-parameters
    optimizer = 'sgd'
    learning_rate = 0.0001   # 0.0001
    # learning_rate = 0.002  # for Nadam only
    lr_decay = 0.0001
    momentum = 0.9
    nesterov = False
    weight_decay = 0.0001 # 0.001
    save_dir = os.path.join("", exp_name)

    batch_size = 1
    epochs = 2000
    images_per_set = None
    patches_per_image = 1      #1000
    patch_dim = (380, 676, 3)
    image_shuffle = True
    patch_shuffle = True
    epoch_random_state = None  # Set this to an integer if you want the data the same in every epoch

    train_samples = 300    #243243 #20490
    val_samples   = 200     #59598 #8398

    ################################################################################
    # Model Definition
    if optimizer.lower() == 'sgd':
        opt = SGD(
            lr=learning_rate,
            decay=lr_decay,
            momentum=momentum,
            nesterov=nesterov,
            # lr_policy='inv' if params_solver.get('lr_policy', None) is None else params_solver['lr_policy'],
            # step=params_solver.get('step', 10000000.),  # useful only when lr_policy == 'step'
            clipnorm=5,
            clipvalue=1)
    elif optimizer.lower() == 'adam':
        logging.info("use Adam solver")
        opt = Adam(
            lr=learning_rate,
            # decay=params_solver.get('lr_decay', 0.),
            clipnorm=5,
            clipvalue=1)
    elif optimizer.lower() == 'nadam':
        logging.info("use Nadam solver")
        opt = Nadam(
            lr=learning_rate,
            clipnorm=5,
            clipvalue=1)
    else:
        logging.error('Unrecognized solver')

    model = build_FCNN(
        batch_size=batch_size,
        patch_size=patch_dim,
        optimizer=opt,
        base_weight_decay=weight_decay,
        output_ROI_mask=False,)
    # loading single-view counting feature extraction weights performs better


    # Generator setup
    scaler_stability_factor = 1000  # 100

    train_path0 = '/opt/visal/home/Multi_view/Datasets/Street/'

    train_view1_1 = train_path0 + 'dmaps/train/Street_view1_dmap_10.h5'
    train_view2_1 = train_path0 + 'dmaps/train/Street_view2_dmap_10.h5'
    train_view3_1 = train_path0 + 'dmaps/train/Street_view3_dmap_10.h5'
    train_GP_1 = train_path0 + 'GP_dmaps/train/Street_groundplane_train_dmaps_10.h5'

    h5file_train_view1 = [train_view1_1]
    h5file_train_view2 = [train_view2_1]
    h5file_train_view3 = [train_view3_1]
    h5file_train_GP = [train_GP_1]

    train_gen = datagen_v3(
        h5file_view1=h5file_train_view1,
        h5file_view2=h5file_train_view2,
        h5file_view3=h5file_train_view3,
        h5file_GP=h5file_train_GP,

        batch_size=batch_size,
        images_per_set=images_per_set,
        patches_per_image=patches_per_image,
        patch_dim=patch_dim[:2],
        density_scaler=scaler_stability_factor,
        image_shuffle=image_shuffle,
        patch_shuffle=patch_shuffle,
        random_state=epoch_random_state
    )

    test_view1_1 = train_path0 + 'dmaps/test/Street_view1_dmap_10.h5'
    test_view2_1 = train_path0 + 'dmaps/test/Street_view2_dmap_10.h5'
    test_view3_1 = train_path0 + 'dmaps/test/Street_view3_dmap_10.h5'
    test_GP_1 = train_path0 + 'GP_dmaps/test/Street_groundplane_test_dmaps_10.h5'

    h5file_test_GP = [test_GP_1]

    h5file_test_view1 = [test_view1_1]
    h5file_test_view2 = [test_view2_1]
    h5file_test_view3 = [test_view3_1]

    val_gen = datagen_v3(
        h5file_view1=h5file_test_view1,
        h5file_view2=h5file_test_view2,
        h5file_view3=h5file_test_view3,
        h5file_GP=h5file_test_GP,

        batch_size=batch_size,
        images_per_set=images_per_set,
        patches_per_image=1,  # 1000,
        patch_dim=patch_dim[:2],
        density_scaler=scaler_stability_factor,
        image_shuffle=image_shuffle,
        patch_shuffle=patch_shuffle,
        random_state=epoch_random_state
    )

    # Model Training
    # Save directory
    if not os.path.exists(save_dir):
        logging.info(">>>> save dir: {}".format(save_dir))
        os.makedirs(save_dir)
    callbacks = list()
    callbacks.append(CSVLogger(
        filename=os.path.join(save_dir, 'train_val.csv'),
        separator=',',
        append=False,  # useful if it's resumed from the latest checkpoint
    ))
    callbacks.append(ModelCheckpoint(
        filepath=os.path.join(save_dir, '{epoch:02d}-{val_loss:.4f}.h5'),
        monitor='val_loss',
        verbose=1,
        save_weights_only=True,
        save_best_only=False,
    ))
    callbacks.append(ModelCheckpoint(
        filepath=os.path.join(save_dir, '{epoch:02d}-{val_loss:.4f}-better.h5'),
        monitor='val_loss',
        verbose=1,
        save_weights_only=True,
        save_best_only=True,  # this will save all the improved models
    ))
    #callbacks.append(EarlyStopping(monitor='val_loss', patience=50, verbose=1, mode='auto'))
    # callbacks.append(TensorBoard(
    #     log_dir=os.path.join(save_dir, 'TensorBoard_info'),
    #     histogram_freq=1,
    #     write_graph=True,
    #     write_images=True,
    #     embeddings_freq=0,  # default, new feature only in latest keras and tensorflow
    #     embeddings_layer_names=None,  # default
    #     embeddings_metadata=None,  # default
    # ))
    if verbosity == 0:
        callbacks.append(batch_loss_callback(
            filename=os.path.join(save_dir, 'train_val_loss_batch.log'),
            append=False,  # useful if it's resumed from the latest checkpoint
        ))

    logging.info('Begin training...')
    start_time = time.time()
    # train the network from here
    model.fit_generator(
        generator=train_gen,
        steps_per_epoch=train_samples // batch_size,
        epochs=epochs,
        verbose=verbosity,
        callbacks=callbacks,
        validation_data=val_gen,
        validation_steps=val_samples // batch_size,
        max_q_size=20,
        workers=1,
        pickle_safe=False)

    # YY = model.layers[60].output
    # print(sum(YY.flatten))

    logging.info('----- {:.2f} seconds -----'.format(time.time() - start_time))

    # # Save model history
    # sys.setrecursionlimit(100000)
    # with open('{}/training.history'.format(save_dir), 'w') as f:
    #     # does not work for Keras2. Also not convinient to use
    #     pickle.dump(history, f)


if __name__ == '__main__':
    logging = logging_level('debug')
    logging.debug('use debug level logging setting')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-e", "--exp_name",
        type=str,
        default='models/Street_all_1output',
        action="store")
    parser.add_argument(
        "-v", "--verbosity",
        type=int,
        default=0,
        choices=[0, 1, 2],
        action="store")
    args = parser.parse_args()
    main(exp_name=args.exp_name, verbosity=args.verbosity)