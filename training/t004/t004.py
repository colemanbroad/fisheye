import numpy as np
import sys
import os, shutil

import skimage.io as io
from scipy.ndimage import zoom, label
from scipy.signal import gaussian
from scipy.ndimage.morphology import binary_dilation

from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard, ReduceLROnPlateau

import unet

img6  = io.imread('data_train/img006.tif', plugin='tifffile')
img6  = img6.transpose((0,1,3,4,2))

net = unet.get_unet_n_pool(n_pool=2,
                            inputchan=2,
                            n_classes=3,
                            n_convolutions_first_layer=32,
                            dropout_fraction=0.2)

net.load_weights('unet_model_weights_checkpoint.h5')

xs = img6[0].copy()

xs = zoom(xs, (356/71, 1, 1, 1))

xs_xy = xs.copy()
xsmean = xs_xy.mean((1,2), keepdims=True)
xs_xy = xs_xy / xsmean
ys_xy = net.predict(xs_xy)

xs_xz = xs.copy().transpose((1,0,2,3))
xsmean = xs_xz.mean((1,2), keepdims=True)
xs_xz = xs_xz / xsmean
ys_xz = net.predict(xs_xz)
ys_xz = ys_xz.transpose((1,0,2,3))

xs_zy = xs.copy().transpose((2,1,0,3))
xsmean = xs_zy.mean((1,2), keepdims=True)
xs_zy = xs_zy / xsmean
ys_zy = net.predict(xs_zy)
ys_zy = ys_zy.transpose((2,1,0,3))

ys_avgd = np.stack([ys_xy, ys_xz, ys_zy], axis=0).mean(0)
np.save('ys_avgd.npy', ys_avgd)

# shutil.copy('i_trainer.py', 'training/')

