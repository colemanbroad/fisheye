import numpy as np
import sys
import os, shutil

import skimage.io as io
from scipy.ndimage import zoom, label, distance_transform_edt, rotate
from scipy.signal import gaussian
from scipy.ndimage.morphology import binary_dilation

from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard, ReduceLROnPlateau

# from segtools import loc_utils

import unet
import warping

import lib
from glob import glob


# img6 = io.imread('data_train/img6_zup.tif')
name = 'prediction/p001/'

img6 = [np.load(file) for file in ['data_train/img6_t0_zup.npy', 'data_train/img6_t1_zup.npy'] + lib.sorted_nicely(glob('training/t007/img6*.npy'))]
img6 = np.array(img6)


net = unet.get_unet_n_pool(n_pool=2,
                            inputchan=2,
                            n_classes=3,
                            n_convolutions_first_layer=32,
                            dropout_fraction=0.2)

def permute_reshape_predict(xs, perm):
  # a,b,c,d,e = xs.shape
  # xs = xs.reshape((a*b,c,d,e))
  xs = xs.transpose(perm)
  xsmean = xs.mean((1,2), keepdims=True)
  xs = xs / xsmean
  output = net.predict(xs)
  # output = output.reshape((a,b,c,d,e))
  xs = xs.transpose(perm)
  return output

net.load_weights('training/t008/w001.h5')

xs = img6.copy()
out_xy = np.array([permute_reshape_predict(x, (0,1,2,3)) for x in xs])
np.save(name + 'out001.npy', out_xy)

net.load_weights('training/t007/w001.h5')

xs = img6.copy()
out_xz = np.array([permute_reshape_predict(x, (1,0,2,3)) for x in xs])
np.save(name + 'out002.npy', out_xz)

xs = img6.copy()
out_zy = np.array([permute_reshape_predict(x, (2,1,0,3)) for x in xs])
np.save(name + 'out003.npy', out_zy)

ys_avgd = np.stack([out_xy, out_xz, out_zy], axis=0).mean(0)

np.save('ys_avgd.npy', ys_avgd)