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
labfull = np.load('data_train/lab6_dense.npy') # first 10 z slices of first time point

xs = img6[0,:10].copy()
ys = labfull.copy()

# classweights = (1-counts/counts.sum())/(len(counts)-1)
classweights = [1/3, 1/3, 1/3]

# permute labels so as to be consistent with previous classifiers
ys[ys>1] = 3
ys[ys==1] = 2
ys[ys==3] = 1
ys = np_utils.to_categorical(ys).reshape(ys.shape + (-1,))

# now downsample and upscale x dimension to make it look like z dimension.
xs = xs[:,:,::5]
xs = zoom(xs, (1, 1, 5, 1))

# 400x400 → 16x100x100
xs = xs.reshape((-1,4,100,4,100,2)).transpose((0,1,3,2,4,5)).reshape((-1,100,100,2))
ys = ys.reshape((-1,4,100,4,100,3)).transpose((0,1,3,2,4,5)).reshape((-1,100,100,3))

# normalize
xsmean = xs.mean((1,2), keepdims=True)
xs = xs / xsmean

# shuffle
inds = np.arange(xs.shape[0])
np.random.shuffle(inds)
invers = np.argsort(np.arange(inds.shape[0])[inds])
xs = xs[inds]
ys = ys[inds]

# train test split
split = 7
n_vali = xs.shape[0]//split
xs_train = xs[:-n_vali]
ys_train = ys[:-n_vali]
xs_vali  = xs[-n_vali:]
ys_vali  = ys[-n_vali:]


net = unet.get_unet_n_pool(n_pool=2,
                            inputchan=2,
                            n_classes=3,
                            n_convolutions_first_layer=32,
                            dropout_fraction=0.2)

# net.load_weights('unet_model_weights_checkpoint.h5')

optim = Adam(lr=1e-4)
loss = unet.my_categorical_crossentropy(weights=classweights, itd=4)
net.compile(optimizer=optim, loss=loss, metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="unet_model_weights_checkpoint.h5", verbose=0, save_best_only=True, save_weights_only=True)
earlystopper = EarlyStopping(patience=30, verbose=0)
reduce_lr    = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
callbacks = [checkpointer, earlystopper, reduce_lr]

history = net.fit(x=xs_train,
                  y=ys_train,
                  batch_size=3,
                  epochs=60,
                  verbose=1,
                  callbacks=callbacks,
                  validation_data=(xs_vali, ys_vali))


xs = img6[0].copy()
xs = zoom(xs, (356/71, 1, 1, 1))

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

net.load_weights('training/t003/unet_model_weights_checkpoint.h5')

xs_xy = xs.copy()
xsmean = xs_xy.mean((1,2), keepdims=True)
xs_xy = xs_xy / xsmean
ys_xy = net.predict(xs_xy)

ys_avgd = np.stack([ys_xy, ys_xz, ys_zy], axis=0).mean(0)
np.save('ys_avgd.npy', ys_avgd)


