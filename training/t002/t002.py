import numpy as np
import sys
import os, shutil
# sys.path.insert(0, '/Users/colemanbroaddus/Desktop/Projects/nucleipix/')

import skimage.io as io
# import matplotlib.pyplot as plt
# plt.ion()
from scipy.ndimage import zoom, label
from scipy.signal import gaussian
from scipy.ndimage.morphology import binary_dilation

from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard, ReduceLROnPlateau

import unet

img6  = io.imread('data_train/img006.tif', plugin='tifffile')
img6  = img6.transpose((0,1,3,4,2))
# lab6  = np.load('data_train/labels006.npy')
# img6p = np.load('data_train/img006_Probabilities.npy')
labfull = np.load('lab6_dense.npy') # first 10 z slices of first time point

# lab6_nuc = io.imread('/net/fileserver-nfs/stornext/snfs2/projects/myersspimdata/Mauricio/from_coleman/img006_borderlabels.tif', plugin='tifffile')
# lab6_nuc = lab6_nuc.transpose((0,1,3,4,2))
# lab6_nuc = lab6_nuc[...,2]

xs = img6[0,:10].copy()
ys = labfull.copy()
# ys2 = img6p[0].copy()

# _, counts = np.unique(ys, return_counts=True)
# classweights = (1-counts/counts.sum())/(len(counts)-1)
classweights = [1/3, 1/3, 1/3]
print("ClassWeights:", classweights)

# permute labels so as to be consistent with previous classifiers
ys[ys>1] = 3
ys[ys==1] = 2
ys[ys==3] = 1
ys = np_utils.to_categorical(ys).reshape(ys.shape + (-1,))

# # combine hand-made labels and RF predictions
# a,b,c = ys.shape
# mask = ys!=0
# ys = np_utils.to_categorical(ys).reshape((a,b,c,-1))
# ys2[mask] = ys[mask][:,[1,2,3]]
# a,b = mask.sum(), (~mask).sum()
# c = b/a
# rb = 0.5
# ra = (1 - rb) * c + 1
# err = a + b - (ra*a + rb*b)
# assert err < 1
# ys2[mask] *= ra
# ys2[~mask] *= rb

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

net.load_weights('unet_model_weights_checkpoint.h5')

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


xs_predict = img6[1].copy()
xsmean = xs_predict.mean((1,2), keepdims=True)
xs_predict = xs_predict / xsmean
ys_predict = net.predict(xs_predict)
# ys_predict = ys_predict.reshape((-1,4,4,100,100,3)).transpose((0,1,3,2,4,5)).reshape((-1,400,400,3))
np.save('ys_predict_1.npy', ys_predict)

# shutil.copy('i_trainer.py', 'training/')

