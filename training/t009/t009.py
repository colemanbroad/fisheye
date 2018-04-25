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

import unet
import warping
import lib as ll

def permz(p1,p2):
  "permutation mapping p1 to p2 for use in numpy.transpose. elems must be unique."
  assert len(p1)==len(p2)
  perm = list(range(len(p1)))
  for i,p in enumerate(p2):
    perm[i] = p1.index(p)
  return perm


name = "training/t009/"

img6 = imread('data/labels_lut.tif')
inds, traindata = ll.fixlabels(img6)

xs_xy = traindata[:,[1,2]].copy()
ys_xy = traindata[:,0].copy()

xs = xs_xy.transpose(permz('zcyx','zyxc'))
ys = ys_xy # already zyx, no c

ys[ys==3] = 1
ys[ys==4] = 1

if False:
  xs1 = np.flip(xs, axis=1)
  xs2 = np.flip(xs, axis=2)
  xs12 = np.flip(np.flip(xs, axis=1), axis=2)
  xs = np.concatenate((xs,xs1,xs2,xs12), axis=0)

  ys1 = np.flip(ys, axis=1)
  ys2 = np.flip(ys, axis=2)
  ys12 = np.flip(np.flip(ys, axis=1), axis=2)
  ys = np.concatenate((ys,ys1,ys2,ys12), axis=0)

n_classes = len(np.unique(ys))
# classweights = (1-counts/counts.sum())/(len(counts)-1)
classweights = [1/n_classes,]*n_classes

distimg = ys.copy()
distimg[distimg!=2] = 1
distimg[distimg==2] = 0
distimg = distance_transform_edt(distimg)
distimg = np.exp(-distimg/10)

# distimg /= distimg.mean()
ys = np_utils.to_categorical(ys).reshape(ys.shape + (-1,))
ysmean = ys.mean()
ys = ys*distimg[...,np.newaxis]
ys *= ysmean/ys.mean()

# split 400x400 into 100x100 patches

nz,ny,nx,nc = xs.shape
ny4,nx4 = ny//4, nx//4
xs = xs.reshape((nz,4,ny4,4,nx4,nc)).transpose(permz("z1y2xc","z12yxc")).reshape((-1,ny4,nx4,nc))
nz,ny,nx,nc = ys.shape
ys = ys.reshape((nz,4,ny4,4,nx4,nc)).transpose(permz("z1y2xc","z12yxc")).reshape((-1,ny4,nx4,nc))

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
                            n_classes=n_classes,
                            n_convolutions_first_layer=32,
                            dropout_fraction=0.2)

# net.load_weights('unet_model_weights_checkpoint.h5')

optim = Adam(lr=1e-4)
loss = unet.my_categorical_crossentropy(weights=classweights, itd=4)
net.compile(optimizer=optim, loss=loss, metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=name + "w001.h5", verbose=0, save_best_only=True, save_weights_only=True)
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

net.save_weights(name + 'w002.h5')


