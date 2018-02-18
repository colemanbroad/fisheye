import numpy as np
import sys

import skimage.io as io
import matplotlib.pyplot as plt
plt.ion()
from scipy.ndimage import zoom, label
from scipy.signal import gaussian
from scipy.ndimage.morphology import binary_dilation

from tabulate import tabulate
import pandas
from skimage.morphology import watershed
from segtools import voronoi

import gputools
import spimagine

from segtools import cell_view_lib as view
from segtools import lib
from . import lib as lib_local

from keras.utils import np_utils

img6 = io.imread('data_train/img006.tif')
img6 = img6.transpose((0,1,3,4,2))

ys_ilastik = np.load('data_train/img006_Probabilities.npy')
ys_ilastik = ys_ilastik[1]

ys_t001 = np.load('training/t001/ys_t1.npy')
ys_t002 = np.load('training/t002/ys_t1.npy')
ys_t005 = np.load('training/t005/ys_avgd_t1.npy')
ys_t005 = ys_t005[:-1:5]
ys_t006 = np.load('training/t006/ys_t1.npy')
ys_t007 = np.load('training/t007/ys_avgd_t0.npy')

# # set up the test data
# labfull = np.load('data_train/lab6_dense.npy')
# ys = labfull.copy()
# ys[ys>1] = 3
# ys[ys==1] = 2
# ys[ys==3] = 1
# sh = ys.shape
# ys = np_utils.to_categorical(ys).reshape(sh + (3,))

# iss = view.ImshowStack(np.concatenate([ys_t001, ys_t006], axis=-2), colorchan=True)
# w = spimagine.volshow(ys_avgd[...,0], interpolation="nearest")

points = lib_local.mkpoints()
seeds = np.zeros_like(potential, dtype=np.int)
seeds[[*points.T]] = np.arange(points.shape[0]) + 1

names = ['RF (SPARSE)', 'NET (SPARSE)', 'NET (DENSE)', 'NET (ISO AVGD)', 'NET (border weights)']
data  = [ys_ilastik, ys_t001, ys_t002, ys_t005, ys_t006]
plt.figure()
for i in range(5):
  img = data[i]
  name = names[i]
  # potential = 1 - img[...,1] + img[...,0]
  # potential = lib.normalize_percentile_to01(potential, 0, 100)*2
  potential = img[...,1] # - img[...,0]
  hx = gaussian(21, 2.0)
  potential  = gputools.convolve_sep3(potential, hx, hx, hx)
  potential = lib.normalize_percentile_to01(potential, 0, 100)
  # hyp = watershed(potential, seeds, mask=potential<0.5)
  # nhl = lib.hyp2nhl(hyp)
  nhl,hyp = lib.two_var_thresh(potential, c1=0.5, c2=0.9)
  areas = np.array([n['area'] for n in nhl])
  plt.plot(sorted(np.log2(areas)), label=name)
plt.legend()

iss = view.ImshowStack([img6[1,...,1], ys_ilastik[...,1], ys_t001[...,1], ys_t002[...,1], ys_t005[...,1], ys_t006[...,1]])
iss = view.ImshowStack([img6[1,...,1], ys_t006[...,1], hyp])