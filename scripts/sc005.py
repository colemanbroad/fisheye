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

# img6 = io.imread('data_train/img006.tif')
# img6 = img6.transpose((0,1,3,4,2))

img = io.imread("/Users/colemanbroaddus/Desktop/img6_0_up_xz.tif")
lab = img[...,2]
lab_inds = lab.max((1,2)) > 0
lab_inds = np.arange(len(lab_inds))[lab_inds]
lab = lab[lab_inds]
m1 = lab==255 # border
m2 = lab==85 # cell centers
lab = np.array([label(1 - img)[0] for img in m1])
lab_semantic = lab.copy()

for i in range(lab.shape[0]):
  ids, cts = np.unique(lab[i][m2[i]], return_counts=True)
  mask = lib.mask_labels(ids, lab[i])
  lab_semantic[i] = 2
  lab_semantic[i][m1[i]] = 0
  lab_semantic[i][mask] = 1


newcents = []
def onclick_centerpoints(event):
  xi, yi = int(event.xdata + 0.5), int(event.ydata + 0.5)
  zi = iss.idx[0]
  print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
      (event.button, event.x, event.y, event.xdata, event.ydata))
  print(zi, yi, xi)
  if event.key=='C':
    print('added! ', event.key)
    newcents.append([zi,yi,xi])
cid = iss.fig.canvas.mpl_connect('button_press_event', onclick_centerpoints)

# iss.fig.canvas.mpl_disconnect()

for pt in newcents:
  z,y,x = pt
  l = lab[z,y,x]
  mask = lab[z]==l
  lab_semantic[z][mask] = 2


xs_ys = img[lab_inds]
xs_ys[...,2] = lab_semantic

np.save('data_train/sc005_xz.npy', xs_ys)