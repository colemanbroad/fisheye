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
from segtools import segtools_simple as ss
from segtools import color
from . import lib as lib_local

from keras.utils import np_utils

img6 = io.imread('data_train/img006.tif')
img6 = img6.transpose((0,1,3,4,2))

ys_ilastik = np.load('data_train/img006_Probabilities.npy')
ys_ilastik = ys_ilastik[1]

# ys_t001 = np.load('training/t001/ys_t1.npy')
# ys_t002 = np.load('training/t002/ys_t1.npy')
# ys_t005 = np.load('training/t005/ys_avgd_t1.npy')
# ys_t005 = ys_t005[1::5]
# ys_t006 = np.load('training/t006/ys_t1.npy')
ys_t007 = np.load('training/t007/ys_avgd_t1.npy')
ys_t007 = ys_t007[1::5]

potential = 1 - ys_t007[...,1] + ys_t007[...,0]
potential = np.clip(potential, 0, 1)
hx = gaussian(21, 2.0)
potential = gputools.convolve_sep3(potential, hx, hx, hx)
potential = lib.normalize_percentile_to01(potential, 0, 100)
# nhl,hyp = watershed(potential, c1=0.5, c2=0.9)
hyp = watershed(potential, label(potential<0.1)[0], mask=potential<0.5)
nhl = lib.hyp2nhl(hyp, img6[0,...,1])
nhl = np.array(nhl)

iss = view.ImshowStack([img6[1,...,1], potential, hyp])
plt.figure()
areas = np.array([n['area'] for n in nhl])
xmom  = np.array([n['moments_img'][0,0,0]/n['area'] for n in nhl])
col = plt.scatter(np.log2(areas), xmom) #np.log2(xmom))
selector = view.SelectFromCollection(plt.gca(), col)

w = spimagine.volshow(img6[1,...,1], interpolation='nearest', stackUnits=[1.0, 1.0, 1.0])

img_spim = img6[0,...,1]
w = spimagine.volshow(img_spim, interpolation='nearest', stackUnits=[1.0, 1.0, 5.0])

def update_selection(r):
  # img = img6[1,...,1].copy()
  # img = w.glWidget.dataModel[]
  img2 = img_spim.copy()
  mask = lib.mask_nhl(nhl[selector.ind], hyp)
  img2[mask] = img2[mask]*r
  lib.update_spim(w, 0, img2)

iss = view.ImshowStack(img6[1,...,1].copy())

def update_stack(r):
  img = img6[1,...,1].copy()
  mask = lib.mask_nhl(nhl[selector.ind], hyp)
  img[mask] = img[mask]*r
  iss.stack[0] = img

img = img6[1,...,1].copy()
cmap = {n['label'] : n['moments_img'][0,0,0]/n['area'] for n in nhl}
cmap[0] = 1.0
hyp_max = color.apply_mapping(hyp, cmap)
lib.update_spim(w, 0, img/hyp_max)

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