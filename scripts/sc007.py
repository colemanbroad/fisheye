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
from segtools import seg_sasha
from segtools import loc_utils
from . import lib as lib_local

from keras.utils import np_utils

from glob import glob

img6 = [np.load(file) for file in ['data_train/img6_t0_zup.npy', 'data_train/img6_t1_zup.npy'] + loc_utils.sorted_nicely(glob('training/t007/img6*.npy'))]
img6 = np.array(img6)

img_t007 = [np.load(file) for file in loc_utils.sorted_nicely(glob('training/t007/ys_avgd*.npy'))]
img_t007 = np.array(img_t007)

# img_fly = io.imread('/Volumes/myersspimdata/Robert/2018-01-18-16-30-25-11-Robert_CalibZAP_Wfixed/processed/tif/000307.raw.tif')

# img = img_t007[0].copy()

for i in range(1, img_t007.shape[0] - 1):
  img1 = img_t007[i].copy()
  img2 = img_t007[i+1].copy()
  plt.figure()
  segment_and_plot(img1)
  segment_and_plot(img2)
  plt.xlim(6,20)
  plt.ylim(0.2, 1.0)

def segment_and_plot(img, with_border_cells=True):
  potential = 1 - img[...,1] + img[...,0]
  potential = np.clip(potential, 0, 1)
  hx = gaussian(21, 2.0)
  potential = gputools.convolve_sep3(potential, hx, hx, hx)
  potential = lib.normalize_percentile_to01(potential, 0, 100)
  # nhl,hyp = watershed(potential, c1=0.5, c2=0.9)
  hyp = watershed(potential, label(potential<0.1)[0], mask=potential<0.5)
  if with_border_cells==False:
    mask_borders = lib.mask_border_objs(hyp)
    hyp[mask_borders] = 0
  nhl = lib.hyp2nhl(hyp, img[...,1])
  nhl = np.array(nhl)
  areas = np.array([n['area'] for n in nhl])
  xmom  = np.array([n['moments_img'][0,0,0]/n['area'] for n in nhl])
  col = plt.scatter(np.log2(areas), xmom) #np.log2(xmom))
  # selector = view.SelectFromCollection(plt.gca(), col)
  return nhl, hyp



img_spim = img6[0,...,1]
w = spimagine.volshow(img_spim, interpolation='nearest', stackUnits=[1.0, 1.0, 1.0])

def update_selection(r):
  # img = img6[1,...,1].copy()
  # img = w.glWidget.dataModel[]
  img2 = img_spim.copy()
  mask = lib.mask_nhl(nhl[selector.ind], hyp)
  img2[mask] = img2[mask]*r
  lib.update_spim(w, 0, img2)

def update_stack(r):
  img = img_spim.copy()
  mask = lib.mask_nhl(nhl[selector.ind], hyp)
  img[mask] = img[mask]*r
  iss.stack = img

def onclick(event):
    xi, yi = int(event.xdata + 0.5), int(event.ydata + 0.5)
    zi = iss.idx[1]
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
    print(zi, yi, xi)
    print(event.key)
    # self.centerpoints.append([zi,yi,xi])
    w.glWidget.dataModel[0][...] = img6[iss.idx[0],...,1].astype('float')
    w.glWidget.dataModel[0][hyp==hyp[zi,yi,xi]] *= 1.8
    w.glWidget.dataPosChanged(0)
iss.fig.canvas.mpl_connect('button_press_event', onclick)
