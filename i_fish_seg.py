import numpy as np
import sys
sys.path.insert(0, '/Users/colemanbroaddus/Desktop/Projects/nucleipix/')

import skimage.io as io
import matplotlib.pyplot as plt
plt.ion()
from scipy.ndimage import zoom, label
from scipy.signal import gaussian
from scipy.ndimage.morphology import binary_dilation

import gputools
import spimagine

from segtools import cell_view_lib as view
from segtools import lib

# from tifffile import TiffFile, FileHandle
# fh = FileHandle('/Volumes/myersspimdata/Mauricio/for_coleman/test_lap2b_H2BGFP_H2BRFP/20_12_17_multiview H2B_RFP&BFP_Lap2bGFP_fish6_Multiview_RIF_Subset.czi')
# fh = FileHandle('fish6_ch2.tif')

img6 = io.imread('img006.tif')
img6 = img6.transpose((0,1,3,4,2))
img6p = np.load('img006_Probabilities.npy')
img6p = img6p.transpose((0,1,4,2,3))
img6all = np.concatenate([img6,img6p], axis=2)
lab6 = np.load('labels006.npy').transpose((0,1,4,2,3))

img6 = np.stack([img6[...,0], img6[...,1], img6[...,1]], axis=-1)

x  = img6p[0,:,1]
hx = gaussian(21, 3.0)
x  = gputools.convolve_sep3(x, hx, hx, hx)
x  = lib.normalize_percentile_to01(x, 0, 100)
nhl,hyp  = lib.two_var_thresh(x)

t = np.linspace(0.95, .999, 30)
cpal = sns.cubehelix_palette(30)
f1 = plt.figure(1)
f2 = plt.figure(2)
def compute(i):
  th = t[i]
  seg = label(x>th)[0]
  n = seg.max()
  f1.gca().plot(th, n, 'o', c=cpal[i])
  ids,cts = np.unique(seg, return_counts=True)
  f2.gca().plot(np.log2(sorted(cts)), c=cpal[i])
for i in range(len(t)):
  compute(i)


