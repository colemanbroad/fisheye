# import IPython
import sys
import os
import shutil
from subprocess import run
from glob import glob

from tabulate import tabulate
from collections import Counter
from math import ceil

import numpy as np
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ion()
from tifffile import imread, imsave
from scipy.ndimage import zoom, label, distance_transform_edt, rotate
from scipy.signal import gaussian
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import watershed
# import scipy.ndimage as nd
import networkx as nx
import skimage.io as io

from segtools import cell_view_lib as view
from segtools import color
from segtools import lib as seglib
from segtools import spima
from segtools import segtools_simple as ss
from segtools import plotting

import spimagine
import gputools

def perm(arr,p1,p2):
  "permutation mapping p1 to p2 for use in numpy.transpose. elems must be unique."
  assert len(p1)==len(p2)
  perm = list(range(len(p1)))
  for i,p in enumerate(p2):
    perm[i] = p1.index(p)
  return arr.transpose(perm)

def timewindow(lst, t, l):
  "window of fixed length l, into list lst. try to center around t."
  if t < l//2: t=l//2
  if t >= len(lst) - l//2: t=len(lst) - ceil(l/2)
  return lst[t-l//2:t+ceil(l/2)]

def qopen():
  run(['rsync', 'efal:qsave.npy .'])
  return np.load('qsave.npy')

def updateall(w,lab):
  for i in range(lab.shape[0]):
    spima.update_spim(w,i,lab[i])
