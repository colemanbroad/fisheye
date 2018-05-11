## python defaults
import sys
import os
import shutil
from subprocess import run
from glob import glob
from collections import Counter
from math import ceil,floor
import json
import pickle
import random
import re

## python 3 only
from pathlib import Path

## stuff I've had to install
from tabulate import tabulate

## anaconda defaults
import networkx as nx
import numpy as np
import pandas as pd
from tifffile import imread, imsave
from scipy.ndimage import zoom, label, distance_transform_edt, rotate
from scipy.signal import gaussian
from scipy.ndimage.morphology import binary_dilation
from skimage.morphology import watershed
import skimage.io as io

## my own stuff
from segtools import color
from segtools import lib as seglib
from segtools import segtools_simple as ss
from segtools import patchmaker as patch
from segtools import voronoi
from segtools import plotting


def perm(arr,p1,p2):
  "permutation mapping p1 to p2 for use in numpy.transpose. elems must be unique."
  assert len(p1)==len(p2)
  perm = list(range(len(p1)))
  for i,p in enumerate(p2):
    perm[i] = p1.index(p)
  return arr.transpose(perm)

def timewindow(lst, t, l):
  "window of fixed length l into list lst. try to center around t."
  if t < l//2: t=l//2
  if t >= len(lst) - l//2: t=len(lst) - ceil(l/2)
  return lst[t-l//2:t+ceil(l/2)]

def qsave(x):
  np.save('qsave', x)

def ensure_exists(dir):
  try:
    os.makedirs(dir)
  except FileExistsError as e:
    print(e)