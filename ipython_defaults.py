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
import itertools
from time import time

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
  assert l <= len(lst)
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

flatten = lambda l: [item for sublist in l for item in sublist]

def run_from_ipython():
  "https://stackoverflow.com/questions/5376837/how-can-i-do-an-if-run-from-ipython-test-in-python"  
  try:
      __IPYTHON__
      return True
  except NameError:
      return False

def collapse(arr, axes=[[0,2],[1,3]]):
  sh = arr.shape
  perm = flatten(axes)
  arr = arr.transpose(perm)
  newshape = [np.prod([sh[i] for i in ax]) for ax in axes]
  arr = arr.reshape(newshape)
  return arr

def merg(arr, ax=0):
  "given a list of axes, merge each one with it's successor."
  if type(ax) is list:
    assert all(ax[i] <= ax[i+1] for i in range(len(ax)-1))
    for i,axis in zip(range(100),ax):
      arr = merg(arr, axis-i)
  else: # int type  
    assert ax < arr.ndim-1
    sh = list(arr.shape)
    n = sh[ax]
    del sh[ax]
    sh[ax] *= n
    arr = arr.reshape(sh)
  return arr

def splt(arr, s1=10, ax=0):
  """
  split an array into more dimensions
  takes a list of split values and a list of axes and divides each axis into two new axes,
  where the first has a length given by the split value (which must by an even divisor of the original axis length)
  res = arange(200).reshape((2,100))
  res = splt(res, 5, 1)
  res.shape == (4,5,20)

  res = arange(3*5*7*11).reshape((3*5,7*11))
  res = splt(res, [3,7],[0,1])
  res.shape == (3,5,7,11)

  you can even list the same dimension multiple times
  res = arange(3*5*7*11).reshape((3*5*7,11))
  res = splt(res, [3,5],[0,0])
  res.shape == (3,5,7,11)
  """
  sh = list(arr.shape)
  if type(s1) is list and type(ax) is list:
    assert all(ax[i] <= ax[i+1] for i in range(len(ax)-1))
    for i,spl,axis in zip(range(100),s1, ax):
      arr = splt(arr, spl, axis+i)
  elif type(s1) is int and type(ax) is int:
    s2,r = divmod(sh[ax], s1)
    assert r == 0
    sh[ax] = s2
    sh.insert(ax, s1)
    arr = arr.reshape(sh)
  return arr

def multicat(lst):
  if type(lst[0]) is list:
    # type of list is list of list of...
    # apply recursively to each element. then apply to result.
    # apply recursively to every element except last
    res = [multicat(l) for l in lst[:-1]] + [lst[-1]]
    res = multicat(res)
  else:
    # lst is list of ndarrays. return an ndarray.
    res = np.concatenate(lst[:-1], axis=lst[-1])
  return res

def multistack(lst):
  if type(lst[0]) is list:
    # type of list is list of list of...
    # apply recursively to each element. then apply to result.
    # apply recursively to every element except last
    res = [multistack(l) for l in lst[:-1]] + [lst[-1]]
    res = multistack(res)
  # elif type(lst[0]) is int:
  #   # lst is list of ndarrays. return an ndarray.
  #   res = np.stack(lst[:-1], axis=lst[-1])
  else:
    res = np.stack(lst[:-1], axis=lst[-1])
  return res

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def add_numbered_directory(path, base):
  s = re.compile(base + r"(\d{3})")
  def f(dir):
    m = s.search(dir)
    return int(m.groups()[0])
  drs = [f(d) for d in os.listdir(path) if s.search(d)]
  new_number = 0 if len(drs)==0 else max(drs) + 1
  newpath = str(path) + '/' + base + '{:03d}'.format(new_number)
  newpath = Path(newpath)
  newpath.mkdir(exist_ok=False)
  return newpath
