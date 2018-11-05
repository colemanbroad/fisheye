from segtools.defaults.ipython import *
from segtools.defaults.training import *

import lib

import gputools
import ipdb
import pandas as pd

from contextlib import redirect_stdout

import train_seg_lib as ts
patch = patchmaker



def test_watershed(homedir, savedir):
  img = imread("")  

def build_rawdata(homedir):
  img = imread("/net/fileserver-nfs/stornext/snfs2/projects/myersspimdata/Coleman/Celegans/HisGFP+BTub-GFP_12Oct2013_spec2_try1/render.tif")
  # img = np.load(str(homedir / 'data/img006_noconv.npy'))
  img = perm(img,"TZCYX", "TZYXC")

  r = 2 ## xy downsampling factor
  imgsem = {'axes':"TZYXC", 'nuc':0, 'mem':1, 'n_channels':2, 'r':r} ## image semantics

  # build point-detection gt
  points = lib.mkpoints()
  cen = np.zeros(img.shape[1:-1])

  sig = 10
  wid = 60
  def f(x): return np.exp(-(x*x).sum()/(2*sig**2))
  kern = math_utils.build_kernel_nd(wid,3,f)
  kern = kern[::3] ## anisotropic kernel matches img
  kern = kern / kern.sum()

  if True:
    cen[list(points.T)] = 1
    cen2 = fftconvolve(cen, kern, mode='same')
  
  if False:
    A = np.newaxis
    padding = np.array(kern.shape)
    padding = padding[:,A]
    padding = padding[:,[0,0]] // 2
    cen = np.pad(cen,padding,mode='constant')
    border  = np.array(kern.shape)
    starts = points
    ends = starts + border[A,:]
    for ss in patch.starts_ends_to_slices(starts, ends):
      cen[ss]=kern
    ss = patch.se2slices(padding[:,0],-padding[:,1])
    cen2 = cen[ss]

  # ipdb.set_trace()
  res = dict()
  res['img'] = img[:,:,::r,::r]
  res['imgsem'] = imgsem
  res['kern'] = kern
  res['cellcenters'] = cen2[:,::r,::r]
  return res