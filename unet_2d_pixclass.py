from segtools.defaults.ipython_remote import *
# from ipython_remote_defaults import *
from segtools.defaults.training import *
from segtools.numpy_utils import collapse2
from segtools import math_utils
from scipy.signal import fftconvolve
import lib

from keras.models import Model
from keras.layers import Input
from keras import losses

import gputools
from contextlib import redirect_stdout
import ipdb
import pandas as pd

import train_seg_lib as ts

def build_trainable2D(rawdata):
  img = rawdata['img']
  lab = rawdata['lab']

  cha = channel_semantics()
  cla = class_semantics()

  def f_xsem():
    d = dict()
    dz = 5
    d['dz'] = dz
    d['nzdim'] = 2*dz+1
    d['n_channels'] = cha['n_channels']*d['nzdim']
    d['mem'] = 2*dz # original, centered membrane channelk
    d['nuc'] = 2*dz + 1
    d['rgb']  = [d['mem'], d['nuc'], d['nuc']]
    return d
  xsem = f_xsem()

  ## compute weights
  weight_stack = compute_weights(rawdata)

  ## padding
  padding = [(0,0)]*5
  padding[1] = (xsem['dz'],xsem['dz'])
  img = np.pad(img, padding, 'constant')
  lab = np.pad(lab, padding[:-1], 'constant', constant_values=cla['bg'])
  weight_stack = np.pad(weight_stack, padding[:-1], 'constant')

  ## extract slices and build xs,ys,ws
  nzdim = xsem['nzdim']
  slices0 = patch.slices_heterostride(lab.shape,(1,nzdim,200,200),(11,75-nzdim+1,2,2))
  slices = [s for s in slices0 if (lab[s][0,xsem['dz']]<2).sum() > 0] ## filter out all slices without much training data
  xs = np.array([img[ss][0] for ss in slices])
  ys = np.array([lab[ss][0] for ss in slices])
  ws = np.array([weight_stack[ss][0] for ss in slices])
  ys = np_utils.to_categorical(ys).reshape(ys.shape + (-1,))

  ## normalize over space. sample and channel independent
  xs = xs/np.mean(xs,(1,2,3), keepdims=True)

  ## move z to channels
  xs = collapse(xs, [[0],[2],[3],[1,4]]) ## szyxc
  ys = ys[:,xsem['dz']] ## szyxc
  ws = ws[:,xsem['dz']] ## szyx
  
  print(xs.shape, ys.shape, ws.shape)
  print(ys.max((0,1,2)))

  unet_params = {
    'n_pool' : 2,
    'inputchan' : xsem['n_channels'],
    'n_classes' : cla['n_classes'],
    'n_convolutions_first_layer' : 16,
    'dropout_fraction' : 0.2,
    'kern_width' : 3,
    'ndim' : 2,
  }
  net = unet.get_unet_n_pool(**unet_params)

  res = shuffle_split({'xs':xs,'ys':ys,'ws':ws})
  res['net'] = net
  res['xsem'] = xsem
  res['slices'] = slices
  res['slices0'] = slices0
  return res

def predict(net,img,xsem):
  clasem = class_semantics()
  container = np.zeros(img.shape[:-1] + (clasem['n_classes'],))

  dz = xsem['dz']
  sh_img = np.array(img.shape)
  sh_container = np.array([1,1,400,400])
  extra_width  = np.array([0,2*dz,0,0])
  sh_grid  = np.ceil(sh_img[:-1] / sh_container).astype(np.int)

  X = np.newaxis
  idx_start_container = patch.starts(sh_grid, sh_container, sh_container).reshape((4,-1)).T
  idx_end_container   = idx_start_container + sh_container[X,:]
  idx_start_input, idx_end_input = idx_start_container, idx_end_container + extra_width[X,:]
  ss_container = patch.starts_ends_to_slices(idx_start_container,idx_end_container)
  ss_input = patch.starts_ends_to_slices(idx_start_input, idx_end_input)

  ## slice applied to output of net
  ss_output = patch.se2slices(idx_start_container[0],idx_end_container[0])
  del ss_output[1]

  padding = [(0,0)]*img.ndim
  dz = xsem['dz']
  padding[1] = (dz,dz)
  imgpad = np.pad(img,padding,mode='constant')

  for i in range(len(ss_container)):
    sc = ss_container[i]
    si = ss_input[i]
    x  = imgpad[si]
    ## "zyxc"
    x = x / x.mean((1,2,3))
    x = collapse(x, [[0],[2],[3],[1,4]])
    container[sc] = net.predict(x)[ss_output]

  return container