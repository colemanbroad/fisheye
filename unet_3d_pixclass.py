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

# class unet3dpixclass(Object):

#   def __init__():
#     pass


def build_trainable(rawdata):
  img = rawdata['img']
  lab = rawdata['lab']

  cha = channel_semantics()
  cla = class_semantics()

  def f_xsem():
    d = dict()
    d['n_channels'] = cha['n_channels']
    d['mem'] = 0
    d['nuc'] = 1
    d['rgb']  = [d['mem'], d['nuc'], d['nuc']]
    return d
  xsem = f_xsem()

  weight_stack = ts.compute_weights(rawdata)

  ## padding
  padding = [(0,0)]*5
  padding[1] = (10,10)
  img = np.pad(img, padding, 'constant')
  lab = np.pad(lab, padding[:-1], 'constant', constant_values=cla['bg'])
  weight_stack = np.pad(weight_stack, padding[:-1], 'constant')

  ## extract slices and build xs,ys,ws
  slices0 = patch.slices_heterostride(lab.shape,(1,32,128,128),(11,30,2,2))
  slices = [s for s in slices0 if (lab[s][0,10:20]<2).sum() > 0] ## filter out all slices without much training data
  xs = np.array([img[ss][0] for ss in slices])
  ys = np.array([lab[ss][0] for ss in slices])
  ws = np.array([weight_stack[ss][0] for ss in slices])
  ys = np_utils.to_categorical(ys).reshape(ys.shape + (-1,))

  ## normalize over space. sample and channel independent
  xs = xs/np.mean(xs,(1,2,3), keepdims=True)
  
  print(xs.shape, ys.shape, ws.shape)

  unet_params = {
    'n_pool' : 2,
    # 'inputchan' : xsem['n_channels'],
    # 'n_classes' : 3, #cla['n_classes'],
    'n_convolutions_first_layer' : 16,
    'dropout_fraction' : 0.2,
    'kern_width' : 5,
    # 'ndim' : 3,
  }
  input0 = Input((None, None, None, xsem['n_channels']))
  unet_out = unet.get_unet_n_pool(input0, **unet_params)
  output1 = unet.acti(unet_out, 3) #cla['n_classes'])
  # output2 = unet.acti(unet_out, 1, last_activation='linear')
  # output2.add_loss()
  net = Model(inputs=input0, outputs=output2)

  res = shuffle_split({'xs':xs,'ys':ys,'ws':ws})
  res['net'] = net
  res['xsem'] = xsem
  res['slices'] = slices
  res['slices0'] = slices0
  return res

def predict_trainvali(trainable, savepath=None):
  net = trainable['net']
  xs_train = trainable['xs_train']
  xs_vali = trainable['xs_vali']
  xsem = trainable['xsem']
  ysem = trainable['ysem']

  rgb_ys = ysem['rgb']
  rgb_xs = xsem['rgb']
  pred_xs_train = net.predict(xs_train, batch_size=1)
  pred_xs_vali = net.predict(xs_vali, batch_size=1)

  x = xs_train[::10,...,rgb_xs]
  y = pred_xs_train[::10,...,rgb_ys]
  xy = np.stack([x,y],0)
  res1 = collapse2(xy, 'iszyxc', 'szy,ix,c')

  x = xs_vali[::10,...,rgb_xs]
  y = pred_xs_vali[::10,...,rgb_ys]
  xy = np.stack([x,y],0)
  res2 = collapse2(xy, 'iszyxc', 'szy,ix,c')

  def norm(img):
    img = img / img.mean() / 5
    img = np.clip(img, 0, 1)
    return img

  if savepath: io.imsave(savepath / 'pred_xs_train.png', norm(res1))
  if savepath: io.imsave(savepath / 'pred_xs_vali.png', norm(res2))

  return res1, res2

def predict(net,img,xsem):
  clasem = class_semantics()
  container = np.zeros(img.shape[:-1] + (clasem['n_classes'],))
  
  # ipdb.set_trace()
  sh_img = np.array(img.shape)
  sh_container  = np.array((1,20,400,400))
  extra = [0,10,0,0]

  slices   = patch.slices_heterostride(sh_img[:-1], sh_container, np.ceil(sh_img[:-1]/sh_container))
  triplets = patch.make_triplets(slices, extra)

  padding = [(0,0)]*5
  padding[1] = (10,10)
  img = np.pad(img,padding,mode='constant')

  for si,so,sc in triplets:
    x = img[si]
    x = x / x.mean((1,2,3))
    container[sc] = net.predict(x)[so]

  return container
