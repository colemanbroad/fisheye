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
import keras.backend as K

import gputools
from contextlib import redirect_stdout
import ipdb
import pandas as pd

import train_seg_lib as ts


def build_trainable(rawdata):
  img = rawdata['img']
  lab = rawdata['lab']

  cha = ts.channel_semantics()
  cla = ts.class_semantics()

  def f_xsem():
    d = dict()
    d['n_channels'] = cha['n_channels']
    d['mem'] = 0
    d['nuc'] = 1
    d['rgb'] = [d['mem'], d['nuc'], d['nuc']]
    d['shape'] = (None, None, None, cha['n_channels'])
    # d['shape.re'] = 
    return d
  xsem = f_xsem()

  def f_ysem():
    d = dict()
    d['n_channels'] = 1
    d['gauss'] = 0
    d['rgb'] = [0,0,0]
    d['shape'] = (None, None, None, 1)
    return d
  ysem = f_ysem()

  weight_stack = ts.compute_weights(rawdata)

  ## add extra cell center channel
  cellcenters = rawdata['cellcenters']
  slices = patch.slices_heterostride(cellcenters.shape,(32,128,128),(30,2,2))
  xs = np.array([img[1][ss] for ss in slices])
  ys = np.array([cellcenters[ss] for ss in slices])
  ys = ys[...,np.newaxis]
  ws = np.array([weight_stack[1][ss] for ss in slices])

  ## normalize over space. sample and channel independent
  xs = xs/np.mean(xs,(1,2,3), keepdims=True)
  
  print(xs.shape, ys.shape, ws.shape)

  res = ts.shuffle_split({'xs':xs,'ys':ys,'ws':ws})
  net = build_net(xsem)
  res['net'] = net
  res['xsem'] = xsem
  res['ysem'] = ysem
  res['slices'] = slices
  res['predict'] = predict
  return res


def build_net(xsem):
  unet_params = {
    'n_pool' : 2,
    'n_convolutions_first_layer' : 16,
    'dropout_fraction' : 0.2,
    'kern_width' : 5,
  }

  input0 = Input(xsem['shape'])
  unet_out = unet.get_unet_n_pool(input0, **unet_params)
  output2  = unet.acti(unet_out, 1, last_activation='linear', name='B')

  net = Model(inputs=input0, outputs=output2)
  # net.layers[-1].add_loss(losses.mean_squared_error)

  optim = Adam(lr=1e-4)
  # loss  = unet.my_categorical_crossentropy(classweights=classweights, itd=0)
  # loss = unet.weighted_categorical_crossentropy(classweights=classweights, itd=0)
  # ys_train = np.concatenate([ys_train, ws_train[...,np.newaxis]], -1)
  # ys_vali  = np.concatenate([ys_vali, ws_vali[...,np.newaxis]], -1)
  def met0(y_true, y_pred):
    # mi,ma = np.percentile(y_pred,[2,98])
    # return ma-mi
    return K.std(y_pred)
  
  net.compile(optimizer=optim, loss={'B':losses.mean_absolute_error}, metrics={'B':met0}) # metrics=['accuracy'])
  return net

def show_trainvali(trainable, savepath=None):
  xs_train = trainable['xs_train']
  xs_vali  = trainable['xs_vali']
  ys_train = trainable['ys_train']
  ys_vali  = trainable['ys_vali']
  ws_train = trainable['ws_train']
  ws_vali  = trainable['ws_vali']
  xsem = trainable['xsem']
  ysem = trainable['ysem']


  middle_z = xs_train.shape[1]//2
  middle_z = slice(13,17)
  xs_train = xs_train[:,middle_z].max(1)
  xs_vali  = xs_vali[:,middle_z].max(1)
  ys_train = ys_train[:,middle_z].max(1)
  ys_vali  = ys_vali[:,middle_z].max(1)

  c = ts.class_semantics()
  d = ts.channel_semantics()

  sx = [slice(None,xs_vali.shape[0]), Ellipsis, xsem['rgb']]
  sy = [slice(None,xs_vali.shape[0]), Ellipsis, ysem['rgb']]
  xt = xs_train[sx]
  xv = xs_vali[sx]
  yt = ys_train[sy]
  yv = ys_vali[sy]
  xt = xt / xt.max()
  xv = xv / xv.max()
  yt = yt / yt.max()
  yv = yv / yv.max()
  xt[...,2] = 0 # turn off blue
  xv[...,2] = 0
  yt[...,2] = 0
  yv[...,2] = 0
  res = multicat([[xt,yt,2], [xv,yv,2], 2])
  res = res[:,::2,::2]
  res = merg(res, 0) #[[0,1],[2],[3]])
  if savepath:
    io.imsave(savepath / 'train_vali_ex.png', res)
  return res

def predict(net,img,xsem):
  clasem = ts.class_semantics()
  container = np.zeros(img.shape[:-1] + (clasem['n_classes'],))
  
  # ipdb.set_trace()
  sh_img = np.array(img.shape)
  sh_container = np.array((1,32,400,400))
  sh_grid = np.ceil(sh_img[:-1]/sh_container).astype(np.int)
  sh_grid[1] *= 2 ## double z coverage
  border = 10
  extra = [0,border,0,0]

  slices_container   = patch.slices_heterostride(sh_img[:-1], sh_container, sh_grid)
  triplets = patch.make_triplets(slices_container, extra)

  padding = [(0,0)]*5
  padding[1] = (border,border)
  img = np.pad(img,padding,mode='constant')

  for si,so,sc in triplets:
    x = img[si]
    x = x / x.mean((1,2,3))
    container[sc] = net.predict(x)[so]

  return container

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


history = """
## Thu Jul 12 18:01:50 2018

Moved into it's own file!

"""