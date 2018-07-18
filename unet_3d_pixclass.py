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

def lab2instance(x, d):
  x[x!=d['nuc']] = 0
  x = label(x)[0]
  return x

def build_rawdata(homedir):
  homedir = Path(homedir)

  def condense_labels(lab, d):
    lab[lab==0]   = d['bg']
    lab[lab==255] = d['mem']
    lab[lab==168] = d['nuc']
    lab[lab==85]  = d['div']
    lab[lab==198] = d['ignore']
    return lab

  ## load data
  # img = imread(str(homedir / 'data/img006.tif'))
  img = np.load(str(homedir / 'data/img006_noconv.npy'))
  img = perm(img,"TZCYX", "TZYXC")
  imgsem = {'axes':"TZYXC", 'nuc':0, 'mem':1}

  # img = np.load(str(homedir / 'isonet/restored.npy'))
  # img = img[np.newaxis,:352,...]
  # lab = np.load(str(homedir / 'data/labels_iso_t0.npy'))
  lab = imread(str(homedir / 'data/labels_lut.tif'))
  lab = lab[:,:,0]
  labsem = {'n_classes':3, 'div':3, 'nuc':1, 'mem':0, 'bg':2, 'ignore':1, 'axes':'TZYX'}

  lab = condense_labels(lab, labsem)
  ## TODO: this will break once we start labeling XZ and YZ in same volume.
  mask_labeled_slices = lab.min((2,3)) < 2
  inds_labeled_slices = np.indices(mask_labeled_slices.shape)[:,mask_labeled_slices]
  gt_slices = np.array([lab2instance(x, labsem) for x in lab[inds_labeled_slices[0], inds_labeled_slices[1]]])

  res = dict()
  res['img'] = img
  res['imgsem'] = imgsem
  res['lab'] = lab
  res['labsem'] = labsem
  res['mask_labeled_slices'] = mask_labeled_slices
  res['inds_labeled_slices'] = inds_labeled_slices
  res['gt_slices'] = gt_slices

  return res

def compute_weights(rawdata):
  lab = rawdata['lab']
  labsem = rawdata['labsem']
  mask_labeled_slices = rawdata['mask_labeled_slices']
  inds_labeled_slices = rawdata['inds_labeled_slices']

  weight_stack = np.zeros(lab.shape,dtype=np.float)
  
  weight_stack[inds_labeled_slices[0], inds_labeled_slices[1]] = 1 ## all the pixels that have been looked at by a human set to 1.

  ignore = labsem['ignore']
  ## turn off the `ignore` class
  weight_stack[lab==ignore] = 0

  ## weight higher near object borders
  for i in range(len(inds_labeled_slices[0])):
    t = inds_labeled_slices[0,i]
    z = inds_labeled_slices[1,i]
    lab_tz = lab[t,z]
    mask = lab_tz==class_semantics()['mem']
    distimg = ~mask
    distimg = distance_transform_edt(distimg)
    distimg = np.exp(-distimg/10)
    print(distimg.mean())
    weight_stack[t,z] = distimg/distimg.mean()

  if False: ## debug
    ws = weight_stack[inds_labeled_slices[0], inds_labeled_slices[1]]
    qsave(collapse2(splt(ws[:30],6,0),'12yx','1y,2x'))

  return weight_stack

def build_trainable(rawdata):
  img = rawdata['img']
  lab = rawdata['lab']

  cha = channel_semantics()
  cla = class_semantics()

  def f_xsem():
    d = dict()
    d['n_channels'] = 2
    d['mem'] = 0
    d['nuc'] = 1
    d['rgb']  = [d['mem'], d['nuc'], d['nuc']]
    return d
  xsem = f_xsem()
  ysem = {'n_channels':3, 'nuc':1, 'mem':0, 'bg':2, 'weight_channel_ytrue':True, 'shape':(None,None,None,3)}

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

  res = shuffle_split({'xs':xs,'ys':ys,'ws':ws})
  res['xsem'] = xsem
  res['ysem'] = ysem
  res['slices'] = slices
  res['slices0'] = slices0
  return res

def build_net(xsem, ysem):
  unet_params = {
    'n_pool' : 2,
    'n_convolutions_first_layer' : 16,
    'dropout_fraction' : 0.2,
    'kern_width' : 5,
  }
  input0 = Input((None, None, None, xsem['n_channels']))
  unet_out = unet.get_unet_n_pool(input0, **unet_params)
  output1 = unet.acti(unet_out, ysem['n_channels'])
  net = Model(inputs=input0, outputs=output2)

  optim = Adam(lr=1e-4)

  if ysem['weight_channel_ytrue'] == True:
    loss = unet.weighted_categorical_crossentropy(classweights=classweights, itd=0)
    def met0(y_true, y_pred):
      return losses.accuracy(y_true[...,:-1], y_pred)
  else:
    loss  = unet.my_categorical_crossentropy(classweights=classweights, itd=0)
    met0 = losses.accuracy

  
  net.compile(optimizer=optim, loss={'B':losses.mean_absolute_error}, metrics={'B':met0}) # metrics=['accuracy'])
  return net



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


## divisions and results

def max_z_divchan(pimg, ysem, savepath=None):
  "max proj across z, then merg across time"
  ch_div = ysem['div']
  res = merg(pimg[:,...,ch_div].max(1),0)
  io.imsave(savepath / 'max_z_divchan.png', res)
  return res

def show_results(pimg, rawdata, trainable, savepath=None):
  img = rawdata['img']
  lab = rawdata['lab']
  inds = rawdata['inds_labeled_slices']
  imgsem = rawdata['imgsem']
  labsem = rawdata['labsem']
  xsem = trainable['xsem']
  ysem = trainable['ysem']
  rgbimg = [imgsem['mem'], imgsem['nuc'], imgsem['nuc']]
  rgblab = [labsem['mem'], labsem['nuc'], labsem['bg']]
  rgbys = [ysem['mem'], ysem['nuc'], ysem['bg']]

  x = img[inds[0], inds[1]][...,rgbimg]
  x[...,2] = 0 # remove blue
  y = lab[inds[0], inds[1]]
  y = np_utils.to_categorical(y).reshape(y.shape + (-1,))
  y = y[...,rglab]
  z = pimg[inds[0], inds[1]][...,rgbys]
  ss = [slice(None,None,5), slice(None,None,4), slice(None,None,4), slice(None)]
  def f(r):
    r = merg(r[ss])
    r = r / np.percentile(r, 99, axis=(0,1), keepdims=True)
    r = np.clip(r,0,1)
    return r
  x,y,z = f(x), f(y), f(z)
  res = np.concatenate([x,y,z], axis=1)
  if savepath:
    io.imsave(savepath / 'results_labeled_slices.png', res)

  ## plot zx view for subset of y and t indices
  yinds = np.floor(np.linspace(0,399,8)).astype(np.int)
  x = pimg[::2,:,yinds][...,rgbys]
  y = img[::2,:,yinds][...,rgbimg]
  mi,ma = np.percentile(y,[2,99],axis=(1,2,3),keepdims=True)
  y = np.clip((y-mi)/(ma-mi),0,1)
  y[...,2] = 0
  x = np.stack([x,y],0)
  x = collapse2(x, "itzyxc","yz,tix,c")
  x = zoom(x.astype(np.float32), (5.0,1.0,1.0), order=1) ## upscale z axis by 5 for isotropic sampling
  if savepath:
    io.imsave(savepath / 'results_zx.png', x)

  ## plot zy view for subset of x and t indices
  xinds = np.floor(np.linspace(0,399,8)).astype(np.int)
  x = pimg[::2,:,:,xinds][...,rgbys]
  y = img[::2,:,:,xinds][...,rgbimg]
  mi,ma = np.percentile(y,[2,99],axis=(1,2,3),keepdims=True)
  y = np.clip((y-mi)/(ma-mi),0,1)
  y[...,2] = 0
  x = np.stack([x,y],0)
  x = collapse2(x, "itzyxc","xz,tiy,c")
  x = zoom(x.astype(np.float32), (5.0,1.0,1.0), order=1) ## upscale z axis by 5 for isotropic sampling
  if savepath:
    io.imsave(savepath / 'results_zy.png', x)

  return res

def find_divisions(pimg, ysem, savepath=None):
  ch_div = ysem['div']
  rgbdiv = [ysem['div'], ysem['nuc'], ysem['bg']]
  x = pimg.astype(np.float32)
  x = x[:,::6] ## downsample z
  div = x[...,ch_div].sum((2,3))
  val_thresh = np.percentile(div.flat, 95)
  n_rows, n_cols = 7, min(7,x.shape[0])
  tz = np.argwhere(div > val_thresh)[:n_rows]
  lst = list(range(x.shape[0]))
  x2 = np.array([x[timewindow(lst, n[0], n_cols), n[1]] for n in tz])
  x2 = collapse(x2, [[0,2],[1,3],[4]]) # '12yxc' -> '[1y][2x][c]'
  x2 = x2[::4,::4,rgbdiv]
  # x2[...,0] *= 40 ## enhance division channel color!
  # x2 = np.clip(x2, 0, 1)
  if savepath:
    io.imsave(savepath / 'find_divisions.png', x2)
  return x2

history = """

## Fri Jul 13 16:28:38 2018

Move unet3D to it's own module.



"""