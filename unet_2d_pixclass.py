from segtools.defaults.ipython import *
from segtools.defaults.training import *

import lib

import gputools
import ipdb
import pandas as pd

from contextlib import redirect_stdout
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
  imgsem = {'axes':"TZYXC", 'nuc':0, 'mem':1, 'n_channels':2}

  # img = np.load(str(homedir / 'isonet/restored.npy'))
  # img = img[np.newaxis,:352,...]
  # lab = np.load(str(homedir / 'data/labels_iso_t0.npy'))
  lab = imread(str(homedir / 'data/labels_lut.tif'))
  lab = lab[:,:,0]
  labsem = {'n_classes':3, 'div':1, 'nuc':1, 'mem':0, 'bg':2, 'ignore':1, 'axes':'TZYX'}

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
    mask = lab_tz==labsem['mem']
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
  labsem = rawdata['labsem']

  dz = 2
  nzdim = 2*dz+1
  xsem = {'n_channels':2*nzdim, 'mem':2*dz, 'nuc':2*dz+1, 'shape':(None,None,2*nzdim), 'dz':dz, 'nzdim':nzdim}
  ysem = {'n_channels':3, 'nuc':1, 'mem':0, 'bg':2, 'weight_channel_ytrue':True, 
          'shape':(None,None,None,3), 'classweights':[1/3]*3,}

  weight_stack = compute_weights(rawdata)

  ## build slices
  patchsize = (1,nzdim,200,200)
  # assert 4 in factors(patchsize[1])
  res = patchmaker.patchtool({'img':img.shape[:-1], 'patch':patchsize, 'overlap_factor':(1,nzdim,1,1)})
  slices = res['slices']
  xsem['patchsize'] = patchsize
  # xsem['borders'] = borders

  borders = (0,dz,0,0)
  ## pad images to work with slices
  if True:
    padding = [(b,b) for b in borders] + [(0,0)]
    img = np.pad(img, padding, 'constant')
    lab = np.pad(lab, padding[:-1], 'constant', constant_values=labsem['bg'])
    weight_stack = np.pad(weight_stack, padding[:-1], 'constant')

  ## extract slices and build xs,ys,ws
  slices_filtered = [s for s in slices if (lab[s][0,xsem['dz']]<2).sum() > 0] ## filter out all slices without much training data
  # slices_filtered = slices_filtered[:20]
  xs = np.array([img[ss][0] for ss in slices_filtered])
  ys = np.array([lab[ss][0] for ss in slices_filtered])
  ws = np.array([weight_stack[ss][0] for ss in slices_filtered])
  ys = np_utils.to_categorical(ys).reshape(ys.shape + (-1,))

  ## normalize over space. sample and channel independent
  xs = xs/np.mean(xs,(1,2,3), keepdims=True)

  ## move z to channels
  xs = collapse2(xs, 'szyxc','s,y,x,zc') ## szyxc
  ys = ys[:,xsem['dz']] ## szyxc
  ws = ws[:,xsem['dz']] ## szyx

  if ysem['weight_channel_ytrue']:
    ys = np.concatenate([ys, ws[...,np.newaxis]], -1)
  
  print(xs.shape, ys.shape, ws.shape)

  res = ts.shuffle_split({'xs':xs,'ys':ys,'ws':ws})
  res['xsem'] = xsem
  res['ysem'] = ysem
  res['slices'] = slices
  res['slices_filtered'] = slices_filtered
  return res

def build_net(xsem, ysem):
  unet_params = {
    'n_pool' : 2,
    'n_convolutions_first_layer' : 16,
    'dropout_fraction' : 0.2,
    'kern_width' : 3,
  }

  input0 = Input(xsem['shape'])
  unet_out = unet.get_unet_n_pool(input0, **unet_params)
  output0 = unet.acti(unet_out, ysem['n_channels'], last_activation='softmax', name='B')

  ss = [slice(None), slice(2,-2), slice(2,-2), slice(2,-2), slice(None)]

  if ysem['weight_channel_ytrue'] == True:
    loss = unet.weighted_categorical_crossentropy()
    def met0(y_true, y_pred):
      return metrics.categorical_accuracy(y_true[...,:-1], y_pred)
  else:
    loss = unet.my_categorical_crossentropy()
    met0 = metrics.categorical_accuracy

  net = Model(inputs=input0, outputs=output0)
  optim = Adam(lr=1e-4)
  net.compile(optimizer=optim, loss={'B':loss}, metrics={'B':met0}) # metrics=['accuracy'])
  return net

def show_trainvali(trainable, savepath):
  xs_train = trainable['xs_train']
  xs_vali  = trainable['xs_vali']
  ys_train = trainable['ys_train']
  ys_vali  = trainable['ys_vali']
  ws_train = trainable['ws_train']
  ws_vali  = trainable['ws_vali']
  xsem = trainable['xsem']
  ysem = trainable['ysem']
  xrgb = [xsem['mem'], xsem['nuc'], xsem['nuc']]
  yrgb = [ysem['mem'], ysem['nuc'], ysem['bg']]

  def norm(img):
    # img = img / img.mean() / 5
    axis = tuple(np.arange(len(img.shape)-1))
    mi,ma = img.min(axis,keepdims=True), img.max(axis,keepdims=True)
    # mi,ma = np.percentile(img, [5,95])
    img = (img-mi)/(ma-mi)
    img = np.clip(img, 0, 1)
    return img

  def plot(xs, ys):
    xs = norm(xs[...,xrgb])
    ys = norm(ys[...,yrgb])
    xs[...,2] = 0
    res = np.stack([xs,ys],0)
    res = collapse2(res, 'isyxc','sy,ix,c')
    return res

  def doit(i):
    res1 = plot(xs_train, ys_train)
    res2 = plot(xs_vali, ys_vali)
    if i in {2,3}:
      res1 = zoom(res1, (5,1,1), order=1)
      res2 = zoom(res2, (5,1,1), order=1)
    io.imsave(savepath / 'dat_train_{:d}.png'.format(i), res1)
    io.imsave(savepath / 'dat_vali_{:d}.png'.format(i), res2)

  doit(1) # z

def predict_trainvali(net, trainable, savepath=None):
  xs_train = trainable['xs_train']
  xs_vali  = trainable['xs_vali']
  ys_train = trainable['ys_train']
  ys_vali  = trainable['ys_vali']
  ws_train = trainable['ws_train']
  ws_vali  = trainable['ws_vali']
  xsem = trainable['xsem']
  ysem = trainable['ysem']
  xrgb = [xsem['mem'], xsem['nuc'], xsem['nuc']]
  yrgb = [ysem['mem'], ysem['nuc'], ysem['bg']]

  pred_xs_train = net.predict(xs_train, batch_size=1)
  pred_xs_vali  = net.predict(xs_vali, batch_size=1)

  def norm(img):
    # img = img / img.mean() / 5
    axis = tuple(np.arange(len(img.shape)-1))
    mi,ma = img.min(axis,keepdims=True), img.max(axis,keepdims=True)
    # mi,ma = np.percentile(img, [5,95])
    img = (img-mi)/(ma-mi)
    img = np.clip(img, 0, 1)
    return img

  def plot(xs, ys, preds):
    xs = norm(xs[...,xrgb])
    ys = norm(ys[...,yrgb])
    preds = norm(preds[...,yrgb])
    xs[...,2] = 0
    res = np.stack([xs,ys,preds],0)
    res = collapse2(res, 'isyxc','sy,ix,c')
    return res

  def doit(i):
    res1 = plot(xs_train, ys_train, pred_xs_train)
    res2 = plot(xs_vali, ys_vali, pred_xs_vali)
    if i in {2,3}:
      res1 = zoom(res1, (5,1,1), order=1)
      res2 = zoom(res2, (5,1,1), order=1)
    io.imsave(savepath / 'pred_train_{:d}.png'.format(i), res1)
    io.imsave(savepath / 'pred_vali_{:d}.png'.format(i), res2)

  doit(1)
  return {'train':pred_xs_train, 'vali':pred_xs_vali}
  
def predict(net,img,xsem,ysem):
  container = np.zeros(img.shape[:-1] + (ysem['n_channels'],))
  
  dz = xsem['dz']

  borders = [0,dz,0,0]
  patchshape_padded = [1,1+2*dz,400,400]
  padding = [(b,b) for b in borders] + [(0,0)]

  patches = patchmaker.patchtool({'img':container.shape[:-1], 'patch':patchshape_padded, 'borders':borders})
  img = np.pad(img, padding, mode='constant')

  # s2 = patches['slice_patch']
  for i in range(len(patches['slices_padded'])):
    s1 = patches['slices_padded'][i]
    s3 = patches['slices_valid'][i]
    x = img[s1]
    x = x / x.mean((1,2,3), keepdims=True)
    x = collapse2(x, 'szyxc','s,y,x,zc')
    container[s3] = net.predict(x) #[s2]

  return container


history = """
## Wed Jul 25 17:47:07 2018

We have to deal with patches in a very different way in this problem.
We can't use patchtool in the normal way, because we all perform a collapse between
extracting the patches from the padded image and replacing them into the container.

But in effect the change of shape is really only like having a small z-padding!
We could deal with this in the normal way?

Full pimg predictions in predict are wack despite net training to 0.222 vali. What's wrong?

- [x] fixed channel independent norm in trainvali show funcs.

## Thu Jul 26 11:53:10 2018

Fixed normalization bug resulting in all-blue background predictions.

"""




