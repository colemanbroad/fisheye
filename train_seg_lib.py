from segtools.defaults.ipython_remote import *
# from ipython_remote_defaults import *
from segtools.defaults.training import *
from segtools.numpy_utils import collapse2

import gputools
from contextlib import redirect_stdout
import ipdb
import pandas as pd


## utility functions (called internally)

def lab2instance(x):
  d = class_semantics()
  x[x!=d['nuc']] = 0
  x = label(x)[0]
  return x

def random_augmentation_xy(xpatch, ypatch, train=True):
  if random.random()<0.5:
      xpatch = np.flip(xpatch, axis=1)
      ypatch = np.flip(ypatch, axis=1)
  # if random.random()<0.5:
  #     xpatch = np.flip(xpatch, axis=0)
  #     ypatch = np.flip(ypatch, axis=0)
  if random.random()<1:
      # randangle = (random.random()-0.5)*60 # even dist between ± 30
      randangle = 90 * random.randint(0,3)
      xpatch  = rotate(xpatch, randangle, reshape=False, mode='reflect')
      ypatch  = rotate(ypatch, randangle, reshape=False, mode='reflect')
  # if train:
  #   if random.random()<0.5:
  #       delta = np.random.normal(loc=0, scale=5, size=(2,3,3))
  #       xpatch = augmentation.warp_multichan(xpatch, delta=delta)
  #       ypatch = augmentation.warp_multichan(ypatch, delta=delta)
  #   if random.random()<0.5:
  #       m = random.random()*xpatch.mean() # channels already normed
  #       s = random.random()*xpatch.std()
  #       noise = np.random.normal(m/4,s/4,xpatch.shape).astype(xpatch.dtype)
  #       xpatch += noise
  xpatch = xpatch.clip(min=0)
  xpatch = xpatch/xpatch.mean((0,1))
  return xpatch, ypatch

def convolve_zyx(pimg, axes="tzyxc"):
  assert pimg.ndim == 5
  pimg = perm(pimg, axes, "tzyxc")
  weights = np.full((3, 3, 3), 1.0/27)
  pimg = [[convolve(pimg[t,...,c], weights=weights) for c in range(pimg.shape[-1])] for t in range(pimg.shape[0])]
  pimg = np.array(pimg)
  pimg = pimg.transpose((0,2,3,4,1))
  pimg = [[convolve(pimg[t,...,c], weights=weights) for c in range(pimg.shape[-1])] for t in range(pimg.shape[0])]
  pimg = np.array(pimg)
  pimg = pimg.transpose((0,2,3,4,1))
  return pimg

## param definitions

def channel_semantics():
  d = dict()
  d['mem'] = 1
  d['nuc'] = 0
  d['n_channels'] = 2
  d['rgb'] = [d['mem'], d['nuc'], d['nuc']]
  return d

def class_semantics():
  d = dict()
  d['n_classes'] = 3
  d['div'] =  1
  d['nuc'] =  1
  d['mem'] =  0
  d['bg'] =  2
  d['ignore'] =  3
  d['rgb.mem'] =  [d['mem'], d['nuc'], d['bg']]
  d['rgb.div'] =  [d['div'], d['mem'], d['div']]
  d['classweights'] = [1/d['n_classes'],]*d['n_classes']
  return d

def segparams():
  # stack_segmentation_function = lambda x,p : stackseg.flat_thresh_two_chan(x, **p)
  # segmentation_params = {'nuc_mask':0.51, 'mem_mask':0.1}
  # segmentation_info  = {'name':'flat_thresh_two_chan', 'param0':'nuc_mask', 'param1':'mem_mask'}

  # stack_segmentation_function = lambda x,p : stackseg.watershed_memdist(x, **p)
  # segmentation_space = {'nuc_mask' :ho.hp.uniform('nuc_mask', 0.3, 0.7),
  #                        'nuc_seed':ho.hp.uniform('nuc_seed', 0.9, 1.0),
  #                        'mem_mask':ho.hp.uniform('mem_mask', 0.0, 0.3),
  #                        'dist_cut':ho.hp.uniform('dist_cut', 0.0, 10),
  #                        # 'mem_seed':hp.uniform('mem_seed', 0.0, 0.3),
  #                        'compactness' : 0, #hp.uniform('compactness', 0, 10),
  #                        'connectivity': 1, #hp.choice('connectivity', [1,2,3]),
  #                        }
  # segmentation_params = {'dist_cut': 9.780390266161866, 'mem_mask': 0.041369622211090654, 'nuc_mask': 0.5623125724186882, 'nuc_seed': 0.9610987296474213}

  stack_segmentation_function = lambda x,p : stackseg.watershed_two_chan(x, **p)
  segmentation_params = {'nuc_mask': 0.51053067684188, 'nuc_seed': 0.9918831824422522} ## SEG:  0.7501725367486776
  segmentation_space = {'nuc_mask' :ho.hp.uniform('nuc_mask', 0.3, 0.7),
                        'nuc_seed':ho.hp.uniform('nuc_seed', 0.9, 1.0), 
                        }
  segmentation_info  = {'name':'watershed_two_chan', 'param0':'nuc_mask', 'param1':'nuc_seed'}

  res = dict()
  res['function'] = stack_segmentation_function
  res['params'] = segmentation_params
  res['space'] = segmentation_space
  res['info'] = segmentation_info
  res['n_evals'] = 40 ## must be greater than 2 or hyperopt throws errors
  res['blur'] = False
  return res

## data loading and network training

def load_rawdata(homedir):
  homedir = Path(homedir)

  chansem = channel_semantics()
  cs = class_semantics()

  def condense_labels(lab):
    d = cs
    lab[lab==0]   = d['bg']
    lab[lab==255] = d['mem']
    lab[lab==168] = d['nuc']
    lab[lab==85]  = d['div']
    lab[lab==198] = d['nuc'] #d['ignore']
    return lab

  ## load data
  # img = imread(str(homedir / 'data/img006.tif'))
  img = np.load(str(homedir / 'data/img006_noconv.npy'))
  img = perm(img,"TZCYX", "TZYXC")

  # img = np.load(str(homedir / 'isonet/restored.npy'))
  # img = img[np.newaxis,:352,...]
  # lab = np.load(str(homedir / 'data/labels_iso_t0.npy'))
  lab = imread(str(homedir / 'data/labels_lut.tif'))
  lab = lab[:,:,0]
  # lab = lab[np.newaxis,:352,...]

  lab = condense_labels(lab)
  ## TODO: this will break once we start labeling XZ and YZ in same volume.
  mask_labeled_slices = lab.min((2,3)) < 2
  inds_labeled_slices = np.indices(mask_labeled_slices.shape)[:,mask_labeled_slices]
  gt_slices = np.array([lab2instance(x) for x in lab[inds_labeled_slices[0], inds_labeled_slices[1]]])

  res = dict()
  res['img'] = img
  res['lab'] = lab
  res['mask_labeled_slices'] = mask_labeled_slices
  res['inds_labeled_slices'] = inds_labeled_slices
  res['gt_slices'] = gt_slices
  return res

def compute_weights(rawdata):
  lab = rawdata['lab']
  mask_labeled_slices = rawdata['mask_labeled_slices']
  inds_labeled_slices = rawdata['inds_labeled_slices']

  weight_stack = np.zeros(lab.shape,dtype=np.float)
  
  weight_stack[inds_labeled_slices[0], inds_labeled_slices[1]] = 1 ## all the pixels that have been looked at by a human set to 1.

  ignore = class_semantics()['ignore']
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
    distimg = np.exp(-distimg/50)
    print(distimg.mean())
    weight_stack[t,z] = distimg/distimg.mean()

  if False: ## debug
    ws = weight_stack[inds_labeled_slices[0], inds_labeled_slices[1]]
    qsave(collapse2(splt(ws[:30],6,0),'12yx','1y,2x'))

  return weight_stack #[inds_labeled_slices[0], inds_labeled_slices[1]]

## models!!!

@DeprecationWarning
def build_trainable(rawdata):
  img = rawdata['img']
  lab = rawdata['lab']
  mask_labeled_slices = rawdata['mask_labeled_slices']
  inds_labeled_slices = rawdata['inds_labeled_slices']

  border_weights = False

  chansem = channel_semantics()
  clasem = class_semantics()

  def f_xsem():
    d = dict()
    dz = 2
    d['dz'] = dz
    d['n_channels'] = chansem['n_channels']*(1+2*dz)
    d['mem'] = 2*dz # original, centered membrane channelk
    d['nuc'] = 2*dz + 1
    d['rgb']  = [d['mem'], d['nuc'], d['nuc']]
    return d
  xsem = f_xsem()
  
  ys = lab[mask_labeled_slices].copy()
  xs = []
  for t,z in inds_labeled_slices.T:
    res = unet.add_z_to_chan(img[t], xsem['dz'], ind=z)
    xs.append(res[0])
  xs = np.array(xs)
  xs = perm(xs, "scyx", "syxc")

  # convert ys to categorical
  ys = np_utils.to_categorical(ys).reshape(ys.shape + (-1,))

  # add distance channel?
  if False:
    if dist_channel:
      a,b,c,d = ys.shape
      mask = ys[...,clasem['mem']]==1
      distimg = ~mask
      distimg = np.array([distance_transform_edt(d) for d in distimg])
      # distimg = np.exp(-distimg/10)
      mask = ys[...,clasem['bg']]==1
      distimg[mask] *= -1
      # distimg = distimg/distimg.mean((1,2), keepdims=True)
      ys = distimg[...,np.newaxis]
      # ys = ys / ys.mean((1,2,3), keepdims=True)

  # split 400x400 into 16x100x100 patches (if r=4) and reshape back into "SYXC"
  # axes are "SYXC"
  r = 2
  xs = collapse(splt(xs,[r,r],[1,2]), [[0,1,3],[2],[4],[5]])
  ys = collapse(splt(ys,[r,r],[1,2]), [[0,1,3],[2],[4],[5]])

  # np.savez(savepath / 'traindat_small', xs=xs[::5], ys=ys[::5])

  ## turn off the `ignore` class
  # mask = ys[...,3]==1
  # ys[mask] = 0

  ## reweight border pixels
  n_sample, n_y, n_x, _ = xs.shape
  ws = np.ones(xs.shape[:-1])
  ws = ws / ws.mean((1,2), keepdims=True)
  if border_weights:
    a,b,c,d = ys.shape
    mask = ys[...,clasem['mem']]==1 # ????
    distimg = ~mask
    distimg = np.array([distance_transform_edt(d) for d in distimg])
    distimg = np.exp(-distimg/10)
    ws = distimg/distimg.mean((1,2), keepdims=True)

  classweights = [1/clasem['n_classes'],]*clasem['n_classes']
  print(xs.shape, ys.shape, mask_labeled_slices.shape)
  print(ys.max((0,1,2)))

  ## normalize
  xs = xs/xs.mean((1,2), keepdims=True)

  ## shuffle
  inds = np.arange(xs.shape[0])
  np.random.shuffle(inds)
  invers = np.argsort(np.arange(inds.shape[0])[inds])
  xs = xs[inds]
  ys = ys[inds]
  ws = ws[inds]

  ## train vali split
  split = 5
  n_vali = xs.shape[0]//split
  xs_train = xs[:-n_vali]
  ys_train = ys[:-n_vali]
  ws_train = ws[:-n_vali]
  xs_vali  = xs[-n_vali:]
  ys_vali  = ys[-n_vali:]
  ws_vali  = ws[-n_vali:]
  
  unet_params = {
    'n_pool' : 2,
    'inputchan' : xsem['n_channels'],
    'n_classes' : clasem['n_classes'],
    'n_convolutions_first_layer' : 16,
    'dropout_fraction' : 0.2,
    'kern_width' : 3,
    'ndim' : 2,
  }
  net = unet.get_unet_n_pool(**unet_params)

  # loss  = unet.my_categorical_crossentropy(weights=classweights, itd=0)

  res = dict()
  res['xs_train'] = xs_train
  res['xs_vali'] = xs_vali
  res['ys_train'] = ys_train
  res['ys_vali'] = ys_vali
  res['ws_train'] = ws_train
  res['ws_vali'] = ws_vali
  res['classweights'] = classweights
  res['xsem'] = xsem
  res['net'] = net
  # res['loss'] = loss

  return res

def build_trainable3D(rawdata):
  img = rawdata['img']
  lab = rawdata['lab']
  mask_labeled_slices = rawdata['mask_labeled_slices']
  inds_labeled_slices = rawdata['inds_labeled_slices']

  border_weights = False

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

  weight_stack = compute_weights(rawdata)

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

  # add distance channel?
  if False:
    if dist_channel:
      a,b,c,d = ys.shape
      mask = ys[...,cla['mem']]==1
      distimg = ~mask
      distimg = np.array([distance_transform_edt(d) for d in distimg])
      # distimg = np.exp(-distimg/10)
      mask = ys[...,cla['bg']]==1
      distimg[mask] *= -1
      # distimg = distimg/distimg.mean((1,2), keepdims=True)
      ys = distimg[...,np.newaxis]
      # ys = ys / ys.mean((1,2,3), keepdims=True)
  
  print(xs.shape, ys.shape, mask_labeled_slices.shape)

  unet_params = {
    'n_pool' : 2,
    'inputchan' : xsem['n_channels'],
    'n_classes' : cla['n_classes'],
    'n_convolutions_first_layer' : 16,
    'dropout_fraction' : 0.2,
    'kern_width' : 5,
    'ndim' : 3,
  }
  net = unet.get_unet_n_pool(**unet_params)

  res = shuffle_split({'xs':xs,'ys':ys,'ws':ws})
  res['net'] = net
  res['xsem'] = xsem
  res['slices'] = slices
  res['slices0'] = slices0
  return res

def build_trainable2D(rawdata):
  img = rawdata['img']
  lab = rawdata['lab']
  mask_labeled_slices = rawdata['mask_labeled_slices']
  inds_labeled_slices = rawdata['inds_labeled_slices']

  border_weights = False

  cha = channel_semantics()
  cla = class_semantics()

  def f_xsem():
    d = dict()
    dz = 2
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
  if True:
    imgshape0 = img.shape
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

  xs = collapse(xs, [[0],[2],[3],[1,4]]) ## szyxc
  ys = ys[:,xsem['dz']] ## szyxc
  ws = ws[:,xsem['dz']] ## szyx

  # add distance channel?
  if False:
    if dist_channel:
      a,b,c,d = ys.shape
      mask = ys[...,cla['mem']]==1
      distimg = ~mask
      distimg = np.array([distance_transform_edt(d) for d in distimg])
      # distimg = np.exp(-distimg/10)
      mask = ys[...,cla['bg']]==1
      distimg[mask] *= -1
      # distimg = distimg/distimg.mean((1,2), keepdims=True)
      ys = distimg[...,np.newaxis]
      # ys = ys / ys.mean((1,2,3), keepdims=True)
  
  print(xs.shape, ys.shape, mask_labeled_slices.shape)
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

  ## totally boilerplate

  ## compute weights

def build_trainable2D_2(rawdata):
  img = rawdata['img']
  lab = rawdata['lab']
  mask_labeled_slices = rawdata['mask_labeled_slices']
  inds_labeled_slices = rawdata['inds_labeled_slices']

  border_weights = False

  cha = channel_semantics()
  cla = class_semantics()

  def f_xsem():
    d = dict()
    dz = 2
    d['dz'] = dz
    d['nzdim'] = 2*dz+1
    d['n_channels'] = cha['n_channels']*d['nzdim']
    d['mem'] = 2*dz # original, centered membrane channelk
    d['nuc'] = 2*dz + 1
    d['rgb']  = [d['mem'], d['nuc'], d['nuc']]
    return d
  xsem = f_xsem()

  weight_stack = compute_weights(rawdata)

  ## padding
  if True:
    imgshape0 = img.shape
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

  xs = collapse(xs, [[0],[2],[3],[1,4]]) ## szyxc
  ys = ys[:,xsem['dz']] ## szyxc
  ws = ws[:,xsem['dz']] ## szyx

  # add distance channel?
  if True:
    if dist_channel:
      a,b,c,d = ys.shape
      mask = ys[...,cla['mem']]==1
      distimg = ~mask
      distimg = np.array([distance_transform_edt(d) for d in distimg])
      # distimg = np.exp(-distimg/10)
      mask = ys[...,cla['bg']]==1
      distimg[mask] *= -1
      # distimg = distimg/distimg.mean((1,2), keepdims=True)
      ys = distimg[...,np.newaxis]
      # ys = ys / ys.mean((1,2,3), keepdims=True)
  
  print(xs.shape, ys.shape, mask_labeled_slices.shape)
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


## utils n stuff

def shuffle_split(xsysws):
  xs = xsysws['xs']
  ys = xsysws['ys']
  ws = xsysws['ws']

  ## shuffle
  inds = np.arange(xs.shape[0])
  np.random.shuffle(inds)
  invers = np.argsort(np.arange(inds.shape[0])[inds])
  xs = xs[inds]
  ys = ys[inds]
  ws = ws[inds]

  ## train vali split
  split = 5
  n_vali = xs.shape[0]//split
  xs_train = xs[:-n_vali]
  ys_train = ys[:-n_vali]
  ws_train = ws[:-n_vali]
  xs_vali  = xs[-n_vali:]
  ys_vali  = ys[-n_vali:]
  ws_vali  = ws[-n_vali:]

  res = dict()
  res['xs_train'] = xs_train
  res['xs_vali'] = xs_vali
  res['ys_train'] = ys_train
  res['ys_vali'] = ys_vali
  res['ws_train'] = ws_train
  res['ws_vali'] = ws_vali

  return res

def train(trainable, savepath):
  xs_train = trainable['xs_train']
  ys_train = trainable['ys_train']
  ws_train = trainable['ws_train']
  xs_vali = trainable['xs_vali']
  ys_vali = trainable['ys_vali']
  ws_vali = trainable['ws_vali']
  cla_sem = class_semantics()
  classweights = cla_sem['classweights']
  net = trainable['net']

  batchsize = 3
  n_epochs  = 100

  stepsperepoch   = xs_train.shape[0] // batchsize
  validationsteps = xs_vali.shape[0] // batchsize

  def print_stats():
    stats = dict()
    stats['batchsize'] = batchsize
    stats['n_epochs'] = n_epochs
    stats['stepsperepoch'] = stepsperepoch
    stats['validationsteps'] = validationsteps
    stats['n_samples_train'] = xs_train.shape[0]
    stats['n_samples_vali'] = xs_vali.shape[0]
    stats['xs_mean'] = xs_train.mean()
    print(stats)
  print_stats()

  optim = Adam(lr=1e-4)
  # loss  = unet.my_categorical_crossentropy(classweights=classweights, itd=0)
  loss  = unet.weighted_categorical_crossentropy(classweights=classweights, itd=0)
  ys_train = np.concatenate([ys_train, ws_train[...,np.newaxis]], -1)
  ys_vali = np.concatenate([ys_vali, ws_vali[...,np.newaxis]], -1)
  
  net.compile(optimizer=optim, loss=loss, metrics=['accuracy'])
  checkpointer = ModelCheckpoint(filepath=str(savepath / "w001.h5"), verbose=0, save_best_only=True, save_weights_only=True)
  earlystopper = EarlyStopping(patience=30, verbose=0)
  reduce_lr    = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
  callbacks = [checkpointer, earlystopper, reduce_lr]

  # f_trainaug = lambda x,y:random_augmentation_xy(x,y,train=True)
  f_trainaug = lambda x,y:(x,y)

  bgen = unet.batch_generator_patches_aug(xs_train, ys_train,
                                    # steps_per_epoch=100,
                                    batch_size=batchsize,
                                    augment_and_norm=f_trainaug,
                                    savepath=savepath)

  # f_valiaug = lambda x,y:random_augmentation_xy(x,y,train=False)
  f_valiaug = lambda x,y:(x,y)

  vgen = unet.batch_generator_patches_aug(xs_vali, ys_vali,
                                    # steps_per_epoch=100,
                                    batch_size=batchsize,
                                    augment_and_norm=f_valiaug,
                                    savepath=None)

  history = net.fit_generator(bgen,
                    steps_per_epoch=stepsperepoch,
                    epochs=n_epochs,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=vgen,
                    validation_steps=validationsteps)

  net.save_weights(savepath / 'w002.h5')
  return history

## predictions from trained network

@DeprecationWarning
def predict_on_new(net,img,xsem):
  assert img.ndim == 5 ## TZYXC
  stack = []
  dz = xsem['dz']
  ntimes = img.shape[0]
  for t in range(ntimes):
    timepoint = []
    # for ax in ["ZYXC", "YXZC", "XZYC"]:
    for ax in ["ZYXC"]: #, "YXZC", "XZYC"]:
      x = perm(img[t], "ZCYX", ax)
      x = unet.add_z_to_chan(x, dz, axes="ZYXC") # lying about direction of Z!
      print(x.shape)
      x = x / x.mean((1,2), keepdims=True)
      pimg = net.predict(x, batch_size=1)
      pimg = pimg.astype(np.float16)
      pimg = perm(pimg, ax, "ZYXC")
      # qsave(collapse(splt(pimg[:64,::4,::4,rgbdiv],8,0),[[0,2],[1,3],[4]]))
      timepoint.append(pimg)
    timepoint = np.array(timepoint)
    timepoint = timepoint.mean(0)
    stack.append(timepoint)
  stack = np.array(stack)
  return stack

def predict_on_new_2D(net,img,xsem):
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
  ss_container = patch.starts_ends_to_sclices(idx_start_container,idx_end_container)
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

def predict_on_new_3D(net,img,xsem):
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

@DeprecationWarning
def predict_on_new_3D_old(net,img):
  assert img.ndim == 5 ## TZYXC
  # slices = patch.
  stack = []
  ntimes = img.shape[0]
  for t in range(ntimes):
    timepoint = []
    # for ax in ["ZYXC", "YXZC", "XZYC"]:
    for ax in ["ZYXC"]: #, "YXZC", "XZYC"]:
      x = perm(img[t], "ZCYX", ax)
      # x = unet.add_z_to_chan(x, dz, axes="ZYXC") # lying about direction of Z!
      slices = patch.slices_grid(x.shape[:-1], (64,128,128), coverall=False)
      b = 20
      slices_all = patch.make_triplets(slices, (b,b,b))
      # ipdb.set_trace()
      pimg = np.zeros(x.shape[:-1] + (3,))
      x = np.pad(x, [(b,b)]*3 + [(0,0)], 'constant')
      count = 0
      for si,so,sc in slices_all:
        count += 1
        p = x[si]
        p = p / p.mean((0,1,2), keepdims=True)
        pimg[sc] = net.predict(p[np.newaxis], batch_size=1)[0][so]
      pimg = pimg.astype(np.float16)
      pimg = perm(pimg, ax, "ZYXC")
      # qsave(collapse(splt(pimg[:64,::4,::4,rgbdiv],8,0),[[0,2],[1,3],[4]]))
      timepoint.append(pimg)
    timepoint = np.array(timepoint)
    timepoint = timepoint.mean(0)
    stack.append(timepoint)
  stack = np.array(stack)
  return stack

def predict_train_vali(trainable, savepath=None):
  net = trainable['net']
  xs_train = trainable['xs_train']
  xs_vali = trainable['xs_vali']
  rgbmem = class_semantics()['rgb.mem']
  pred_xs_train = net.predict(xs_train, batch_size=1)
  pred_xs_vali = net.predict(xs_vali, batch_size=1)
  
  if xs_train.ndim == 5:
    pred_xs_train = pred_xs_train[:,14] #.max(1) # max proj across z
    pred_xs_vali  = pred_xs_vali[:,14] #.max(1) # max proj across z

  rows, cols = rowscols(pred_xs_train.shape[0], 8)
  collshape = [[0,2],[1,3],[4]]
  res1 = collapse(splt(pred_xs_train[:cols*rows,...,rgbmem],rows,0), collshape)
  rows, cols = rowscols(pred_xs_vali.shape[0], 5)
  res2 = collapse(splt(pred_xs_vali[:cols*rows,...,rgbmem],rows,0), collshape)
  
  if savepath: io.imsave(savepath / 'pred_xs_train.png', res1)
  if savepath: io.imsave(savepath / 'pred_xs_vali.png', res2)
  return res1, res2

## plotting

def show_trainvali(trainable, savepath=None):
  xs_train = trainable['xs_train']
  xs_vali  = trainable['xs_vali']
  ys_train = trainable['ys_train']
  ys_vali  = trainable['ys_vali']
  ws_train = trainable['ws_train']
  ws_vali  = trainable['ws_vali']
  xsem = trainable['xsem']

  if xs_train.ndim==5:
    middle_z = xs_train.shape[1]//2
    middle_z = slice(13,17)
    xs_train = xs_train[:,middle_z].max(1) #np.max(ws_train[...,np.newaxis]*xs_train, axis=1) # max over z dim
    xs_vali  = xs_vali[:,middle_z].max(1)  #np.max(ws_vali[...,np.newaxis]*xs_vali, axis=1)   # max over z dim
    ys_train = ys_train[:,middle_z].max(1) #np.max(ws_train[...,np.newaxis]*ys_train, axis=1) # max over z dim
    ys_vali  = ys_vali[:,middle_z].max(1)  #np.max(ws_vali[...,np.newaxis]*ys_vali, axis=1)   # max over z dim

  c = class_semantics()
  d = channel_semantics()
  e = trainable['xsem']

  sx = [slice(None,xs_vali.shape[0]), Ellipsis, e['rgb']]
  sy = [slice(None,xs_vali.shape[0]), Ellipsis, c['rgb.mem']]
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

def plot_history(history, savepath=None):
  if savepath:
    print(history.history, file=open(savepath / 'history.txt','w'))
  def plot_hist_key(k):
    plt.figure()
    y1 = history.history[k]
    y2 = history.history['val_' + k]
    plt.plot(y1, label=k)
    plt.plot(y2, label='val_' + k)
    plt.legend()
    if savepath:
      plt.savefig(savepath / (k + '.png'))
  plot_hist_key('loss')
  plot_hist_key('acc')

def max_z_divchan(pimg, savepath=None):
  "max proj across z, then merg across time"
  ch_div = class_semantics()['div']
  res = merg(pimg[:,...,ch_div].max(1),0)
  io.imsave(savepath / 'max_z_divchan.png', res)
  return res

def show_results(pimg, rawdata, savepath=None):
  img = rawdata['img']
  lab = rawdata['lab']
  inds = rawdata['inds_labeled_slices']
  rgbimg = channel_semantics()['rgb']
  rgbmem = class_semantics()['rgb.mem']
  cs = class_semantics()
  rgbimg = [cs['mem'],cs['nuc'],cs['nuc']]

  x = img[inds[0], inds[1]][...,rgbimg]
  x[...,2] = 0 # remove blue
  y = lab[inds[0], inds[1]]
  y = np_utils.to_categorical(y).reshape(y.shape + (-1,))
  y = y[...,rgbmem]
  z = pimg[inds[0], inds[1]][...,rgbmem]
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
  x = pimg[::2,:,yinds][...,rgbmem]
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
  x = pimg[::2,:,:,xinds][...,rgbmem]
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

def find_divisions(pimg, savepath=None):
  ch_div = class_semantics()['div']
  rgbdiv = class_semantics()['rgb.div']
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

## run-style functions requiring

def run_predict_on_new(loaddir, savedir):
  unet_params = json.load(open(loaddir / 'unet_params.json', 'r'))
  net = unet.get_unet_n_pool(**unet_params)
  net.load_weights(str(loaddir / 'w001.h5'))
  img = np.load('data/img006_noconv.npy')
  dz = channel_semantics()['dz']
  pimg = predict_on_new(net, img, dz)
  np.save(savedir / 'pimg', pimg)
  loaddir = savedir
  print("PREDICT COMPLETE")

def optimize_segmentation(pimg, rawdata, segparams, mypath_opt):
  lab  = rawdata['lab']
  inds = rawdata['inds_labeled_slices'][:,:-4]
  ## recompute gt_slices! don't use gt_slices from rawdata
  gt_slices = np.array([lab2instance(x) for x in lab[inds[0], inds[1]]])
  stack_segmentation_function = segparams['function']
  segmentation_space = segparams['space']
  segmentation_info = segparams['info']
  n_evals = segparams['n_evals']
  img_instseg = pimg[[0]]

  ## optimization params
  # mypath_opt = add_numbered_directory(savepath, 'opt')
  
  def risk(params):
    print('Evaluating params:',params)
    t0 = time()
    hyp = np.array([stack_segmentation_function(x,params) for x in img_instseg])
    hyp = hyp.astype(np.uint16)
    pred_slices = hyp[inds[0], inds[1]]
    res = np.array([ss.seg(x,y) for x,y in zip(gt_slices, pred_slices)])
    t1 = time()
    val = res.mean()
    print("SEG: ", val)
    return -val

  ## perform the optimization

  trials = ho.Trials()
  best = ho.fmin(risk,
      space=segmentation_space,
      algo=ho.tpe.suggest,
      max_evals=n_evals,
      trials=trials)

  pickle.dump(trials, open(mypath_opt / 'trials.pkl', 'wb'))
  print(best)

  losses = [x['loss'] for x in trials.results if x['status']=='ok']
  df = pd.DataFrame({**trials.vals})
  df = df.iloc[:len(losses)]
  df['loss'] = losses

  ## save the results

  def save_img():
    plt.figure()
    # plt.scatter(ps[0], ps[1], c=values)
    n = segmentation_info['name']
    p0 = segmentation_info['param0']
    p1 = segmentation_info['param1']
    x = np.array(trials.vals[p0])
    y = np.array(trials.vals[p1])
    c = np.array([t['loss'] for t in trials.results if t.get('loss',None) is not None])
    plt.scatter(x[:c.shape[0]],y[:c.shape[0]],c=c)
    plt.title(n)
    plt.xlabel(p0)
    plt.ylabel(p1)
    plt.colorbar()
    filename = '_'.join([n,p0,p1])
    plt.savefig(mypath_opt / (filename + '.png'))
  save_img()

  from hyperopt.plotting import main_plot_vars
  from hyperopt.plotting import main_plot_history
  from hyperopt.plotting import main_plot_histogram

  plt.figure()
  main_plot_histogram(trials=trials)
  plt.savefig(mypath_opt / 'hypopt_histogram.pdf')

  plt.figure()
  main_plot_history(trials=trials)
  plt.savefig(mypath_opt / 'hypopt_history.pdf')

  domain = ho.base.Domain(risk, segmentation_space)

  plt.figure()
  main_plot_vars(trials=trials, bandit=domain)
  plt.tight_layout()
  plt.savefig(mypath_opt / 'hypopt_vars.pdf')

  plt.figure()
  g = sns.PairGrid(df) #, hue="connectivity")
  # def hist(x, **kwargs):
  #   plt.hist(x, stacked=True, **kwargs)
  g.map_diag(plt.hist)
  g.map_upper(plt.scatter)
  g.map_lower(sns.kdeplot, cmap="Blues_d")
  # g.map(plt.scatter)
  g.add_legend();
  plt.savefig(mypath_opt / 'hypopt_seaborn_004.pdf')

  if False:
    plt.figure()
    ax = plt.subplot(gspec[0])
    sns.swarmplot(x='connectivity', y='loss', data=df, ax=ax)
    ax = plt.subplot(gspec[1])
    sns.swarmplot(x='nuc_mask', y='loss', data=df, ax=ax)
    ax = plt.subplot(gspec[2])
    sns.swarmplot(x='nuc_seed', y='loss', data=df, ax=ax)
    ax = plt.subplot(gspec[3])
    sns.swarmplot(x='compactness', y='loss', data=df, ax=ax)  
    plt.savefig(mypath_opt / 'hypopt_seaborn_005.pdf')

    fig = plt.figure(figsize=(16,4))
    gspec = matplotlib.gridspec.GridSpec(1,4) #, width_ratios=[3,1])
    # gspec.update(wspace=1, hspace=1)
    cmap = np.array(sns.color_palette('deep'))
    conn = df.connectivity.as_matrix()
    colors = cmap[conn.flat].reshape(conn.shape + (3,))
    ax0 = plt.subplot(gspec[0])
    ax0.scatter(df["compactness"], df.loss, c=colors)
    plt.ylabel('loss')

    ax1 = plt.subplot(gspec[1], sharey=ax0)
    ax1.scatter(df["nuc_mask"], df.loss, c=colors)
    plt.setp(ax1.get_yticklabels(), visible=False)

    ax2 = plt.subplot(gspec[2], sharey=ax0)
    ax2.scatter(df["nuc_seed"], df.loss, c=colors)
    plt.setp(ax2.get_yticklabels(), visible=False)
    
    ax3 = plt.subplot(gspec[3])
    sns.swarmplot(x="connectivity", y="loss", data=df, ax=ax3)
    plt.setp(ax3.get_yticklabels(), visible=False)
    plt.savefig(mypath_opt / 'hypopt_seaborn_006.pdf')

  return best

def compute_seg_on_slices(hyp, rawdata):
  print("COMPARE SEGMENTATION AGAINST LABELED SLICES")
  inds_labeled_slices = rawdata['inds_labeled_slices']
  lab = rawdata['lab']
  img = rawdata['img']
  inds = inds_labeled_slices[:,:-4] # use full
  gt_slices  = np.array([lab2instance(x) for x in lab[inds[0], inds[1]]])
  pre_slices = hyp[inds[0], inds[1]]
  seg_scores = np.array([ss.seg(x,y) for x,y in zip(gt_slices, pre_slices)])
  # print({'seg': seg_scores.mean(), 'std':seg_scores.std()}, file=open(savepath / 'SEG.txt','w'))
  return seg_scores

def analyze_hyp(hyp, rawdata, segparams, savepath):
  segmentation_params = segparams['params']
  stack_segmentation_function = segparams['function']
  inds = rawdata['inds_labeled_slices']
  lab = rawdata['lab']
  img = rawdata['img']
  ch_nuc_img = channel_semantics()['nuc']

  np.save(savepath / 'hyp', hyp)

  print("COMPARE SEGMENTATION AGAINST LABELED SLICES")
  seg_scores = compute_seg_on_slices(hyp, rawdata)
  print({'seg': seg_scores.mean(), 'std':seg_scores.std()}, file=open(savepath / 'SEG.txt','w'))
  
  print("CREATE DISPLAY GRID OF SEGMENTATION RESULTS")
  img_slices = img[inds[0], inds[1]]
  gt_slices  = np.array([lab2instance(x) for x in lab[inds[0], inds[1]]])
  pre_slices = hyp[inds[0], inds[1]]
  # container = np.zeros((1600,1600,3))
  # slices = patch.slices_heterostride((1600,1600),(400,400))
  gspec = matplotlib.gridspec.GridSpec(4, 4)
  gspec.update(wspace=0., hspace=0.)
  fig = plt.figure()
  ids = np.arange(img_slices.shape[0])
  # np.random.shuffle(ids)
  ids = np.concatenate([ids[:24:2],ids[-4:]])
  for i in range(16):
    ax = plt.subplot(gspec[i])
    im1 = img_slices[ids[i],...,ch_nuc_img]
    im2 = pre_slices[ids[i]]
    im3 = gt_slices[ids[i]]
    psg = ss.pixel_sharing_bipartite(im3, im2)
    matching = ss.matching_iou(psg, fraction=0.5)
    ax = plotting.make_comparison_image(im1,im2,im3,matching,ax=ax)
    ipdb.set_trace()
    # container[slices[i]] = ax
    # res.append(ax.get_array())
    ax.set_axis_off()

  fig.set_size_inches(10, 10, forward=True)
  plt.savefig(savepath / 'seg_overlay.png',dpi=200, bbox_inches='tight')
  
  print("INSTANCE SEGMENTATION ANALYSIS COMPLETE")

def build_nhl(hyp, rawdata, savepath):
  print("GENERATE NHLS FROM HYP AND ANALYZE")
  img = rawdata['img']
  ch_nuc_img = channel_semantics()['nuc']
  nhls = nhl_tools.labs2nhls(hyp, img[:,:,ch_nuc_img])
  pickle.dump(nhls, open(savepath / 'nhls.pkl', 'wb'))

  plt.figure()
  cm = sns.cubehelix_palette(len(nhls))
  for i,nhl in enumerate(nhls):
    areas = [n['area'] for n in nhl]
    areas = np.log2(sorted(areas))
    plt.plot(areas, '.', c=cm[i])
  plt.savefig(savepath / 'cell_sizes.pdf')
  print("INSTANCE ANALYSIS COMPLETE")
  return nhls

def track_nhls(nhls, savepath):
  assert len(nhls) > 1
  factory = tt.TrackFactory(knn_dub=50, edge_scale=50)
  alltracks = [factory.nhls2tracking(nhls[i:i+2]) for i in range(len(nhls)-1)]
  tr = tt.compose_trackings(factory, alltracks, nhls)
  with open(savepath / 'stats_tr.txt','w') as f:
    with redirect_stdout(f):
      tt.stats_tr(tr)
  cost_stats = '\n'.join(tt.cost_stats_lines(factory.graph_cost_stats))
  print(cost_stats, file=open(savepath / 'cost_stats.txt','w'))
  return tr



history = """
We want to move away from a script and to a library to work better with an interactive ipython.
How should we organize this library?
We want it to be able to work with job_submit.
We wanted a script so that we could compare the history of files easily.
But now its sufficiently complex that we need to refactor into simple functions.
This means we have to have a script that takes loaddir and savedir arguments.

`load_source_target`
Turn our raw image and label data into somehting that we can feed to a net for training.
Note. If we want to vary the size and shape of input and output to the network then we have to couple the network and the data loading.

model structure depends on input and output shape, but all optimization params are totally independent.
And we can apply any processing to image as long as it doesn't change the shape or channel semantics.
The *loss* depends strongly on the target shape and semantics.
We should define the loss with the target and the model.
What if we want an ensemble? What if we want multiple models? Should we allow training multiple models?
We should define an ensemble/model collection and allow functions that train ensembles?

Instance segmentation and hyper optimization are totally independent?

The shape of Xs and Ys and totally coupled to the model.
There are a few params we can adjust when building X and Y...
Input/Output patch size. (just make sure that it's divisible by 2**n_maxpool)
Most conv nets accept multiple different sizes of input patches.
We can adjust pixelwise weights and masks on the loss including slice masks and border weights.
weights are strict generalization of mask, or?

should split_train_vali be an inner function of load_source_target?
1. It does much more than split train and vali. It builds xs, ys, and ws given knowledge of the network.
2. Maybe we should separate gathering and cleaning the input image and labels with the task of building xs,ys,ws, and net.


## Sat Jun 30 09:08:10 2018

- trainable is built correctly from rawdata.

## Wed Jul  4 14:26:32 2018

- finish porting train_and_seg into train_seg_lib
  - can train a trainable and save results
  - can optimize a segmentation on a pimg w hyperopt
  - all functions have been ported, all the way through tracking.
  simple ipython scheme for calling train_seg_lib is trainseg_ipy.

TODO:
- use weights in loss
- build true 3D unet
  - new xs,ys,ws,net and predict
    - but keep rawdata, keep trainable as name for collection
- apply as patches to larger retina image
- tracking ground truth and hyperopt

1. We can hack the mask into the loss function without having to produce a *new* loss function
for every x,y pair if we just add the weights as an extra channel on ys_train, and ys_vali.
This now means that ys_train and ys_vali will have a different shape from the output of the network!
But only during training!
BUG: now the old computation of accuracy no longer works, because it required that ys_pred be a categorical one hot encoding.
now that this is done, do we need to test it's effectiveness?
(ignore for now)
TODO: set the weights to zero wherever we have an explicit "ignore" class from the annotations.

2. Let's make it 3D

This issue is complex enough that it deserves it's own notes...
see [unet_3d.md]

So the 3D unet is training, and the loss looks reasonable, but i don't yet know if it's good.
The network i'm currently trying is quite small. 353k params.

There are multiple layers of processing we may want to apply during training.
- whole dataset level
  - spacetime registration
  - calculating indices of annotated slices
  - compute gt_slices
- individual timepoint
  - cropping borders
  - calculating 3D distances or weights.
- individual z slices
  - 2d distance to boundaries
- xs, ys: training set level
  - pixelwise norm across samples?
- batch level
  - batch norm
- patch level
  - patch normalization

we can do dataset, timepoint, zslices, xs,ys, and patch all from within build_trainable.
We don't have access to patches *after* augmentation, nor to batches (which are generated dynamically).

Is there a way we can combine the 2D and 3D models?
There is much they have in common.
If we provide a simple slices-based interface for training we could combine them?
we can replace the tricky `add_z_to_channels` method with a slices-based approach.
first we compute the proper set of slices for the full stack. then we collapse z and channels.
we can probably replace that entire function.

also, it often seems like we want to do some padding of our image in combination with pulling slices from it.
should we combine padding and slices together into a single function?

I could use a padding setting that takes a patch size and the image size and returns the padding
vector that will make the imgsize a perfect multiple in every dimension.

OK. Now the training loss won't go below 2.5 which is terrible and the accuracy is 95, which is also terrible
because we haven't fixed the one-hot-encoding problem.
But the slices and padding all seem to be working nicely, which is a huge plus.

## Sun Jul  8 20:12:31 2018

Using the z-to-channels model with dz=2 we can get a val_loss of 0.11 and an accuracy of >86 and a 
seg score of 0.75 using no augmentation at all and no special weights.


## Mon Jul  9 18:53:04 2018

Really nice to have the 3D unet working.
Loss is low with flat weights. around 0.08.
The xz and yz views of classifier results is very helpful.
We can see how adjusting weights or network params makes it easier to identify boundaires separating instances.
We might want to take an active learning approach to segmentation and add new gt data guided by predictions.
- It makes sense to simply draw the boundaries between objects where they are needed!
The xz and yz views make this obvious!
We could even perform an automated segmentation on each slice w manual boundary addition.
- The simplest way to do guided annotation is just to only annotate the slices where it's clear the boundaries are wrong.
- ALSO. Let's try downscaling img and lab. The x,y resolution is too high. We can double our receptive field this way.

The loss got down to 0.05 in `test3` with kernel width = 5!

But the results in xz and yz are still inadequate for segmentation.
- (linearly or isonet) upscale *before* training.
- 

"""