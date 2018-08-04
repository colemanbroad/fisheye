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

  ## turn off the `ignore` class
  if False:
    ignore = labsem['ignore']
    weight_stack[lab==ignore] = 0

  ## weight higher near object borders
  if False:
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



  xsem = {'n_channels':2, 'mem':0, 'nuc':1, 'shape':(None,None,None,2)}
  ysem = {'n_channels':3, 'nuc':1, 'mem':0, 'bg':2, 'weight_channel_ytrue':True, 
          'shape':(None,None,None,3), 'classweights':[1/3]*3,}

  weight_stack = compute_weights(rawdata)

  ## build slices
  patchsize = (1,64,128,128)
  xsem['patchsize'] = patchsize

  borders = (0,12,12,12)
  res = patchmaker.patchtool({'img':img.shape[:-1], 'patch':patchsize, 'overlap_factor':(1,1,1,1), 'borders':borders})
  slices = res['slices_padded']
  xsem['patchsize'] = patchsize
  xsem['borders'] = borders
  xsem['res_patches'] = res

  ## pad images to work with slices
  if True:
    padding = [(b,b) for b in borders] + [(0,0)]
    img = np.pad(img, padding, 'constant')
    lab = np.pad(lab, padding[:-1], 'constant', constant_values=labsem['bg'])
    weight_stack = np.pad(weight_stack, padding[:-1], 'constant')

  ## extract slices and build xs,ys,ws
  slices_filtered = [s for s in slices if (lab[s][0,12:-12]<2).sum() > 0] ## filter out all slices without much training data
  slices_filtered = slices_filtered[:20]
  xs = np.array([img[ss][0] for ss in slices_filtered])
  ys = np.array([lab[ss][0] for ss in slices_filtered])
  ws = np.array([weight_stack[ss][0] for ss in slices_filtered])
  ys = np_utils.to_categorical(ys).reshape(ys.shape + (-1,))

  ## normalize over space. sample and channel independent
  xs = xs/np.mean(xs,(1,2,3), keepdims=True)

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
    'n_pool' : 3,
    'n_convolutions_first_layer' : 16,
    'dropout_fraction' : 0.2,
    'kern_width' : 5,
  }

  mul = 2**unet_params['n_pool']
  fac = [factors(x) for x in xsem['patchsize'][1:-1]]
  for x in fac: assert mul in x

  input0 = Input(xsem['shape'])
  unet_out = unet.get_unet_n_pool(input0, **unet_params)
  output0 = unet.acti(unet_out, ysem['n_channels'], last_activation='softmax', name='B')

  net = Model(inputs=input0, outputs=output0)

  optim = Adam(lr=1e-4)

  ss = [slice(None), slice(2,-2), slice(2,-2), slice(2,-2), slice(None)]

  if ysem['weight_channel_ytrue'] == True:
    loss = unet.weighted_categorical_crossentropy()
    def met0(y_true, y_pred):
      return metrics.categorical_accuracy(y_true[...,:-1], y_pred)
  else:
    loss = unet.my_categorical_crossentropy()
    met0 = metrics.categorical_accuracy

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
    n_patches = res.shape[1]
    r, c = max(1, n_patches//5), 5
    res = splt(res[:,:r*c], r, 1)
    res = collapse2(res, 'iRCyxc','Ry,Cix,c')
    return res

  def mid(arr,i):
    ss = [slice(None) for _ in arr.shape]
    ss[i] = arr.shape[i]//2
    return arr[ss]

  def doit(i):
    res1 = plot(mid(xs_train,i), mid(ys_train,i))
    res2 = plot(mid(xs_vali,i), mid(ys_vali,i))
    if i in {2,3}:
      res1 = zoom(res1, (5,1,1), order=1)
      res2 = zoom(res2, (5,1,1), order=1)
    io.imsave(savepath / 'dat_train_{:d}.png'.format(i), res1)
    io.imsave(savepath / 'dat_vali_{:d}.png'.format(i), res2)

  doit(1) # z
  doit(2) # y
  doit(3) # x

def predict_trainvali(net, trainable, savepath):
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
    n_patches = res.shape[1]
    r, c = max(1, n_patches//5), 5
    res = splt(res[:,:r*c], r, 1)
    res = collapse2(res, 'iRCyxc','Ry,Cix,c')
    return res

  def mid(arr,i):
    ss = [slice(None) for _ in arr.shape]
    ss[i] = arr.shape[i]//2
    return arr[ss]

  def doit(i):
    res1 = plot(mid(xs_train,i), mid(ys_train,i), mid(pred_xs_train,i))
    res2 = plot(mid(xs_vali,i), mid(ys_vali,i), mid(pred_xs_vali,i))
    if i in {2,3}:
      res1 = zoom(res1, (5,1,1), order=1)
      res2 = zoom(res2, (5,1,1), order=1)
    io.imsave(savepath / 'pred_train_{:d}.png'.format(i), res1)
    io.imsave(savepath / 'pred_vali_{:d}.png'.format(i), res2)

  doit(1) # z
  doit(2) # y
  doit(3) # x

def predict(net,img,xsem,ysem):
  container = np.zeros(img.shape[:-1] + (ysem['n_channels'],))
  
  # borders = [0,4,0,0]
  # patchshape_padded = [1,24,400,400]
  borders = xsem['borders']
  patchshape_padded = list(xsem['patchsize'])
  patchshape_padded[2] = 400
  patchshape_padded[3] = 400
  padding = [(b,b) for b in borders] + [(0,0)]

  patches = patchmaker.patchtool({'img':container.shape[:-1], 'patch':patchshape_padded, 'borders':borders})
  # patches = xsem['res_patches']
  img = np.pad(img, padding, mode='constant')

  s2 = patches['slice_patch']
  for i in range(len(patches['slices_padded'])):
    s1 = patches['slices_padded'][i]
    s3 = patches['slices_valid'][i]
    x = img[s1]
    x = x / x.mean((1,2,3), keepdims=True)
    # x = collapse2(x, 'szyxc','s,y,x,zc')
    container[s3] = net.predict(x)[s2]

  return container

## divisions and results

def max_z_divchan(pimg, ysem, savepath=None):
  "max proj across z, then merg across time"
  ch_div = ysem['div']
  res = merg(pimg[:,...,ch_div].max(1),0)
  io.imsave(savepath / 'max_z_divchan.png', res)
  return res

def show_results(pimg, rawdata, trainable, savepath):
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

  def norm(img):
    # img = img / img.mean() / 5
    axis = tuple(np.arange(len(img.shape)-1))
    mi,ma = img.min(axis,keepdims=True), img.max(axis,keepdims=True)
    # mi,ma = np.percentile(img, [5,95])
    img = (img-mi)/(ma-mi)
    img = np.clip(img, 0, 1)
    return img

  x = img[inds[0], inds[1]][...,rgbimg]
  x[...,2] = 0 # remove blue
  y = lab[inds[0], inds[1]]
  y = np_utils.to_categorical(y).reshape(y.shape + (-1,))
  y = y[...,rgblab]
  z = pimg[inds[0], inds[1]][...,rgbys]
  ss = [slice(None,None,5), slice(None,None,4), slice(None,None,4), slice(None)]
  def f(r):
    r = merg(r[ss])
    r = r / np.percentile(r, 99, axis=(0,1), keepdims=True)
    r = np.clip(r,0,1)
    return r
  x,y,z = f(x), f(y), f(z)
  res = np.concatenate([x,y,z], axis=1)
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
  io.imsave(savepath / 'results_zy.png', x)

  return res

def show_results2(pimg, rawdata, trainable, savepath=None):
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

  # pred_xs_train = net.predict(xs_train, batch_size=1)
  # pred_xs_vali  = net.predict(xs_vali, batch_size=1)

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
    n_patches = res.shape[1]
    r, c = max(1, n_patches//5), 5
    res = splt(res[:,:r*c], r, 1)
    res = collapse2(res, 'iRCyxc','Ry,Cix,c')
    return res

  def mid(arr,i):
    ss = [slice(None) for _ in arr.shape]
    ss[i] = arr.shape[i]//2
    return arr[ss]

  def doit(i):
    res1 = plot(mid(xs_train,i), mid(ys_train,i), mid(pred_xs_train,i))
    res2 = plot(mid(xs_vali,i), mid(ys_vali,i), mid(pred_xs_vali,i))
    if i in {2,3}:
      res1 = zoom(res1, (5,1,1), order=1)
      res2 = zoom(res2, (5,1,1), order=1)
    io.imsave(savepath / 'pred_train_{:d}.png'.format(i), res1)
    io.imsave(savepath / 'pred_vali_{:d}.png'.format(i), res2)

  doit(1) # z
  doit(2) # y
  doit(3) # x



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

## Thu Jul 19 21:41:55 2018

I've got memory problems. These problems can only be fixed by saving xs and ys to disk.
Then our generator will load them from disk.
The memory problems were caused by an obvious

## Mon Jul 30 16:54:50 2018

Updated to use patchtool and to make pred_ and show_ trainvali consistent with 2d case.

## Thu Aug  2 13:35:11 2018

We have so little training data that there are only 20 samples for the 3D net.
This makes the loss trajectories very noisy!
It also means that we can overfit on the validation data by selecting based on best vali score!
For very small vali sets we might want to select models based on a combination of training and validation loss?
Also, we might want to train multiple models and allow training to run for a long time to explore the noise well?
Or... we just want to annotate more training examples.

Strange: Very low loss, but the pix class predictions don't look very good! How can this be?
Also bad instance seg scores... maybe my scoring is wrong...

## Sat Aug  4 16:25:54 2018

Setting the pixel weights to flat 0/1 per slice doesn't help the problem.
The loss is low (0.08/0.1) but the predictions look weak and uncertain. How can this be?
The weights weren't being used!
Not even the basic slice weights!

OK! After talking to Uwe I've realized that the actually differences between the 2D and 3D models might be very small.
They are similar in that channels are combined via a *learned, linear combination* which is like a convolution with a wide kernel?
OK. the model does a convolution which performs a learned, linear combination across all channels and across spatial neighbors.
And usually we learn twice the number of conv kernels with each downsampling.
If z were spatial and not in channels then we'd have to have a 3D convolution which had a kernel width across all of z?
And the boundary conditions would have to be 'same' such that the size of the z dimension after conv was the same as the size pre conv.
Then we double the number of true channels and we still double the size.
No! we would just use a large z width and have a convolution kernel the size of zdim!
Then we predict on many zslices simultaneously.
This is the same gain in efficiency that we get when moving from a classifier which works per-pixel to one which works per-plane.
The difference is that now our training labels come in large blocks.
The separation between training and vali data is more coarse.
With per-slice models we can have adjacent planes which are split: 1 in train, 1 in vali.
Naturally, because SPIM images have strong spatial variations both in image quality, microscope model and content we expect labels to be strongly locally correlated.
AND the input is strongly spatially correlated.
AND the transformation mapping in->out is strongly locally correlated! (appreciate that is a separate point.)

"""