from segtools.defaults.ipython import *
from segtools.defaults.training import *

import lib

import gputools
import ipdb
import pandas as pd

from contextlib import redirect_stdout
import train_seg_lib as ts
from sklearn.metrics import confusion_matrix
from segtools import label_tools
import skimage.morphology as morph
import gputools

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
  img = perm(img, "TZCYX", "TZYXC")
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

def build_rawdata2(homedir):
  homedir = Path(homedir)
  ## load data
  # img = imread(str(homedir / 'data/img006.tif'))
  img = np.load(str(homedir / 'data/img006_noconv.npy'))
  img = img[1]
  img = perm(img, "ZCYX", "ZYXC")
  img = norm_szyxc_per(img,(0,1,2))

  imgsem = {'axes':"ZYXC", 'nuc':1, 'mem':0, 'n_channels':2}

  points = lib.mkpoints()
  cen = np.zeros(img.shape[:-1])
  cen[list(points.T)] = 1
  x = img[...,1]
  hx = np.array([1,1,1]) / 3
  x = gputools.convolve_sep3(x, hx, hx, hx)
  lab = watershed(-x, label(cen)[0], mask=x>x.mean())

  bor = label_tools.find_boundaries(lab)
  bor = np.array([morph.binary_dilation(b) for b in bor]) ## just x,y
  bor = np.array([morph.binary_dilation(b) for b in bor]) ## just x,y
  # bor = np.array([morph.binary_dilation(b) for b in bor]) ## just x,yp
  # bor = morph.binary_erosion(bor)
  # bor = morph.binary_dilation(bor)
  # bor = morph.binary_erosion(bor)
  # bor = morph.binary_dilation(bor)
  # bor = morph.binary_dilation(bor)
  lab[lab!=0] = 1
  lab[bor] = 2
  labsem = {'n_classes':3, 'nuc':1, 'mem':2, 'bg':0, 'axes':'ZYX'}

  res = dict()
  res['img'] = img
  res['imgsem'] = imgsem
  res['lab'] = lab
  res['labsem'] = labsem
  res['points'] = points
  res['cen'] = cen
  # res['mask_labeled_slices'] = mask_labeled_slices
  # res['inds_labeled_slices'] = inds_labeled_slices
  # res['gt_slices'] = gt_slices
  return res

def border_weights_full(rawdata):
  lab = rawdata['lab']
  labsem = rawdata['labsem']

  mask = lab==labsem['mem']
  distimg = ~mask
  distimg = distance_transform_edt(distimg)
  distimg = np.exp(-distimg/10)
  distimg[mask] *= 3  ## higher class weight for membrane (similar to distance weight)
  distimg = distimg / distimg.mean()
  return distimg

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

def simple_weights(rawdata):
  lab = rawdata['lab']
  weight_stack = np.ones(lab.shape)
  return weight_stack

def build_trainable(rawdata):
  img = rawdata['img']
  lab = rawdata['lab']
  labsem = rawdata['labsem']

  xsem = {'n_channels':2, 'mem':0, 'nuc':1, 
          'shape':(None,None,None,2),
          }
  ysem = {'n_channels':3, 'mem':2, 'nuc':1, 'bg':0,
          'shape' : (None,None,None,3),
          'classweights' : [1/3]*3,
          'weight_channel_ytrue' : True,
          }

  # weight_stack = compute_weights(rawdata)
  # weight_stack = simple_weights(rawdata)
  weight_stack = border_weights_full(rawdata)
  train_mask = np.random.rand(*lab.shape) > 1/6
  vali_mask = ~train_mask ## need to copy data because of zero padding later

  ## build slices
  patchsize = (1,32,104,104)
  patchsize = (32,104,104)

  xsem['patchsize'] = patchsize

  bw = 0
  borders = (0,bw,bw,bw)
  borders = (bw,bw,bw)
  overlap = (1,1,1,1)
  overlap = (1,1,1)
  patches = patchmaker.patchtool({'img':img.shape[:-1], 'patch':patchsize, 'overlap_factor':overlap, 'borders':borders})
  slices = patches['slices_padded']
  xsem['patchsize'] = patchsize
  xsem['borders'] = borders

  ## pad images to work with slices
  padding = [(b,b) for b in borders]
  padding_chan = padding + [(0,0)]
  img = np.pad(img, padding_chan, 'constant')
  lab = np.pad(lab, padding, 'constant', constant_values=labsem['bg'])
  weight_stack = np.pad(weight_stack, padding, 'constant')
  train_mask = np.pad(train_mask, padding, 'constant')
  vali_mask = np.pad(vali_mask, padding, 'constant')

  ## extract slices and build xs,ys,ws
  # slices_filtered = [s for s in slices if (lab[s][0,:]<2).sum() > 0] ## filter out all slices without much training data
  slices_filtered = slices
  def f(x): return x # x[0]
  xs = np.array([f(img[ss]) for ss in slices_filtered])
  ys = np.array([f(lab[ss]) for ss in slices_filtered])
  ys = np_utils.to_categorical(ys).reshape(ys.shape + (-1,))
  ws = np.array([f(weight_stack[ss]) for ss in slices_filtered])
  tm = np.array([f(train_mask[ss]) for ss in slices_filtered])
  vm = np.array([f(vali_mask[ss]) for ss in slices_filtered])
  print(xs.shape, ys.shape, ws.shape)

  ## normalize over space. sample and channel independent
  ax_zyx = (1,2,3)
  xs = xs/np.mean(xs, ax_zyx, keepdims=True)

  if ysem['weight_channel_ytrue']:
    ys = np.concatenate([ys, ws[...,np.newaxis]], -1)

  # res = ts.copy_split_mask_vali({'xs':xs,'ys':ys,'ws':ws,'tm':tm,'vm':vm})
  res = ts.shuffle_split({'xs':xs,'ys':ys,'ws':ws,'slices':slices})
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
    'kern_width' : 3,
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

  if ysem['weight_channel_ytrue']:
    loss = unet.weighted_categorical_crossentropy()
    def met0(y_true, y_pred):
      # y has axes "SZYXC"
      # yt = y_true[...,:-1]
      # ws = y_true[...,-1] > 0
      # print(ws.dtype)
      # # ss = np.s_[::2,::2,::2,::2] ## speed things up
      # print(ws.shape, y_pred.shape, yt.shape)
      # predlab = np.argmax(y_pred, axis=-1)[ws]
      # truelab = np.argmax(yt, axis=-1)[ws]
      # return confusion_matrix(truelab.flat, predlab.flat)
      # return (predlab==truelab).sum() / predlab.size
      return metrics.categorical_accuracy(y_true[...,:-1], y_pred)
  else:
    loss = unet.my_categorical_crossentropy()
    met0 = metrics.categorical_accuracy

  net.compile(optimizer=optim, loss={'B':loss}, metrics={'B':met0}) # metrics=['accuracy'])
  return net

def norm_szyxc(img,axs=(1,2,3)):
  mi,ma = img.min(axs,keepdims=True), img.max(axs,keepdims=True)
  img = (img-mi)/(ma-mi)
  return img

def norm_szyxc_per(img,axs=(1,2,3)):
  mi,ma = np.percentile(img,[2,99],axis=axs,keepdims=True)
  img = (img-mi)/(ma-mi)
  img = img.clip(0,1)
  return img

def midplane(arr,i):
  ss = [slice(None) for _ in arr.shape]
  n = arr.shape[i]
  ss[i] = n//2 #slice(n//3, (2*n)//3)
  # return arr[ss].max(i)
  return arr[ss]

def plotlist(lst,i):
  "takes a list of form [arr1, arr2, ...] and "
  lst2 = [norm_szyxc_per(midplane(data,i)) for data in lst]
  lst2[0][...,2] = 0 # remove blue from xs
  res = ts.plotgrid(lst2)
  return res

def show_trainvali(trainable, savepath):
  xsem = trainable['xsem']
  ysem = trainable['ysem']
  xrgb = [xsem['mem'], xsem['nuc'], xsem['nuc']]
  yrgb = [ysem['mem'], ysem['nuc'], ysem['bg']]
  visuals = {'xrgb':xrgb, 'yrgb':yrgb, 'plotlist':plotlist}
  ts.show_trainvali(trainable, visuals, savepath)

def predict_trainvali(net, trainable, savepath):
  xsem = trainable['xsem']
  ysem = trainable['ysem']
  xrgb = [xsem['mem'], xsem['nuc'], xsem['nuc']]
  yrgb = [ysem['mem'], ysem['nuc'], ysem['bg']]
  visuals = {'xrgb':xrgb, 'yrgb':yrgb, 'plotlist':plotlist}
  ts.predict_trainvali(net, trainable, visuals, savepath)

def predict(net,img,xsem,ysem):
  container = np.zeros(img.shape[:-1] + (ysem['n_channels'],))
  
  # borders = [0,4,0,0]
  # patchshape_padded = [1,24,400,400]
  borders = xsem['borders']
  patchshape_padded = list(xsem['patchsize'])
  patchshape_padded[1] = 400
  patchshape_padded[2] = 400
  padding = [(b,b) for b in borders] + [(0,0)]

  patches = patchmaker.patchtool({'img':container.shape[:-1], 'patch':patchshape_padded, 'borders':borders})
  # patches = xsem['res_patches']
  img = np.pad(img, padding, mode='constant')
  ax_zyx = (0,1,2)

  s2 = patches['slice_patch']
  for i in range(len(patches['slices_padded'])):
    s1 = patches['slices_padded'][i]
    s3 = patches['slices_valid'][i]
    x = img[s1]
    x = x / x.mean(ax_zyx, keepdims=True)
    # x = collapse2(x, 'szyxc','s,y,x,zc')
    container[s3] = net.predict(x[np.newaxis])[s2]

  return container

def load_img0_predict_eval(net, trainable, rawdata, homedir):
  img2 = np.load(str(homedir / 'data/img006_noconv.npy'))
  img2 = img2[0]
  img2 = perm(img2, "ZCYX", "ZYXC")
  img2 = norm_szyxc_per(img2,(0,1,2))
  pimg = predict(net, img2, trainable['xsem'], trainable['ysem'])

  if False:
    cen = rawdata['cen']

    x = pimg[...,1]
    # hx = np.array([1,1,1]) / 3
    # x = gputools.convolve_sep3(x, hx, hx, hx)
    hyp = watershed(-x, label(cen)[0], mask=x>0.5)
    print(hyp.max())

    rawdata1 = build_rawdata(homedir)
    gt_patches = dict()
    gt_patches['gt_slices'] = rawdata1['gt_slices'][:-4]
    gt_patches['inds_labeled_slices'] = rawdata1['inds_labeled_slices'][:, :-4]

    seg_scores = ts.compute_seg_on_slices(hyp, gt_patches)
    print(seg_scores)
    print(seg_scores.mean())

  return pimg

def scores(pimg):
  hyp = ts.segment(pimg, ts.segparams())
  hypslices = ts.hyp2hypslices(hyp, rawdata1['inds_labeled_slices'][1], [0]*31)

def optimize_pimg_on_t0(pimg, homedir, mypath_opt):
  segparams = ts.segparams()
  rawdata = build_rawdata(homedir)
  gt_patches = dict()
  gt_patches['gt_slices'] = rawdata['gt_slices'][:-4]
  gt_patches['inds_labeled_slices'] = rawdata['inds_labeled_slices'][:, :-4]
  best = ts.optimize_segmentation(pimg, {**rawdata, **gt_patches}, segparams, mypath_opt)
  hyp = np.array([segparams['function'](x, best) for x in pimg])
  seg_scores = ts.compute_seg_on_slices(hyp, rawdata)
  return seg_scores

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
    res = ts.pad_divisible(res, 1, 5)
    r,c = res.shape[1]//5, 5
    res = splt(res, r, 1)
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

I don't think that the 3D unet is the same as the z-in-channels architecture.
The receptive field from the z-in-channels architecture is only exactly as big as zdim.
But in the real 3D conv model the low level features are very local in space, but later in the network 
the features draw on information from a much wider area.
If we put z-in-channels we get effectively a very wide convolution in the first layer, but then no 
convolution after that because the receptive field doesn't grow at all after the first layer.
It just the same features/channels being remixed and remixed.
However the receptive field does grow in x/y as we go deeper.
The effective receptive field in true spatial dimensions is very well defined in z, but less well
defined in x/y, because it's much easier for the net to nearby info, even though it's possible to use far away info.
The effective receptive field might even depend on the data and training!
It must be measured empirically, just like the max patch size before exceeding GPU memory.

## Mon Aug  6 18:37:16 2018

https://stackoverflow.com/questions/33736795/tensorflow-numpy-like-tensor-indexing
numpy style indexing in tensorflow would make cusom metrics easier to create.

If I dont' add borders there are only 8 patches available in the 3D unet with 40x104x104 size.
This means only one patch for validation.
This is annoying.

## Thu Aug 23 12:15:28 2018

Let's try something new...
Let's throw away all the handmade ground truth that we have.
Instead we're going to bootstrap our way into ground truth using the centerpoint annotations.
These annotations can be the start of a full 3D seeded watershed segmentation.
And we can use that segmentation to train a full 3D unet.

In order for this to be useful the 3D unet must be *better than the watershed* and it must even be 
better than the watershed after some simple pre- and post- processing! (if we compare the 
watershed on img vs watershed on pimg with same seeds.)

but the result after unet looks much smoother, denoised, and nicer.
also, since we have full 3D instances, we could use them to train an end-to-end model like 3D stardist
or 3D GMM.

We can also play with the 3D instances to make them easier to learn or to produce better segmentations.
- try dilating the membrane region. membrane is much more important than nuc or bg.
  this also effectively shrinks the nuc region, which is essentially what 


It appears that having dist-2-mem weights doesn't help to strengthen the intensity of the membrane
*within* and *between* adjacent nuclei, only the membrane between nuclei and bg!!
This is obnoxious. It may be because the membrane between nuclei is hard to get right...

What if we increase the width of the membrane region everywhere?

- test out these predictions against hand-annotated data and compute seg score.
- 


How to test against hand-curated data?
- we need a pimg / hyp covering the entire timeseries
  - it doesn't matter how it was made
  - it doesn't require the classifier
  - it should be a separate module for testing segmentation stuff

idea: dilate membrane, but only in x&y directions! That's where the signal is weak!

idea: do watershed from seed points on the fly while training using random, reasonable mask level!
(also use rotations.)
also augment the input *before* watershed? should be robust to noise, blur, etc.

classweights just make every pixel look a bit more like membrane class.

let's try to regress dist to bg from this data...






"""