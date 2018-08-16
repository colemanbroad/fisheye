from segtools.defaults.ipython import *
from segtools.defaults.training import *

import lib

import gputools
import ipdb
import pandas as pd

from contextlib import redirect_stdout

import train_seg_lib as ts
patch = patchmaker
import czifile

def build_rawdata(homedir):
  imgname = '/net/fileserver-nfs/stornext/snfs2/projects/myersspimdata/Mauricio/for_coleman/ph3_labels_trainingdata_07_10_2018/trainingdata_ph3labels_hspph2bandbactinlap2bline_fish6_justph3andh2b.czi'
  img = czifile.imread(imgname)
  img = img[0,0,0,0,0,:,0,:,:,:,0] # CZYX
  img = perm(img, "CZYX","ZYXC")
  sh = np.array(img.shape)[:-1] // 8
  ss = patchmaker.se2slices(sh,7*sh)
  img = img[ss]
  img = img / img.mean(axis=(0,1,2), keepdims=True)

  sig = 5
  wid = 16
  def f(x): return np.exp(-(x*x).sum()/(2*sig**2))
  kern = math_utils.build_kernel_nd(wid,3,f)
  kern = kern[::4] ## anisotropic kernel matches img
  kern = kern / kern.sum()

  # cen[list(points.T)] = 1
  img[...,0] = fftconvolve(img[...,0], kern, mode='same')
  # img = img / img.mean(axis=(0,1,2), keepdims=True)

  r = 1 ## xy downsampling factor
  imgsem = {'axes':"ZYXC", 'ph3':0, 'h2b':1, 'n_channels':2, 'r':r} ## image semantics

  res = dict()
  res['img'] = img[:,::r,::r]
  res['imgsem'] = imgsem
  return res

def compute_weights(rawdata):
  img = rawdata['img']
  weight_stack = np.ones(img.shape[:-1])
  return weight_stack

def build_trainable(rawdata):
  img = rawdata['img']
  imgsem = rawdata['imgsem']

  xsem = {'n_channels':1, 'h2b':0, 'shape':(None, None, None, 1)}
  ysem = {'n_channels':1, 'h2b':None, 'ph3':0, 'shape':(None, None, None, 1)}

  weight_stack = compute_weights(rawdata)

  ## add extra cell center channel
  patchsize = (16,240,240)
  borders = (0,0,0)
  res = patch.patchtool({'img':img.shape[:-1], 'patch':patchsize, 'borders':borders}) #'overlap_factor':(2,1,1)})
  slices = res['slices_padded']
  # slices = slices[::6]
  xsem['patchsize'] = patchsize
  xsem['borders'] = borders

  ## pad images
  if False:
    cat = np.concatenate
    padding = np.array([borders, borders]).T
    img = np.pad(img, cat([padding, [[0,0]] ], 0), mode='constant')
    weight_stack = np.pad(weight_stack, padding, mode='constant')

  ## extract slices
  xs = np.array([img[ss][...,[1]] for ss in slices])
  ys = np.array([img[ss][...,[0]] for ss in slices])
  ws = np.array([weight_stack[ss] for ss in slices])
  inds = np.argsort(ys.sum((1,2,3,4)))
  res['inds'] = inds
  xs = xs[inds[::-1][:9]]
  ys = ys[inds[::-1][:9]]
  ws = ws[inds[::-1][:9]]
  # ipdb.set_trace()
  # ys = (ys > np.percentile(ys,90.,axis=(1,2,3,4)).reshape((-1,1,1,1,1))).astype(np.int)
  # ys = np_utils.to_categorical(ys).reshape(ys.shape[:-1] + (-1,))

  ## normalize over space. sample and channel independent
  xs = xs/np.mean(xs,(1,2,3), keepdims=True)
  ys = ys/np.mean(ys,(1,2,3), keepdims=True)
  
  print(xs.shape, ys.shape, ws.shape)

  # res = ts.shuffle_split({'xs':xs,'ys':ys,'ws':ws})
  res = ts.copy_split({'xs':xs,'ys':ys,'ws':ws})
  res['xsem'] = xsem
  res['ysem'] = ysem
  res['slices'] = slices
  return res

def build_net(xsem, ysem):
  unet_params = {
    'n_pool' : 3,
    'n_convolutions_first_layer' : 32,
    'dropout_fraction' : 0.2,
    'kern_width' : 3,
  }

  mul = 2**unet_params['n_pool']
  faclist = [factors(x) for x in xsem['patchsize']]
  for fac in faclist: assert mul in fac

  input0 = Input(xsem['shape'])
  unet_out = unet.get_unet_n_pool(input0, **unet_params)
  output2  = unet.acti(unet_out, ysem['n_channels'], last_activation='linear', name='B')

  net = Model(inputs=input0, outputs=output2)

  optim = Adam(lr=2e-4)
  # loss  = unet.my_categorical_crossentropy(classweights=classweights, itd=0)
  # loss = unet.weighted_categorical_crossentropy(classweights=classweights, itd=0)
  # ys_train = np.concatenate([ys_train, ws_train[...,np.newaxis]], -1)
  # ys_vali  = np.concatenate([ys_vali, ws_vali[...,np.newaxis]], -1)
  def met0(y_true, y_pred):
    # mi,ma = np.percentile(y_pred,[2,98])
    # return ma-mi
    return K.std(y_pred)
  
  loss = losses.mean_absolute_error
  cw = [0.2,5]
  loss = unet.my_categorical_crossentropy(classweights=cw)
  loss = losses.mean_squared_error

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
  xrgb = [xsem['h2b'], xsem['h2b'], xsem['h2b']]
  yrgb = [ysem['ph3'], ysem['ph3'], ysem['ph3']]

  def norm(img):
    # img = img / img.mean() / 5
    mi,ma = img.min(), img.max()
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
    n = arr.shape[i]
    ss[i] = slice(n//3,(2*n)//3)
    # return arr[ss]
    return arr[ss].max(i)

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

def predict_trainvali(net, trainable, savepath=None):
  xs_train = trainable['xs_train']
  xs_vali  = trainable['xs_vali']
  ys_train = trainable['ys_train']
  ys_vali  = trainable['ys_vali']
  ws_train = trainable['ws_train']
  ws_vali  = trainable['ws_vali']
  xsem = trainable['xsem']
  ysem = trainable['ysem']
  xrgb = [xsem['h2b']]*3
  yrgb = [ysem['ph3']]*3

  # rgb_xs = [xsem['h2b']]*3
  # rgb_ys = [ysem['ph3']]*3

  pred_xs_train = net.predict(xs_train, batch_size=1)
  pred_xs_vali = net.predict(xs_vali, batch_size=1)

  def norm(img):
    # img = img / img.mean() / 5
    mi,ma = img.min(), img.max()
    # mi,ma = np.percentile(img, [5,95])
    img = (img-mi)/(ma-mi)
    img = np.clip(img, 0, 1)
    return img

  def plot(xs, ys, ps):
    xs = norm(xs[...,xrgb])
    ys = norm(ys[...,yrgb])
    ps = norm(ps[...,yrgb]) # predictions
    xs[...,2] = 0
    res = np.stack([xs,ys,ps],0)
    n_patches = res.shape[1]
    r, c = max(1, n_patches//5), 5
    res = splt(res[:,:r*c], r, 1)
    res = collapse2(res, 'iRCyxc','Ry,Cix,c')
    return res

  def mid(arr,i):
    # ss = [slice(None) for _ in arr.shape]
    # ss[i] = arr.shape[i]//2
    # return arr[ss]
    return arr.max(i)

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

def predict_trainvali_1(net, trainable, savepath=None):
  xs_train = trainable['xs_train']
  xs_vali = trainable['xs_vali']
  ys_train = trainable['ys_train']
  ys_vali = trainable['ys_vali']
  xsem = trainable['xsem']
  ysem = trainable['ysem']

  rgb_xs = [xsem['h2b']]*3
  rgb_ys = [ysem['ph3']]*3
  
  pred_xs_train = net.predict(xs_train, batch_size=1)
  pred_xs_vali = net.predict(xs_vali, batch_size=1)

  xs_train = xs_train.max(1)
  xs_vali = xs_vali.max(1)
  pred_xs_train = pred_xs_train.max(1)
  pred_xs_vali = pred_xs_vali.max(1)
  ys_train = ys_train.max(1)
  ys_vali = ys_vali.max(1)

  def norm(img):
    # img = img / img.mean() / 5
    mi,ma = img.min(), img.max()
    # mi,ma = np.percentile(img, [5,95])
    img = (img-mi)/(ma-mi)
    img = np.clip(img, 0, 1)
    return img

  def plot(xs,pred,ys,rows,cols):
    x = xs[:rows*cols,...,rgb_xs]
    y = pred[:rows*cols,...,rgb_ys]
    z = ys[:rows*cols,...,rgb_ys]
    x = norm(x)
    y = norm(y)
    z = norm(z)
    xy = np.stack([x,z,y],0)
    res = collapse2(splt(xy,cols,1), 'iCRyxc', 'Ry,Cix,c')
    return res

  n = xs_train.shape[0]
  cols = min(8, n)
  rows = floor(n/cols)
  res1 = plot(xs_train, pred_xs_train, ys_train, rows, cols)

  n = xs_vali.shape[0]
  cols = min(8, n)
  rows = floor(n/cols)
  res2 = plot(xs_vali, pred_xs_vali, ys_vali, rows, cols)

  if savepath: io.imsave(savepath / 'pred_xs_train.png', norm(res1))
  if savepath: io.imsave(savepath / 'pred_xs_vali.png', norm(res2))

  return res1, res2

def predict(net, img, xsem, ysem):
  container = np.zeros(img.shape[:-1] + (ysem['n_channels'],))
  cat = np.concatenate

  
  borders = np.array((0,20,20,20))
  patchshape = np.array([1,64,200,200]) + 2*borders
  assert np.all([4 in factors(n) for n in patchshape[1:]]) ## unet shape requirements
  res = patch.patchtool({'sh_img':img.shape[:-1], 'sh_patch':patchshape, 'sh_borders':borders})
  padding = np.array([borders, borders]).T
  img = np.pad(img, cat([padding,[[0,0]]],0), mode='constant')
  s2  = res['slice_patch']

  for i in range(len(res['slices_valid'])):
    s1 = res['slices_padded'][i]
    s3 = res['slices_valid'][i]
    x = img[s1]
    x = x / x.mean((1,2,3))
    container[s3] = net.predict(x)[s2]

  return container

def detect(pimg, rawdata, n=None):
  r = rawdata['imgsem']['r']
  kern = rawdata['kern'][:,::r,::r]

  def n_cells(pimg): return pimg.sum() / kern.sum()
  
  pimgcopy = pimg.copy()

  borders = np.array(kern.shape)
  padding = np.array([borders, borders]).T
  pimgcopy = np.pad(pimgcopy, padding, mode='constant')

  if n is None: n = ceil(n_cells(pimgcopy))

  centroids = []
  n_remaining = []
  peaks = []

  for i in range(n):
    nc = n_cells(pimgcopy)
    n_remaining.append(nc)
    ma = pimgcopy.max()
    peaks.append(ma)
    centroid = np.argwhere(pimgcopy == ma)[0]
    centroids.append(centroid)
    start = centroid - np.ceil(borders/2).astype(np.int)  #+ borders
    end   = centroid + np.floor(borders/2).astype(np.int) #+ borders
    ss = patchmaker.se2slices(start, end)
    print(ss, pimgcopy[centroid[0], centroid[1], centroid[2]])
    print(patchmaker.shape_from_slice(ss))
    pimgcopy[ss] -= kern
    pimgcopy = pimgcopy.clip(min=0)

  centroids = np.array(centroids) - borders[np.newaxis,:]
  centroids[:,[1,2]] *= r

  res = dict()
  res['n_cells'] = n_cells
  res['centroids'] = centroids
  res['remaining'] = n_remaining
  res['peaks'] = peaks
  return res

def detect2(pimg, rawdata):
  ## estimate number of cell centerpoints
  ## TODO: introduce non-max suppression?
  r = rawdata['imgsem']['r']
  kern = rawdata['kern']
  mi,ma = 0.2*kern.max(), 1.5*kern.max()
  thresholds = np.linspace(mi,ma,100)
  n_cells = [label(pimg>i)[1] for i in thresholds]
  n_cells = np.array(n_cells)
  delta = n_cells[1:]-n_cells[:-1]
  thresh_maxcells = np.argmax(n_cells)
  yneg = np.where(delta<0,delta,0)
  n_fused = -yneg[:thresh_maxcells].sum()
  estimated_number_of_cells = n_fused + n_cells[thresh_maxcells]
  optthresh = thresholds[thresh_maxcells]

  ## plot centroid for each cell and compare to gt (time zero only)
  seg = label(pimg>optthresh)[0]
  nhl = nhl_tools.hyp2nhl(seg)
  centroids = [a['centroid'] for a in nhl]
  centroids = np.array(centroids)

  centroids[:,[1,2]] *= r

  res = dict()
  res['thresh'] = optthresh
  res['centroids'] = centroids
  res['n_cells'] = n_cells[thresh_maxcells]
  res['n_fused'] = n_fused
  return res




history = """

Tue Aug 14 16:22:15 2018

Let's adadpt the dist2cellcenter model so it can regress our ph3 labels!

Wed Aug 15 15:23:08 2018

The regression doesn't work. The task is too hard. The loss after 40 epochs is 0.2100 and the 
net only learns essentially the identity function.
Now I'll try to make the task easier by applying a threshold to the target, essentially
turning it into a classification problem.
Regression on the thresholded target with mean_abs_err loss also doesn't work. Just produces flat grey predictions.
Now I'm changing the threshold to 99.5%% and changing the last layer + loss to softmax and cross entropy with classweights.
Maybe the classweights will help prevent it from getting stuck.

Thu Aug 16 13:46:13 2018

I've reduced the patch size to just a few cell widths (240px), sorted the data by content and
kept only the 100 3D patches with the most ph3 fluorescence. Hopefully this will convince the net
to learn something!

Tried the new training set with:
sqaured error regression
abs error regression
crossentropy classification of 99.5% bin mask

going to try blurring the ph3 channel on input.
Don't try to predict the fine structure! Just roughly localize. Well. Predicting a blurred image is better than localization/detection.

It appears that the 3-layer net w 32 channels is able to learn a very rough single patch after 120
epochs. But with three patches it already becomes difficult...

Why does my vali loss stay high as my train loss goes down?
Vali also goes down, but much more slowly.
It's possible to overfit on many patches.
If the overfitting does't look very good then the model is too constrained.
How many params? 1.5 mil. how many pixels? 240x240x16x9. 8 mil.
OK, now the overfitting looks *very* good. The 3down 32chan 3px kern net is able to memorize 9 patches.
see `w020.h5` in ph3_test for the model weights.



"""

