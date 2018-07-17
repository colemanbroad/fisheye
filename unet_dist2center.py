from segtools.defaults.ipython_remote import *
from segtools.defaults.training import *

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

def build_rawdata(homedir):
  img = np.load(str(homedir / 'data/img006_noconv.npy'))
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
  kern = kern[::4] ## anisotropic kernel matches img
  
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
  res['cellcenters'] = cen2[:,::r,::r]
  return res

def compute_weights(rawdata):
  img = rawdata['img']
  weight_stack = np.ones(img.shape[1:-1])
  return weight_stack

def build_trainable(rawdata):
  img = rawdata['img']
  imgsem = rawdata['imgsem']
  cellcenters = rawdata['cellcenters']
  imgdat = img[1]

  xsem = {'n_channels':imgsem['n_channels'], 'mem':0, 'nuc':1, 'shape':(None, None, None, imgsem['n_channels'])}
  ysem = {'n_channels':1, 'gauss':0, 'rgb':[0,0,0], 'shape':(None, None, None, 1)}

  weight_stack = compute_weights(rawdata)

  ## add extra cell center channel
  patchsize = (52,100,100)
  borders = (10,10,10)
  res = patch.patchtool({'sh_img':cellcenters.shape, 'sh_patch':patchsize, 'sh_borders':borders}) #'overlap_factor':(2,1,1)})
  slices = res['slices_padded']
  xsem['patchsize'] = patchsize

  ## pad images
  cat = np.concatenate
  padding = np.array([borders, borders]).T
  imgdat = np.pad(imgdat, cat([padding, [[0,0]] ], 0), mode='constant')
  cellcenters  = np.pad(cellcenters, padding, mode='constant')
  weight_stack = np.pad(weight_stack, padding, mode='constant')

  ## extract slices
  xs = np.array([imgdat[ss] for ss in slices])
  ys = np.array([cellcenters[ss] for ss in slices])
  ws = np.array([weight_stack[ss] for ss in slices])
  ## add channels to target
  ys = ys[...,np.newaxis]

  ## normalize over space. sample and channel independent
  xs = xs/np.mean(xs,(1,2,3), keepdims=True)
  
  print(xs.shape, ys.shape, ws.shape)

  res = ts.shuffle_split({'xs':xs,'ys':ys,'ws':ws})
  res['xsem'] = xsem
  res['ysem'] = ysem
  res['slices'] = slices
  return res

def build_net(xsem, ysem):
  unet_params = {
    'n_pool' : 2,
    'n_convolutions_first_layer' : 16,
    'dropout_fraction' : 0.2,
    'kern_width' : 5,
  }

  input0 = Input(xsem['shape'])
  unet_out = unet.get_unet_n_pool(input0, **unet_params)
  output2  = unet.acti(unet_out, ysem['n_channels'], last_activation='linear', name='B')

  net = Model(inputs=input0, outputs=output2)

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
  yrgb = [ysem['gauss']]*3

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
    res = collapse2(res, 'isyxc','sy,ix,c')
    return res

  res = plot(xs_train.max(1), ys_train.max(1))
  io.imsave(savepath / 'dat_train_z.png', res)
  res = plot(xs_vali.max(1), ys_vali.max(1))
  io.imsave(savepath / 'dat_vali_z.png', res)

  res = plot(xs_train.max(2), ys_train.max(2))
  io.imsave(savepath / 'dat_train_y.png', res)
  res = plot(xs_vali.max(2), ys_vali.max(2))
  io.imsave(savepath / 'dat_vali_y.png', res)

  res = plot(xs_train.max(3), ys_train.max(3))
  io.imsave(savepath / 'dat_train_x.png', res)
  res = plot(xs_vali.max(3), ys_vali.max(3))
  io.imsave(savepath / 'dat_vali_x.png', res)

  if False:
    wz = xsem['patchsize'][0]
    middle_z = slice(2*wz//5,3*wz//5)
    xs_train = xs_train[:,middle_z].max(1)
    xs_vali  = xs_vali[:,middle_z].max(1)
    ys_train = ys_train[:,middle_z].max(1)
    ys_vali  = ys_vali[:,middle_z].max(1)

def predict_trainvali(net, trainable, savepath=None):
  xs_train = trainable['xs_train']
  xs_vali = trainable['xs_vali']
  ys_train = trainable['ys_train']
  ys_vali = trainable['ys_vali']
  xsem = trainable['xsem']
  ysem = trainable['ysem']

  rgb_ys = [ysem['gauss']]*3 
  rgb_xs = [xsem['mem'], xsem['nuc'],xsem['nuc']]
  
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


history = """

## Thu Jul 12 12:11:46 2018

We can predict *something* for the cell centerpoint channel, but it's pretty blurry.
We want to sharpen it up.
Let's see how small we can make the kernel while still being able to learn.

## Thu Jul 12 18:01:50 2018

Moved into it's own file!
There is an interesting instability in the training that only gets fixed if I increase the
size of my blobs! If the cell centerpoint blobs are too small then the network just predicts 
everything with a constant low value.

*Maybe I can slowly reduce the width of the blob during training?*

See `res066` for the results of "successful" training with a reasonable blob size.
The model is incapable of learning cell centers. It just takes does the very conservative
thing and guesses roughly everywhere that nuclear intensity can be found. This is only with 10 epochs
training.

Should I abandon this idea and now try to use the cell centers to do a seeded watershed for
semi-gt training data?

First, let's continue training for more epochs and see if we get an improvement.

Yes, we *do* see an improvement. After 40 epochs the val loss is till going down slightly.
val_loss: 0.0625

- We may want to do some smarter non-maximum suppression for finding seeds.
- We may want to try training on smaller, downscaled images.

Yes! With smaller size we get to 0.055 loss already by the 18th epoch. 
Look at `res067`! These are predictions on the *validation data*. With a loss of 0.0390.

NOTE: you weren't using the entire training data previously! you only had 2 patches of width 128 in a 400 width image!

Now to actually detect cell centers we want to identify the peaks in this signal.
We can do this in several ways:
- Local max filter of proper size. This effectively does local non max suppression.
- apply low threshold then try to fix the "undersegmentations"
- fit circles / spheres ?

## Fri Jul 13 10:45:47 2018

Now let's try something sneaky. Let's reduce the width of the blobs marking centerpoints and continue training.
We change sigma to 10 from 15 and continue training the previous model on 2x xy downsampled data.
The val loss goes all the way down to 0.0133!
The centerpoints are small and reasonable, although the heights of the peaks seem to vary too much.
One issue might be the way we make the ground truth data!
We should not be *convolving* with the gaussian kernel, beause then the height of nearby peaks will grow.
We should just be *placing* the shapes in the image.
But when two kernels overlap, ideally, the result should not choose one over another but squish them both together...

PROBLEM:
We've been using isotropic kernels to mark the cell center. But we're not using isotropic images!
We should rescale z s.t. the image is isotropic!

Actually the easiest thing to do is just make the kernels anisotropic in the same way as the images.
Downsample them by 5x.

The corrected, anisotropic target fails to train. Predicts const values.
Let's try the anisotropic kernel, but do use convolutions to make the training data instead of just assignment.
(still with 2x down in xy)

I guess making the kernel aniotropic like the image reduced the total intensity significantly, making it hard to learn.

No it appears that the smaller kernel shapes are not stable in training for convolutions either.

Let's increase the size of sigma but keep the kernel anisotropic...

This works! With sigma=15 we train successfully on the first try. Down to 0.0568 after 10 epochs. 
0.0340 after 23 epochs. 0.0205 after 50. The results look excellent. A peak of 95 cell centers are
identified. The scatterplots show that most of the missing points come at the z-extremes.

Let's try placing the kernels instead of convolving them, but with this larger kernel size.
- try training from scratch
- try building off of the previous model

## Mon Jul 16 11:46:12 2018

The cell centers don't do a good job of identifying cells at the image boundaries.
Before tryin to shrink the centerpoint size we want to retrain exhaustively with training data
from a zero-padded image.
see `centerpoint_predictions_analysis.py` for seg analysis.

Now it's having trouble learning even with sig=15, conv placement, large patches, etc...

## Tue Jul 17 14:54:54 2018

The network has no trouble learning even small kernels and high downsampling ratios, but you've
got to coax the net into it! We train for 10 or 20 epochs, then downsize kernel, then retrain.
Do this with small sig = 10, but NO downsampling, then progress until downsampling==4. The
resulting model predicts 108 cells for t==2 and 102 cells for t==1. This is good.

Also added a preview of trainvali that works for x,y and z views.


TODO:
- [ ] train model succesfully that includes black border in xs
- [ ] train to saturation
- [ ] find sig values that is always stable
- [ ] tracking from centerpoint matching
- [ ] change training data s.t. centerpoints don't clip at boundaries but extend beyond them
      - this is just like the shape completion version of stardist!
- [ ] callback for saving/visualizing patch predictions over epochs
- [ ] 3D show_trainvali
- [ ] 


"""
