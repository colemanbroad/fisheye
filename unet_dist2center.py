from segtools.defaults.ipython import *
from segtools.defaults.training import *

import lib

import gputools
import ipdb
import pandas as pd

from contextlib import redirect_stdout

import train_seg_lib as ts
patch = patchmaker


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
  res['kern'] = kern
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
  xsem['borders'] = borders

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
  yrgb = [ysem['mem'], ysem['nuc'], ysem['bg']]

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

  def mid(arr,i):
    # ss = [slice(None) for _ in arr.shape]
    # ss[i] = arr.shape[i]//2
    # return arr[ss]
    return arr.max(i)


  def doit(i):
    res = plot(mid(xs_train,i), mid(ys_train,i))
    io.imsave(savepath / 'dat_train_{:d}.png'.format(i), res)
    res = plot(mid(xs_vali,i), mid(ys_vali,i))
    io.imsave(savepath / 'dat_vali_{:d}.png'.format(i), res)

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
  xrgb = [xsem['mem'], xsem['nuc'], xsem['nuc']]
  yrgb = [ysem['mem'], ysem['nuc'], ysem['bg']]

  pred_xs_train = net.predict(xs_train, batch_size=1)
  pred_xs_vali = net.predict(xs_vali, batch_size=1)

  def norm(img):
    # img = img / img.mean() / 5
    mi,ma = img.min(), img.max()
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

## Wed Jul 18 11:32:15 2018

See [1]. The simple technique of integrating the signal also works for us for counting cells!
We can estimate cell density this way! The estimates for each timepoint resulting from this technique are:
[106.01638041469874,
 114.6351055695662,
 113.91586985594505,
 114.59327060832476,
 112.37034932491525,
 116.04253640111408,
 114.48877341882898,
 112.42902679021451,
 111.61588675789606,
 108.85128195723728,
 107.38460025291587]

These are much better estimates than what we get from trying to maximize the number of cells that
we get out of a flat threshold segmentation!
Now how do we turn this density estimate into centerpoint detection and, ultimately, segmentation?

IDEA! Let's try the following:
So we want some kind of non-max-suppression, but we don't want a harsh method like placing a forbidden
mask region around each extracted maximum. There is a much more natural way of extraction for this output!
We find the global max in the image, and then we simply subtract the kernel centered at that location!
Ideally this means that even if two cells overlap heavily we will be able to identify the max of one
and then subtract it's kernel, leaving only the kernel of the second!
This way we also have total control over the *number* of maxima which are extracted.
And there are zero free parameters introduced in the detection phase! (local max requires choosing a neighborhood shape for non-max-suppression)
BUT this technique will fail for cells which overlap heavily if the max appears right in the midpoint of 
the two cells. Then the density is split in half. Is there some way we can search for the 
gaussian mixture which best fits the resulting density distribution? This is the best way,
but it is also computationally overly expensive. Even though we know exactly the number of gaussians to pick!

EXTENSION:
Extend the idea of using different classes in your pixelwise classifier to this type of problem.
Use a different pixel value for your cell centerpoint annotations to differentiate between
different cell types! Especially between mitotic and interphase, but potentially also between
normal internal retina cells oriented along the apical/basal tissue axis vs those on the exterior.

Now we've implemented the detection of cells from pimg according to the scheme described above:
find max, subtract kern centered at max, repeat. The centers are well separated, but they don't
seem to agree with the ground truth centroids from the first image... hmmmm.... doing a thresh 
and then finding center of thresh region is more robust against "noise" / fluctuations in the 
peaks of the predicted density images....

## Thu Jul 19 13:20:57 2018

See if this no-param detection technique works.
It does!
How can we determine which centerpoint detection method works best?
We have ground truth centerpoints and we want to compare detection methods.
So far we've been doing visual analysis. It's pretty easy to see when points align.
It's even relatively easy to guess when points have been split or fused!
We can try matching the pointclouds and comparing matching scores?
We can even plot the pointclouds from the ground truth along with *both* detection alternatives.
This shows us that they agree very well almost everywhere. And both make similar mistakes.

More ways of evaluating detection performance:
Plotting pimg as the 3rd color channel in a stackview (3D view?) of img.
This doesn't actually compare detection methods, but allows you to visually see the quality of the 
output.

Can and should we do matching to ground truth?
We want a numerical way of scoring the quality of the detection phase independently of the density phase.
For cell segmentation we had `seg`, even though our pixelwise classes were trained with crossentropy.
There must be standard metrics?
- minimal distance matching
- 1-1 only? or n-1?
- n-1 matching:
  - voronoi tessellation around gt points
  - every point matches to it's closest neighbor in the gt
- n-1 *soft* matching
  - use a kernel centered around gt points (or proposed points) and soft assign based on kernel values
    and normalize across connected-to points
- allow for alignment/translation/rotation/warps before computing matching score?

## Fri Jul 20 09:16:57 2018

I want to see what kinds of cell localization/detection metrics are commonly used.
Does the cell tracking challenge use a metric for evaluating localization accuracy?
Yes they first perform a hard matching between gt and proposed particle centers by minimizing
the total distance between pairs using the Munkres algorithm.
What is the cost of cells that don't match to anything?
If all costs are positive isn't the best solution no matches at all?
Does munkres alg find a min cost maximum bipartite matching?
Is this what we want? Or do we want the matching algorithm to decide how many matches there are?
The particle tracking challenge use dummy tracks s.t. the ground truth points always match to something.
what is the cost of a match to a dummy track? This is a free parameter.
It could just be whatever the maximum real distance is between points...

Easy! Just use the linear sum assignment function from scipy.optimize. This uses the hungarian matching 
(munkres) algorithm to do a 

LIT:
[1] 2018, 67, Microscopy cell counting and detection with fully convolutional regression networks
  - They do 2d regression of gaussian kernels placed at manually annotated centerpoints.
    They show that this type of gt data makes it easy to estimate cell number by integrating!
  - They argue that there is no need to use skip layers because:
    "deep networks that can represent highly semantic information are not necessary; and (ii), based on this,
    we consider only simple architectures (no skip layers)."
  - tried nets with 1.3 and 3.6 million params.
  - bilinear interpolation upsampling
  - They find the networks difficult to train for exactly the same reasons as i do:
  "Moreover, even for non-zero regions, the peak value of a Gaussian with σ = 2is only about 0.07, the networks tend to be very difficult to train."
  - To alleviate this problem, we simply scale the Gaussian-annotated ground truth (Figure 1 (b)) by a factor of 100, forcing the network
    to fit the Gaussian shapes rather than background zeros. After pretraining with patches, we fine-tune the parameters
    with whole images to smooth the estimated density map, since the 100 × 100 image patches sometimes may only contain part
    of a cell on the boundary.
  - They only describe detection as "taking local maxima"... never details...
[2] 2017, 19, Learning non-maximum suppression https://www.youtube.com/watch?v=SnYMimFnKuY
[3] 2015, 54, End-to-end integration of a convolution network, deformable parts model and non-maximum suppression
[4] 2015, 25, You should use regression to detect cells. https://www.youtube.com/watch?v=4FhdkiZ51Js
[5] 2014, 32, Non-Maximum Suppression for Object Detection by Passing Messages between Windows
  - Affinity Propagation Clustering (APC) is used to cluster proposals
  - resulting bounding box / object choice is taken as representative of each group
  - segmentation via thresholding + connected components is also a clustering technique, albeit
    very simple. But it is used in different way. The shape of the cell *is* the shape of the cluster,
    vs the shape / location of the cell is fully described by every element/pixel in the cluster and we need to
    find the best by taking an average or a representative element.
  - any clustering technique will work for this task!
  - 
[6] 2010, 333, Learning to count objects in images. (learn density but avoid detection/localization)
[7] Objective Comparison of Particle Tracking Methods. Uses munkres matching and total distance as point matching metric.



TODO:
- [x] train model succesfully that includes black border in xs
- [ ] train to saturation
- [ ] find sig values that are always stable
- [ ] tracking from centerpoint matching
- [ ] change training data s.t. centerpoints don't clip at boundaries but extend beyond them
      - this is just like the shape completion version of stardist!
- [ ] callback for saving/visualizing patch predictions over epochs
- [x] 3D show_trainvali
- [ ] scatterplot with color value showing peak score


"""

