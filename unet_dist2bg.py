from segtools.defaults.ipython import *
from segtools.defaults.training import *

import lib

import gputools
import ipdb
import pandas as pd

from contextlib import redirect_stdout

import train_seg_lib as ts
patch = patchmaker
from segtools import label_tools
import skimage.morphology as morph
import gputools


def build_rawdata(homedir):
  homedir = Path(homedir)
  ## load data
  # img = imread(str(homedir / 'data/img006.tif'))
  img = np.load(str(homedir / 'data/img006_noconv.npy'))
  img = img[1]
  img = perm(img, "ZCYX", "ZYXC")
  img = norm_szyxc_per(img,(0,1,2))
  img = img.astype(np.float32)
  points = lib.mkpoints()

  imgsem = {'axes':"ZYXC", 'nuc':1, 'mem':0, 'n_channels':2}

  # img = img[:70,::2,::2]

  scale = (2.5,.5,.5)
  def f(idx): return gputools.scale(img[...,idx[0]],scale)
  img = ts.broadcast_over(f,(2,))
  img = perm(img, "czyx","zyxc")
  points = (points * scale).astype(np.int)
  # points[:,0] = points[:,0].clip(0,img.shape[0]-1)
  
  # scale!
  # img = zoom(img, (5,1,1,1))
  if False:
    shape = np.array(img.shape)
    res = patchmaker.patchtool({'img':shape, 'patch':shape//2, 'grid':(2,2,2,2)})
    scaled = [gputools.scale((img[ss][...,0]).astype(np.float32), (5,1,1)) for ss in res['slices']]
    ends = res['starts'] + shape//2 * [5,1,1,1]

    container = np.zeros(shape * [5,1,1,1])
    big_patches = patchmaker.starts_ends_to_slices(res['starts'], ends)
    for i in range(len(big_patches)):
      container[big_patches[i]] = scaled[i][...,np.newaxis]

  container = img.copy()

  cen = np.zeros(container.shape[:-1])
  cen[list(points.T)] = 1
  x = container[...,1]
  hx = np.array([1,1,1]) / 3
  x = gputools.convolve_sep3(x, hx, hx, hx, sub_blocks=(1,1,1))
  lab = watershed(-x, label(cen)[0], mask=x>x.mean())
  
  # bor = np.array([morph.binary_dilation(b) for b in bor]) ## just x,y
  # bor = np.array([morph.binary_dilation(b) for b in bor]) ## just x,y

  if False:
    target = lab[::2,::2,::2]
    points = points//2
    bg = target==0
    fg = target==1
    target[fg] = 0.0

    hx = np.array([1,1,1]) / 3
    for i in range(200):
      target[bg] = 0
      target[list(points.T)] = 1
      target = gputools.convolve_sep3(target, hx, hx, hx, sub_blocks=(2,2,2))

    points = points * 2
    target = gputools.scale(target, (2,2,2))

  if False:
    xx = np.linspace(-2,2,80)
    xx = np.exp(-xx**2/2)
    xx = xx / xx.sum()
    gauss = gputools.convolve_sep3(cen, xx, xx, xx, sub_blocks=(2,2,2))

  if True:
    target = lab2bgdist(lab)
    if False:
      target = target * gauss
      target = np.exp(-target/10)


  res = dict()
  res['source'] = container
  res['imgsem'] = imgsem
  res['target'] = target
  res['points'] = points
  res['cen'] = cen
  return res


def lab2bgdist(lab):
  distimg = lab.copy()
  distimg[lab!=0] = 1
  bor = label_tools.find_boundaries(lab)
  distimg[bor] = 0 ## mem is also bg!
  distimg = distance_transform_edt(distimg)
  hx = np.array([1,1,1]) / 3
  distimg = gputools.convolve_sep3(distimg, hx, hx, hx, sub_blocks=(1,1,1))
  distimg = gputools.convolve_sep3(distimg, hx, hx, hx, sub_blocks=(1,1,1))
  distimg[lab==0] = 0
  distimg[bor] = 0
  for i in range(1,lab.max()):
    m = lab==i
    distimg[m] = distimg[m] / max(distimg[m].max(),1)
  return distimg

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

def simple_weights(rawdata):
  target = rawdata['target']
  weight_stack = np.ones(target.shape)
  return weight_stack

def build_trainable(rawdata):
  img = rawdata['source']
  imgsem = rawdata['imgsem']
  target = rawdata['target']

  xsem = {'n_channels':imgsem['n_channels'], 'mem':0, 'nuc':1, 'shape':(None, None, None, imgsem['n_channels']), 'zyx':(1,2,3)}
  ysem = {'n_channels':1, 'gauss':0, 'rgb':[0,0,0], 'shape':(None, None, None, 1)}

  weight_stack = simple_weights(rawdata)

  ## add extra cell center channel
  patchshape = 8*(np.array([1,2.5,2.5])*5).astype(np.int)
  patchshape = 8*(np.array([1,1,1])*8).astype(np.int)
  borders = (0,0,0)
  res = patch.patchtool({'img':target.shape, 'patch':patchshape, 'borders':borders}) #'overlap_factor':(2,1,1)})
  slices = res['slices_padded']
  xsem['patchshape'] = patchshape
  xsem['borders'] = borders

  ## pad images
  cat = np.concatenate
  padding = np.array([borders, borders]).T
  img = np.pad(img, cat([padding, [[0,0]] ], 0), mode='constant')
  target  = np.pad(target, padding, mode='constant')
  weight_stack = np.pad(weight_stack, padding, mode='constant')

  ## extract slices
  xs = np.array([img[ss] for ss in slices])
  ys = np.array([target[ss] for ss in slices])
  ws = np.array([weight_stack[ss] for ss in slices])
  ## add channels to target
  ys = ys[...,np.newaxis]

  ## normalize over space. sample and channel independent
  xs = xs/np.mean(xs,(1,2,3), keepdims=True)
  # xs = norm_szyxc_per(xs)
  # ys = norm_szyxc_per(ys)
  # mask = np.isnan(xs)
  # xs[mask] = 0
  # mask = np.isnan(ys)
  # ys[mask] = 0

  # ys = ys/np.mean(ys,(1,2,3), keepdims=True)
  
  print(xs.shape, ys.shape, ws.shape)

  res = ts.shuffle_split({'xs':xs,'ys':ys,'ws':ws,'slices':slices})
  res['xsem'] = xsem
  res['ysem'] = ysem
  res['slices'] = slices
  return res

def build_net(xsem, ysem):
  unet_params = {
    'n_pool' : 2,
    'n_convolutions_first_layer' : 32,
    'dropout_fraction' : 0.2,
    'kern_width' : 3,
  }

  mul = 2**unet_params['n_pool']
  faclist = [factors(x) for x in xsem['patchshape'][1:-1]]
  for fac in faclist: assert mul in fac

  input0 = Input(xsem['shape'])
  unet_out = unet.get_unet_n_pool(input0, **unet_params)
  output2  = unet.acti(unet_out, ysem['n_channels'], last_activation='linear', name='B')

  net = Model(inputs=input0, outputs=output2)

  optim = Adam(lr=2e-5)
  # loss  = unet.my_categorical_crossentropy(classweights=classweights, itd=0)
  # loss = unet.weighted_categorical_crossentropy(classweights=classweights, itd=0)
  # ys_train = np.concatenate([ys_train, ws_train[...,np.newaxis]], -1)
  # ys_vali  = np.concatenate([ys_vali, ws_vali[...,np.newaxis]], -1)
  def met0(y_true, y_pred):
    # mi,ma = np.percentile(y_pred,[2,98])
    # return ma-mi
    return K.std(y_pred)
  
  def loss(y_true, y_pred):
    return losses.mean_squared_error(y_true,y_pred) #+ 10.0 * (K.mean(y_true) - K.mean(y_pred))**2

  net.compile(optimizer=optim, loss={'B':loss}, metrics={'B':met0})
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
  return arr[ss] #.max(i)

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
  yrgb = [ysem['gauss'], ysem['gauss'], ysem['gauss']]
  visuals = {'xrgb':xrgb, 'yrgb':yrgb, 'plotlist':plotlist}
  ts.show_trainvali(trainable, visuals, savepath)

def predict_trainvali(net, trainable, savepath):
  xsem = trainable['xsem']
  ysem = trainable['ysem']
  xrgb = [xsem['mem'], xsem['nuc'], xsem['nuc']]
  yrgb = [ysem['gauss'], ysem['gauss'], ysem['gauss']]
  visuals = {'xrgb':xrgb, 'yrgb':yrgb, 'plotlist':plotlist}
  ts.predict_trainvali(net, trainable, visuals, savepath)

def load_and_predict_t0(net, trainable, homedir):
  homedir = Path(homedir)
  ## load data
  # img = imread(str(homedir / 'data/img006.tif'))
  img = np.load(str(homedir / 'data/img006_noconv.npy'))
  img = img[0]
  img = perm(img, "ZCYX", "ZYXC")
  img = norm_szyxc_per(img,(0,1,2))
  img = img.astype(np.float32)

  scale = (2.5,.5,.5)
  def f(idx): return gputools.scale(img[...,idx[0]],scale)
  img = ts.broadcast_over(f,(2,))
  img = perm(img, "czyx","zyxc")
  return img

  # pimg = predict(net, img, trainable['xsem'], trainable['ysem'])
  # return pimg

def watershedme(pimg):
  # for t1 in np.linspace(pimg.min(),pimg.max(),10):
  #   for t2 in np.linspace(pimg.min(),pimg.max(),10):
  #     lab = watershed(-pimg, label(pimg>0.25)[0], mask = pimg > 0.1)
  # for p1 in [30,40,50,60,70,80]:
  for p2 in [82,85,90,95,98,99,99.5,99.9,99.99]:
    t_low,t_hi = np.percentile(pimg,[50,p2])
    lab = watershed(-pimg, label(pimg>t_hi)[0], mask = pimg > t_low)
    print(lab.max())

def testseg(pimg,img,homedir):
  import unet_3d_pixclass as u3
  rawdata1 = u3.build_rawdata(homedir)
  for p1 in [50,55,60,65]:
    t_low = np.percentile(img,p1)
    t_hi  = np.percentile(pimg,90)
    hyp = watershed(-img, label(pimg>t_hi)[0], mask = img > t_low)
    zsli = rawdata1['inds_labeled_slices'][1]
    zsli = (zsli*2.5).astype(np.int)
    hypslices = ts.hyp2hypslices(hyp, zsli, [0]*31)
    gts = rawdata1['gt_slices']
    gts = gts[:,::2,::2]
    segscores = ts.scores(hypslices, gts)
    print(segscores)
    print(segscores[:20].mean())

def predict(net, img):
  "img must have axes: TZYXC and xyz voxels of (.2,.2,.5)um"
  n_channels = (1,)
  container = np.zeros(img.shape[:-1] + n_channels)
  
  borders = (40,40,40)
  patchshape = (4*40,4*40,4*40)
  # assert np.all([4 in factors(n) for n in patchshape[1:]]) ## unet shape requirements
  res = patch.patchtool({'img':img.shape[:-1], 'patch':patchshape, 'borders':borders})
  padding = [(b,b) for b in borders] + [(0,0)]
  img = np.pad(img, padding, mode='constant')
  s2  = res['slice_patch']

  for i in range(len(res['slices_valid'])):
    s1 = res['slices_padded'][i]
    s3 = res['slices_valid'][i]
    x = img[s1]
    x = x / x.mean((0,1,2))
    x = net.predict(x[np.newaxis])
    container[s3] = x[0][s2]

  return container

def scale_whole_stack(homedir):
  homedir = Path(homedir)
  ## load data
  # img = imread(str(homedir / 'data/img006.tif'))
  img = np.load(str(homedir / 'data/img006_noconv.npy'))
  # img = img[0]
  img = perm(img, "TZCYX", "TZYXC")
  img = norm_szyxc_per(img,(1,2,3))
  img = img.astype(np.float32)

  scale = (2.5,.5,.5)
  def f(idx): return gputools.scale(img[idx[0],...,idx[1]],scale)
  img = ts.broadcast_over(f,(11,2))
  img = perm(img, "tczyx","tzyxc")
  return img

def predict_whole_stack(net, homedir, savedir):
  imgs = scale_whole_stack(homedir)
  pimgs = np.array([predict(net, img) for img in imgs])
  np.save(savedir / 'pimgs', pimgs)
  return pimgs

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

Let's try regressing the dist to bg from bootstrapped GT data...

Prediction with 2x down model and 32 features works with 40^3 patches but not 50^3.

The results look pretty decent, but tend to fuse objects together. But they tend *not* to oversegment!
Now we just have to figure out a way of adjusting the target to maximize watershed segmentability...

Minor grid artifacts are still visible with border width = 50!!
The means the useful receptive field is very large!

After normalizing per instance and retraining we have a nice result.
The key to making watershed work nicely is setting the max of the distance map for each instance to 1.
Then we don't do any further normalization of the ys.
The xs we normalize by the mean only.

Experiments we must do:
- we have to compare the learned watershed potential with the original fluorescence image + blur / other preprocessing.
- 



"""



