from segtools.defaults.ipython import *
from segtools.defaults.training import *
import keras
from segtools import label_tools
from segtools import graphmatch
from segtools import render

import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import lib

import gputools
import ipdb
import pandas as pd

from contextlib import redirect_stdout
import train_seg_lib as ts
from sklearn.metrics import confusion_matrix

cat = np.concatenate
import scipy.ndimage.morphology as morph
from sklearn.mixture import GaussianMixture

from pykdtree.kdtree import KDTree as pyKDTree

from skimage.morphology import disk         
from skimage.filters import threshold_otsu, rank

import segmentation

## testing segmentations against GT

## load challenge GT data

seg1 = "/net/fileserver-nfs/stornext/snfs2/projects/myersspimdata/Coleman/Celegans/ISBI/Fluo-N3DH-CE_challenge/01/SEG"
seg2 = "/net/fileserver-nfs/stornext/snfs2/projects/myersspimdata/Coleman/Celegans/ISBI/Fluo-N3DH-CE_challenge/02/SEG"

datasets = dict()
datasets['t1'] = {'base':'Fluo-N3DH-CE/',          'dind':1,'dset':'train1','n':250,'z':192,'sh':(192,256,354),'osh':(35, 512, 708),'path':Path('Fluo-N3DH-CE/01/')}
datasets['t2'] = {'base':'Fluo-N3DH-CE/',          'dind':2,'dset':'train2','n':250,'z':170,'sh':(170,256,356),'osh':(31, 512, 712),'path':Path('Fluo-N3DH-CE/02/')}
datasets['c1'] = {'base':'Fluo-N3DH-CE_challenge/','dind':1,'dset':'chall1','n':190,'z':170,'sh':(170,256,356),'osh':(31, 512, 712),'path':Path('Fluo-N3DH-CE_challenge/01/')}
datasets['c2'] = {'base':'Fluo-N3DH-CE_challenge/','dind':2,'dset':'chall2','n':190,'z':170,'sh':(170,256,356),'osh':(31, 512, 712),'path':Path('Fluo-N3DH-CE_challenge/02/')}

## ipy
from gputools import pad_to_shape


## pimg -> centerpoint optimization is unncecessary.

@DeprecationWarning
def optimize_seg_separate_net(results_gt, homedir, savedir):
  if data is None:
    data = labnames2imgs_cens(labnames(1),1)
  results = results_gt['train']

  for d,pimg in zip(data,results['pimg']):

    z2 = d['z2']
    img = d['img']

    def f_eval(params):
      hyp = segmentation.segment_pimg_img(pimg[...,1], params)
      s,n = scores_dense.seg(d['lab'],hyp[z2],partial_dataset=True)
      seg = s/n if n>0 else 0
      print(params,seg,n)
      return -seg

    trials = ho.Trials()
    best = ho.fmin(f_eval,
      space=space,
      algo=ho.tpe.suggest,
      max_evals=15,
      trials=trials)
    best = ho.space_eval(space,best)

    hyp = segmentation.segment_pimg_img(pimg, best)[z2]
    show = seg_compare2gt_rgb(hyp,lab,img[0,z2])
    io.imsave(savedir / 'seg_{:03d}.png'.format(t),show)

    pickle.dump(trials, open(savedir / 'trials{:03d}.pkl'.format(t), 'wb'))
    print(trials.best_trial['result']['loss'])
    dc.append({'trials':trials})

  dc = invertdict(dc,f=lambda x:x)
  print("DONE: losses are: ", [t.best_trial['result']['loss'] for t in dc['trials']])

## build training data

# def build_challenge_traindata(homedir):

@DeprecationWarning
def build_rawdata(homedir):
  # raw = {'train':times2raw([10,20,50,100,150],1,homedir), 'vali':times2raw([25,105],1,homedir)}
  raw = {'train':times2raw([100],1,homedir), 'vali':times2raw([105],1,homedir)}
  # raw = {'train':times2raw([10, 60, 140, 180],1,homedir), 'vali':times2raw([20, 75, 155, 175],1,homedir)}
  # raw = {'train':times2raw([50],1,homedir), 'vali':times2raw([55],1,homedir)}
  return raw

def build_gt_rawdata():
  raw = {'1':times2raw([ 21,  28,  78, 141, 162],1,basedir='Fluo-N3DH-CE'),
         '2':times2raw([10, 12, 106, 120, 126],2,basedir='Fluo-N3DH-CE')}
  return raw


@DeprecationWarning
def update_weights(rawdata, r=10/7):
  ws = rawdata['train']['weights']
  cen = rawdata['train']['target'][...,0]
  ws[cen==1] = ws[cen==1]*r
  ws = ws/ws.mean((1,2,3),keepdims=True)
  rawdata['train']['weights'] = ws

  ws = rawdata['vali']['weights']
  cen = rawdata['vali']['target'][...,0]
  ws[cen==1] = ws[cen==1]*r
  ws = ws/ws.mean((1,2,3),keepdims=True)
  rawdata['vali']['weights'] = ws


## do everything under the sun

## displaying raw data

def midplane(arr,i):
  ss = [slice(None) for _ in arr.shape]
  n = arr.shape[i]
  ss[i] = slice(n//3, (2*n)//3)
  return arr[ss].max(i)

def plotlist(lst, i, c=5, norm=norm_szyxc_per, mid=midplane):
  "takes a list of form [ndarray, ndarray, ...]. each has axes 'SZYXC' "
  lst2 = [norm(mid(data,i)) for data in lst]
  lst2[0][...,2] = 0 # turn off blue channel in xs
  lst2[0][...,0] = 0 # turn off red channel in xs
  res = ts.plotgrid(lst2,c=c)
  return res

def show_trainvali(trainable, savepath):
  # xsem = trainable['xsem']
  # ysem = trainable['ysem']
  # xrgb = [xsem['nuc'], xsem['nuc'], xsem['nuc']]
  # yrgb = [ysem['gauss'], ysem['gauss'], ysem['gauss']]
  xrgb = [0,0,0]
  yrgb = [0,0,0]
  visuals = {'xrgb':xrgb, 'yrgb':yrgb, 'plotlist':plotlist}
  old = new2old_trainable(trainable)
  ts.show_trainvali(old, visuals, savepath)



# Deprecated!!!

def build_trainable(rawdata):
  res = {'train': build_single_trainable(rawdata['train']),
          'vali': build_single_trainable(rawdata['vali']),
          }
  return res

def build_single_trainable(rawdata):
  source = rawdata['source']
  target = rawdata['target']
  weights = rawdata['weights']

  ## add extra cell center channel
  patchsize = [1,8*15,8*15,8*15]
  borders = (0,0,0,0)
  res = patchmaker.patchtool({'img':source.shape[:-1], 'patch':patchsize, 'borders':borders}) #'overlap_factor':(2,1,1)})
  slices = res['slices_padded']

  ## pad images
  padding = [(b,b) for b in borders] + [(0,0)]
  source  = np.pad(source, padding, mode='constant')
  target  = np.pad(target, padding, mode='constant')
  weights = np.pad(weights, padding[:-1], mode='constant')

  ## reduce data (same for train and vali)
  # slices = [ss for ss in slices[::3] if weights[ss].mean() > 0.5]
  # slices = [ss for ss in slices[::3]]

  ## extract slices. zero index to forget about time dimension.
  xs = np.array([source[tuple(ss)][0] for ss in slices])
  ys = np.array([target[tuple(ss)][0] for ss in slices])
  ws = np.array([weights[tuple(ss)][0] for ss in slices])

  ## fix ys
  # ys = (ys > ys.max()*0.85)[...,0]
  # ys = ys[...,0]
  # ys = np_utils.to_categorical(ys).reshape(ys.shape + (-1,))
  ys = cat([ys,ws[...,np.newaxis]],-1)

  ## normalize over space. sample and channel independent
  xs = xs/np.mean(xs,(1,2,3), keepdims=True)
  
  print(xs.shape, ys.shape, ws.shape)

  res = {'xs':xs,'ys':ys,'ws':ws,'slices':slices}
  return res

def new_target_from_gt(optres, rawgt):
  rawgt['gt']['target2'] = np.array([lab2bgdist(x['hyp'])[...,np.newaxis] for x in optres])

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

## Tue Aug 21 18:34:08 2018
copy over the 3d unet for use with celegans data



"""