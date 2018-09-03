from segtools.defaults.ipython import *
from segtools.defaults.training import *
import keras
from segtools import label_tools
from segtools import graphmatch
from segtools import render

import matplotlib.pyplot as plt
plt.switch_backend('Agg')

import lib

import gputools
import ipdb
import pandas as pd

from contextlib import redirect_stdout
import train_seg_lib as ts
from sklearn.metrics import confusion_matrix

import gputools
cat = np.concatenate
import scipy.ndimage.morphology as morph
from sklearn.mixture import GaussianMixture

## testing segmentations against GT

## load challenge GT data

def labnames2imgs_cens(labnames,n):
  data = []
  for ln in labnames:
    t,z = (int(ln[-11:-8]), int(ln[-7:-4]))
    z2  = floor((z+0.5)*5.5)
    img = imread(imgnames(n)[t])
    cen = imread(cennames(n)[t])

    lab = imread(ln)
    lab = zoom(lab, (0.5,0.5), order=0)
    lab = label(lab)[0]
    
    img = gputools.scale(img, (5.5,0.5,0.5))
    img = img[...,np.newaxis]

    ## upscale centerpoints. make them just a single pixel.
    pts = np.array([n['centroid'] for n in nhl_tools.hyp2nhl(cen)]).astype(np.int)
    pts = np.floor((pts+0.5)*[5.5,0.5,0.5]).astype(np.int) 
    cen = np.zeros(img.shape[:-1])
    cen[tuple(pts.T)] = 1
    data.append({'img':img, 'cen':cen, 'lab':lab, 't':t, 'z':z, 'z2':z2, 'labname':ln})
  return data

def labnames(n):
  return sorted(glob('Fluo-N3DH-CE/{:02d}_GT/SEG/man_seg_*'.format(n)))

def imgnames(n):
  return sorted(glob('Fluo-N3DH-CE/{:02d}/t???.tif'.format(n)))

def cennames(n):
  return sorted(glob('Fluo-N3DH-CE/{:02d}_GT/TRA/man_track???.tif'.format(n)))

def gtdata():
  return {1: labnames2imgs_cens(labnames(1),1), 2: labnames2imgs_cens(labnames(2),2)}

## optimization routines. optimize watershed segmentation on GT centerpoints.

def optimize_seg_separate(homedir, savedir, data=None):
  homedir = Path(homedir)

  if data is None:
    data = labnames2imgs_cens(labnames(1),1)

  space = segment_paramspace()

  res = []

  for d in data:

    savedir2 = savedir / 'd{:03d}'.format(d['t'])
    savedir2.mkdir(exist_ok=True)
    print('\n')
    print(d['labname'])

    def f_eval(params):
      score,n_gt,n_hyp = evalme(d,params,savedir2)
      print(score/n_gt, n_gt, n_hyp)
      return -score/n_gt

    trials = ho.Trials()
    best = ho.fmin(f_eval,
      space=space,
      algo=ho.tpe.suggest,
      max_evals=5,
      trials=trials)
    best = ho.space_eval(space,best)

    seg_and_save_d(d,best,savedir)
    pickle.dump(trials, open(savedir / 'trials.pkl', 'wb'))
    print("Best params: ", best)
    seg = trials.best_trial['result']['loss']
    res.append({'hyp':segment(d,best), 'params':best, 'seg':-seg})

  return res

def seg_and_save_d(d,params,savedir):
  hyp = segment(d, params)[d['z2']]
  show = seg_compare2gt_rgb(hyp,d['lab'],d['img'][d['z2'],...,0])
  s1 = scores_dense.seg(d['lab'],hyp)
  # s2,n_gt = scores_dense.seg(d['lab'],hyp, partial_dataset=True)
  io.imsave(savedir / 'seg_1_time_{:03d}_SEG_{:.3f}.png'.format(d['t'], s1), show)
  print("saved ", d['labname'], "with seg = ", s1)

def optimize_seg_joint(homedir, savedir, data=None):

  if data is None:
    data = labnames2imgs_cens(labnames(1),1)

    space = segment_paramspace()

  def f_eval(params):
    scores = np.array([evalme(d,params,savedir) for d in data])
    tot,n_gt,n_hyp = scores.sum(0)
    mu = tot / n_gt
    sig = np.std(scores[:,0]/scores[:,1])
    print(params, mu, sig)
    return -mu

  trials = ho.Trials()
  best = ho.fmin(f_eval,
    space=space,
    algo=ho.tpe.suggest,
    max_evals=30,
    trials=trials)
  best = ho.space_eval(space,best)

  for d in data:
    seg_and_save_d(d,best,savedir)

  pickle.dump(trials, open(savedir / 'trials.pkl', 'wb'))
  print(best)

def evalme(d, params, savedir):
  hyp = segment(d, params)[d['z2']]
  n_hyp = label(hyp)[0].max()-1
  seg, n_gt = scores_dense.seg(d['lab'], hyp, partial_dataset=True)

  if False:
    def norm(x): return (x-x.min())/(x.max()-x.min())
    out = [norm(x) for x in [d['img'][z2,...,0],d['lab'],d['cen'][z2-3:z2+3].max(0),hyp[z2]]]
    out = cat(out,1)
    savedir2 = savedir / '_'.join([str(k)+'_'+str(v)[:4] for k,v in params.items()])
    savedir2.mkdir(exist_ok=True)
    io.imsave(savedir2 / "img{:03d}.png".format(d['t']), out)
    # print(params, seg, n_gt)
  return seg, n_gt, n_hyp

def segment(d, params):
  # print(d['img'].max(), d['img'].min(), d['img'].mean())
  img2 = gputools.denoise.nlm3(d['img'][...,0], sigma=params['sigma'])
  hx = np.array([1,1,1]) / 3
  for _ in range(params['nblur']):
    img2 = gputools.convolve_sep3(img2,hx,hx,hx)
  hyp = watershed(img2, label(d['cen']>0)[0], mask=img2>img2.mean()*params['m_mask'], compactness=params['compactness'])
  return hyp

def segment_paramspace():
  space = dict()
  space['sigma']       = ho.hp.uniform('sigma',      1,50)
  space['nblur']       = ho.hp.choice('nblur',       [0,1,2,3])
  space['m_mask']      = ho.hp.uniform('m_mask',     1.0,2.0)
  space['compactness'] = ho.hp.choice('compactness', [10, 100, 500, 1000])
  return space

def seg_compare2gt_rgb(hyp,lab,img):
  "takes 2d slices [no channels] and makes rgb"
  hyp = hyp.copy()
  # img = img.copy()
  bor = label_tools.find_boundaries(hyp)
  hyp[hyp>0] = 1.0
  hyp[bor] = 0.5
  img = norm_szyxc_per(img,(0,1))
  img = img[...,np.newaxis]
  img = img[...,[0,0,0]]
  img[...,[0,2]] = 0
  img[bor] += 1
  bor_gt = label_tools.find_boundaries(lab)
  img[bor_gt] += [-.7,-.7,1]
  img = img.clip(0,1)
  return img

## GMM is WAAAAY too slow

def load_and_fit_gmm(homedir, savedir, data=None):
  homedir = Path(homedir)

  if data is None:
    data = labnames2imgs_cens(labnames(1),1)

  for i in range(len(labnames)):
    ln = labnames[i]
    tz = (int(ln[-11:-8]), int(ln[-7:-4]))
    t,z = tz
    lab = imread(ln)
    img = imread('Fluo-N3DH-CE/01/t{:03d}.tif'.format(t))
    cen = imread('Fluo-N3DH-CE/01_GT/TRA/man_track{:03d}.tif'.format(t))
    
    img = gputools.scale(img, (5.5,0.5,0.5))
    lab = zoom(lab, (0.5,0.5), order=0)
    lab = label(lab)[0]
    # cen = label(cen)[0].astype(np.uint16)
    cen = (cen>0).astype(np.float32)
    cen = gputools.scale(cen, (5.5,0.5,0.5), interpolation='linear')
    cen = label(cen>0.5)[0]
    # scores = []
    img2 = gputools.denoise.nlm3(img, 20)
    hx = np.array([1,1,1])/3
    img3 = gputools.convolve_sep3(img2, hx,hx,hx)
    gm = fitgauss(img3, cen)
    return gm

def fitgauss(img,labcen):
  nhl = nhl_tools.hyp2nhl(labcen)
  nc  = len(nhl)
  points = [n['centroid'] for n in nhl]
  gm = GaussianMixture(n_components=nc,
                       covariance_type='spherical',
                       tol=0.001,
                       reg_covar=1e-06,
                       max_iter=100,
                       n_init=1,
                       init_params='kmeans',
                       weights_init=None,
                       means_init=points,
                       precisions_init=None,
                       random_state=None,
                       warm_start=False,
                       verbose=0,
                       verbose_interval=10)
  X = np.indices(img.shape).reshape((3,-1)).T
  gm.fit(X)
  return gm

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
      hyp = segment_pimg_img(pimg[...,1], params)
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

    hyp = segment_pimg_img(pimg, best)[z2]
    show = seg_compare2gt_rgb(hyp,lab,img[0,z2])
    io.imsave(savedir / 'seg_{:03d}.png'.format(t),show)

    pickle.dump(trials, open(savedir / 'trials{:03d}.pkl'.format(t), 'wb'))
    print(trials.best_trial['result']['loss'])
    dc.append({'trials':trials})

  dc = invertdict(dc,f=lambda x:x)
  print("DONE: losses are: ", [t.best_trial['result']['loss'] for t in dc['trials']])

## compute seg scores on training datasets

def mk_hyps_compute_seg(data, results_gt):
  pimgs = results_gt['pimg']
  best = {'compactness': 500, 'm_mask': 1.519235931075873, 'nblur': 1, 'sigma': 15.106708203111125}
  best.update({'thi':0.6})
  
  seg,ngt = 0,0
  for d,p in zip(data,pimgs):
    d['pimg'] = p[...,1]
    hyp = segment_pimg_img(d, best)
    s,n = scores_dense.seg(d['lab'], hyp[d['z2']], partial_dataset=True)
    seg+=s
    ngt+=n
    print(s,n)
  print(seg, ngt, seg/ngt)

def segment_pimg_img(d, params):
  img = d['img'][...,0].astype(np.float32)
  pimg = d['pimg']
  hx = np.array([1,1,1]) / 3
  img2 = gputools.denoise.nlm3(img, sigma=params['sigma'])
  for _ in range(params['nblur']):
    img2 = gputools.convolve_sep3(img2,hx,hx,hx)
  hyp = watershed(-img2, label(pimg > params['thi'])[0], mask=img2>img2.mean()*params['m_mask'], compactness=params['compactness'])
  return hyp

## utils

def invertdict(lod, f=lambda x: np.array(x)):
  d2 = dict()
  for k in lod[0].keys():
    d2[k] = f([x[k] for x in lod])
  return d2

def revertdict(dol):
  res = []
  for i in range(len(list(dol.values())[0])):
    print(i)
    res.append({k:v[i] for k,v in dol.items()})
  return res

## build training data

def build_rawdata(homedir):
  # raw = {'train':times2raw([10,20,50,100,150],1,homedir), 'vali':times2raw([25,105],1,homedir)}
  # raw = {'train':times2raw([100],1,homedir), 'vali':times2raw([105],1,homedir)}
  raw = {'train':times2raw([15, 70, 150],1,homedir), 'vali':times2raw([20, 75, 155],1,homedir)}
  # raw = {'train':times2raw([50],1,homedir), 'vali':times2raw([55],1,homedir)}
  return raw

def build_gt_rawdata(homedir):
  raw = {'train':times2raw([ 21,  28,  78, 141, 162],1,homedir),
         'vali':times2raw([10, 12, 106, 120, 126],2,homedir)}
  return raw

def times2raw(times, n, homedir):
  homedir = Path(homedir)

  ds = []
  for i in times:
    img = imread(imgnames(n)[i])
    cen = imread(cennames(n)[i])
    
    img = gputools.scale(img, (5.5,0.5,0.5))
    img = img[...,np.newaxis]

    ## upscale centerpoints. make them just a single pixel.
    pts = np.array([n['centroid'] for n in nhl_tools.hyp2nhl(cen)]).astype(np.int)
    pts = np.floor((pts+0.5)*[5.5,0.5,0.5]).astype(np.int) 
    cen = np.zeros(img.shape[:-1])
    cen[tuple(pts.T)] = 1

    dat = dict()

    if False:
      w=6
      target,ksum = soft_gauss_target(pts,cen.shape,w=w)
      target = target[...,np.newaxis]
      weights = np.ones(target.shape[:-1])
      dat['tmax'] = target.max()

    if True:
      w=5
      target,ksum = make_hard_round_target(pts,cen.shape,w=w)
      weights = np.ones(target.shape)
      cm=3
      weights[target==1] *= cm
      weights = weights / weights.mean()
      target = np_utils.to_categorical(target).reshape(target.shape + (2,))
      dat['cm'] = cm

    dat.update({'source':img, 'target':target, 'weights':weights, 'ksum':ksum, 'w':w, 'ncells':pts.shape[0], 'pts':pts})
    ds.append(dat)

  ds = invertdict(ds)
  return ds

def make_hard_round_target(pts,outshape,w=3):
  k = 3*w
  d = np.indices((k,k,k))
  m = ((d-k//2)**2).sum(0) < w**2
  target = sum_kernels_at_points(pts,m,outshape)
  target[target>1] = 1
  return target, m.sum()

def soft_gauss_target(pts,outshape,w=6):
  sig = np.array([1,1,1])/w
  wid = w*6
  def f(x): return np.exp(-(sig*x*sig*x).sum()/2)
  kern = math_utils.build_kernel_nd(wid,3,f)
  target = sum_kernels_at_points(pts,kern,outshape)
  return target, kern.sum()

def sum_kernels_at_points(pts,kern,out_shape):
  st = np.array(kern.shape)
  sh = np.array(out_shape)
  output = np.zeros(out_shape)

  for p in pts:
    start = p - np.floor(st/2).astype(np.int)
    end   = p + np.ceil(st/2).astype(np.int)
    ss = patchmaker.se2slices(start,end)
    output[ss] += kern
  return output

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

## rendering

def show_rawdata(rawdata_train, pimg=None, i=1):
  xrgb = [0,0,0]
  yrgb = [1,1,1]
  src = rawdata_train['source'][...,xrgb]
  trg = rawdata_train['target'][...,yrgb]
  lzt = [src, trg]
  if pimg is not None:
    pim = pimg[...,yrgb]
    lzt = [src, trg, pim]
  def mid(szyxc,i):
    perm = ["sz,y,x,c", "sy,z,x,c", "sx,z,y,c"]
    return collapse2(szyxc,"szyxc",perm[i-1])[::10]
  res = plotlist(lzt,i,c=3,mid=mid)
  return res

def save_pimgs(net, savedir, a=0, b=250):
  times = range(a,b)
  counts = []
  n = 2
  for t in times:
    img = imread(imgnames(n)[t])
    img = gputools.scale(img, (5.5,0.5,0.5))

    pimgname = savedir / 'pimg_{:03d}.tif'.format(t)
    if pimgname.exists():
      pimg = imread(str(pimgname))
    else:
      img = img[np.newaxis,...,np.newaxis]
      pimg = predict(net, img, outchan=2)
      img = img[0,...,0]
      pimg = pimg[0,...,1].astype(np.float16)
      imsave(str(pimgname), pimg)
    counts.append((t, label(pimg>0.6)[1]))
  print(counts)
  print(counts, file=open(savedir / 'counts.txt','w'))
  return counts

def get_n_gt_cells_over_time():
  res = []
  cens = cennames(1)
  for t in range(len(cens)):
    # cens
    cen = imread(cens[t])
    res.append(label(cen)[1])
  return res

def pimgs2movie():
  times = range(0,250)
  a,b,c = (192, 256, 354)
  rate = 0.01
  decay = np.exp(np.arange(a//2)*rate).reshape((-1,1,1))
  decay = np.ones(a).reshape((-1,1,1))
  for t in times:
    png = io.imread('training/ce_013/render/p_{:03d}.png'.format(t))
    print(png.max(), png.min())
    # continue
    y,x,_ = png.shape
    pimgname = 'training/ce_014/render/pimg_{:03d}.tif'.format(t)
    pimg = imread(str(pimgname))
    pimg = pimg > 0.6
    zcolor = (pimg[:]*decay).argmax(0)
    png[y//2:,:x//2] = 255*viridis(zcolor) #[...,np.newaxis]
    zcolor = (pimg[::-1,:][:]*decay).argmax(0)
    png[y//2:,x//2:] = 255*viridis(zcolor) #[...,np.newaxis]
    io.imsave('training/ce_014/render2/zcolor{:03d}.png'.format(t), png)

def viridis(img):
  # img.ndim == 2. can be float or int.
  mi,ma = img.min(), img.max()
  cm = np.array(plt.get_cmap('viridis').colors)
  img2 = np.floor((img-mi)/(ma-mi)*255).astype(np.int)
  rgb = cm[img2.flat].reshape(img2.shape + (3,))
  return rgb

def fig3(data=None):
  pklname = Path("training/figures/fig003.pkl")
  dubs = range(1,15,3)
  if data is None:
    data = []
    for t in range(0,195):
      cen = imread(cennames(1)[t])
      # pimg = imread('training/ce_014/render/pimg_{:03d}.tif'.format(t))
      pimg = imread('training/ce_014/pimgs/pimg_{:03d}.tif'.format(t))
      pts = np.array(invertdict(nhl_tools.hyp2nhl(label(pimg>0.6)[0]))['centroid'])
      # pts_gt = np.array(invertdict(nhl_tools.hyp2nhl(label(cen)[0]))['centroid'])
      pts_gt = np.array([n['centroid'] for n in nhl_tools.hyp2nhl(cen)]).astype(np.int)
      pts_gt = np.floor((pts_gt+0.5)*[5.5,0.5,0.5]).astype(np.int) 
      n_match = [n_matchpoints(pts,pts_gt,k=1,dub=d) for d in dubs]
      data.append({'t':t, 'n_gt':pts_gt.shape[0], 'n_pr':pts.shape[0], 'n_match':n_match})
    data = invertdict(data)
  plt.figure()
  plt.plot(data['t'], data['n_gt'], label="GT")
  plt.plot(data['t'], data['n_pr'], label="PRED")
  for i,d in enumerate(dubs):
    plt.plot(data['t'], [x[i] for x in data['n_match']], label="MATCH dub {:02d}".format(d))
  plt.legend()
  plt.savefig("training/figures/fig003.pdf")
  pickle.dump(data,open(pklname,"wb"))
  return data

def fig4():
  pklname = Path("training/figures/fig004.pkl")
  dubs = range(1,15,3)
  if data is None:
    data = []
    for t in range(0,195):
      cen = imread(cennames(1)[t])
      pimg = imread('training/ce_014/render/pimg_{:03d}.tif'.format(t))
      pts = np.array(invertdict(nhl_tools.hyp2nhl(label(pimg>0.6)[0]))['centroid'])
      # pts_gt = np.array(invertdict(nhl_tools.hyp2nhl(label(cen)[0]))['centroid'])
      pts_gt = np.array([n['centroid'] for n in nhl_tools.hyp2nhl(cen)]).astype(np.int)
      pts_gt = np.floor((pts_gt+0.5)*[5.5,0.5,0.5]).astype(np.int) 
      n_match = [n_matchpoints(pts,pts_gt,k=1,dub=d) for d in dubs]
      data.append({'t':t, 'n_gt':pts_gt.shape[0], 'n_pr':pts.shape[0], 'n_match':n_match})
    data = invertdict(data)
  plt.figure()
  plt.plot(data['t'], data['n_gt'], label="GT")
  plt.plot(data['t'], data['n_pr'], label="PRED")
  for i,d in enumerate(dubs):
    plt.plot(data['t'], [x[i] for x in data['n_match']], label="MATCH dub {:02d}".format(d))
  plt.legend()
  plt.savefig("training/figures/fig004.pdf")
  pickle.dump(data,open(pklname,"wb"))
  return data

def figs_from_pimgs(loaddir, savedir, a=0, b=250):
  # times = range(0,40)
  # times = range(40,50)
  times = range(a,b)
  # times = range(0,250)
  counts = []
  n = 1
  for t in times:
    # img = imread(imgnames(n)[t])
    # img = gputools.scale(img, (5.5,0.5,0.5))

    pimgname = savedir / 'pimg_{:03d}.tif'.format(t)
    if pimgname.exists():
      pimg = imread(str(pimgname))
    else:
      img = img[np.newaxis,...,np.newaxis]
      pimg = predict(net, img, outchan=2)
      img = img[0,...,0]
      pimg = pimg[0,...,1].astype(np.float16)
      imsave(str(pimgname), pimg)
    counts.append((t, label(pimg>0.6)[1]))
    # show = render_frame([pimg, img])
    # io.imsave(savedir / 'p_{:03d}.png'.format(t), show)
    # nhl  = invertdict(nhl_tools.hyp2nhl(label(pimg>0.6)[0]))
    # plt.figure()
    # plt.plot(sorted(nhl['area']))
    # plt.savefig(str(savedir / 'area_{:03d}'.format(t)))
  print(counts)
  print(counts, file=open(savedir / 'counts.txt','w'))
  return counts

def render_frame(lzt):
  # dimension for ds 1 are (192, 256, 354, 1)
  pimg, img = lzt
  a,b,c = img.shape
  pimg = norm_szyxc_per(pimg,(0,1,2),[2,99.5])
  img = norm_szyxc(img,(0,1,2))
  x11 = render.decay(img[:a//2]) #.max(0)
  x21 = render.decay(img[a//2:]) #.max(0)
  x12 = render.decay(pimg[:a//2]) #.max(0)
  x22 = render.decay(pimg[a//2:]) #.max(0)
  top = cat([x11,x21],1)
  bot = cat([x12,x22],1)
  full = cat([top,bot],0)
  return full

def cell_count_over_time(savedir, a=50,b=80):
  for t in range(a,b):
    pimgname = 'training/ce_013/pimg_{:03d}.tif'.format(t)
    pimg = imread(str(pimgname))



## training

class Histories(keras.callbacks.Callback):
    def __init__(self,examples,savedir):
      self.examples = examples
      # self.ys = mini_ys
      self.savedir = savedir

    def on_train_begin(self, logs={}):
        self.aucs = []
        self.losses = []
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        # self.losses.append(logs.get('loss'))
        # y_pred = self.model.predict(self.model.validation_data[0])
        xrgb = [0,0,0]
        yrgb = [1,1,1]
        xs = np.array([x[0] for x in self.examples])
        ys = np.array([x[1] for x in self.examples])
        yp = self.model.predict(xs, batch_size=1)
        xs = xs[...,xrgb]
        ys = ys[...,yrgb]
        yp = yp[...,yrgb]
        res = plotlist([xs,ys,yp],1,c=1) #min(xs.shape[0],1))
        io.imsave(self.savedir / 'e_{:03d}.png'.format(epoch), res)
        # self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))
        return
 
    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

def build_net(inshape, out_channels, activation='softmax'):

  # xs_train = trainable['train']['xs']
  # ys_train = trainable['train']['ys']

  unet_params = {
    'n_pool' : 3,
    'n_convolutions_first_layer' : 32,
    'dropout_fraction' : 0.2,
    'kern_width' : 3,
  }

  mul = 2**unet_params['n_pool']
  faclist = [factors(x) for x in inshape[:-1]]
  for fac in faclist: assert mul in fac

  input0 = Input(inshape)
  unet_out = unet.get_unet_n_pool(input0, **unet_params)
  output2  = unet.acti(unet_out, out_channels, last_activation=activation, name='B')
  # output2  = unet.acti(unet_out, out_channels, last_activation='linear', name='B')
  net = Model(inputs=input0, outputs=output2)
  optim = Adam(lr=1e-4)


  if activation=='linear':
    def loss(yt, yp):
      w  = yt[...,-1]
      yt = yt[...,:-1]
      mse = losses.mean_squared_error(yt,yp)
      cellcount = (K.mean(yt) - K.mean(yp))**2
      return mse + 10.0 * cellcount

    def met0(yt, yp):
      w  = yt[...,-1]
      yt = yt[...,:-1]
      return K.std(yp)

    def met1(yt,yp):
      w  = yt[...,-1]
      yt = yt[...,:-1]
      cellcount = (K.mean(yt) - K.mean(yp))**2
      return cellcount

  elif activation=='softmax':
    def met0(yt,yp):
      return metrics.categorical_accuracy(yt[...,:-1], yp)
  
    loss = unet.crossentropy_w

    def met1(yt,yp):
      w  = yt[...,-1]
      yt = yt[...,:-1]
      cellcount = (K.mean(yt[...,1]) - K.mean(yp[...,1]))**2
      return cellcount


  net.compile(optimizer=optim, loss={'B':loss}, metrics={'B':[met0,met1]})
  return net

def datagen(rawdata_train, targetname='target'):
  src = rawdata_train['source']
  trg = rawdata_train[targetname]
  wgh = rawdata_train['weights']

  borders = np.array([0,20,20,20])
  padding = [(b,b) for b in borders] + [(0,0)]

  src = np.pad(src, padding, mode='constant')
  trg = np.pad(trg, padding, mode='constant')
  wgh = np.pad(wgh, padding[:-1], mode='constant')

  while True:
    patchshape = np.array([1,120,120,120])
    u_bound = np.array(src.shape[:-1]) - patchshape
    l_bound = [0,0,0,0]
    ind = np.array([random.randint(l_bound[i],u_bound[i]) for i in range(len("TZYX"))])
    ss = patchmaker.se2slices(ind,ind+patchshape)
    y = trg[tuple(ss)]
    w = wgh[tuple(ss)]
    x = src[tuple(ss)]
    x = x / x.mean((1,2,3))
    z = cat([y, w[...,np.newaxis]],-1)
    yield (x, z)

## analysis

def analyze_cpnet(net,rawdata,savedir):
  result = dict()
  for k in ['train', 'vali']:
    pimg = predict(net, rawdata[k]['source'], outchan=2)
    for i in [1,2,3]:
      res = show_rawdata(rawdata[k],pimg,i)
      io.imsave(savedir / 'pred_{}_{:02d}.png'.format(k,i), res)
    thresh,ncells = plot_cell_count_v_thresh(pimg[...,1], savedir / '{}_cell_count_plot.pdf'.format(k))
    data = []
    for i in range(pimg.shape[0]):
      pts_gt = rawdata[k]['pts'][i]
      pts = pimg2pts(pimg[i,...,1],th=0.6)
      dub,nmatches = plot_point_connections(pts_gt, pts, savedir / '{}_pts_connections_{:02d}.pdf'.format(k,i))
      data.append({'gt':pts_gt, 'pred':pts})
    result[k] = {'pimg':pimg, 'thresh':thresh, 'ncells':ncells, 'points':invertdict(data), 'dub':dub, 'nmatches':nmatches}
  return result

def pimg2pts(pimg,th=0.9):
  "axes are 'tzyxc'"
  lab = label(pimg>th)[0]
  nhl = nhl_tools.hyp2nhl(lab)
  pts = np.array([n['centroid'] for n in nhl])
  return pts

def n_matchpoints(pts0,pts1,k=1,dub=3):
  "dub is distance upper bound"
  g = graphmatch.kdmatch(pts0,pts1,k=1,dub=dub)
  nmatches = len(np.unique(g[g<len(pts1)]))
  return nmatches

def plot_point_connections(pts_gt,pts,filename=None):
  def f(d):
    return n_matchpoints(pts, pts_gt, k=1, dub=d)
  x = np.arange(15)
  y = [f(d) for d in x]
  fstr = "{:3d} "*15
  print(fstr.format(*x))
  print(fstr.format(*y))
  print('out of n_(gt,pred):', len(pts_gt), len(pts))
  if filename:
    plt.figure()
    plt.plot(x,y)
    plt.savefig(filename)
  return x,y

def plot_cell_count_v_thresh(pimg,filename):
  x = np.linspace(0,1,50)
  plt.figure()
  ys = []
  for i,img in enumerate(pimg):
    y = [label(img>th)[1] for th in x]
    ys.append(y)
    plt.plot(x,y,label=str(i))
  plt.legend()
  plt.savefig(filename)
  return x,ys



## Deprecated

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

## normalization, plotting, etc

def norm_szyxc(img,axs=(1,2,3)):
  mi,ma = img.min(axs,keepdims=True), img.max(axs,keepdims=True)
  img = (img-mi)/(ma-mi)
  return img

def norm_szyxc_per(img,axs=(1,2,3),pc=[2,99.9]):
  mi,ma = np.percentile(img,pc,axis=axs,keepdims=True)
  img = (img-mi)/(ma-mi)
  img = img.clip(0,1)
  return img

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

## prediction, requires net

def predict_trainvali(net, trainable, savepath):
  # xsem = trainable['xsem']
  # ysem = trainable['ysem']
  # xrgb = [xsem['nuc'], xsem['nuc'], xsem['nuc']]
  # yrgb = [ysem['gauss'], ysem['gauss'], ysem['gauss']]
  xrgb = [0,0,0]
  yrgb = [0,0,0]
  visuals = {'xrgb':xrgb, 'yrgb':yrgb, 'plotlist':plotlist}
  old = new2old_trainable(trainable)
  ts.predict_trainvali(net, old, visuals, savepath)

def new2old_trainable(trainable):
  res = dict()
  res['xs_train'] = trainable['train']['xs']
  res['xs_vali']  = trainable['vali']['xs']
  res['ys_train'] = trainable['train']['ys']
  res['ys_vali']  = trainable['vali']['ys']
  return res

def predict(net,img,outchan=1):
  container = np.zeros(img.shape[:-1] + (outchan,))
  
  # borders = [0,4,0,0]
  # patchshape_padded = [1,24,400,400]
  # borders = xsem['borders']
  borders = (0,30,30,30)
  # patchshape_padded = list(xsem['patchsize'])
  patchshape_padded = [1,8*15,8*15,8*15]
  # patchshape_padded[2] = 400
  # patchshape_padded[3] = 400
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