# %load_ext autoreload
# %autoreload 2
from ipython_remote_defaults import *

from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard, ReduceLROnPlateau

import unet
import lib as ll
import augmentation
from segtools import lib as seglib
from segtools import segtools_simple as ss
from segtools import plotting
from scipy.ndimage.filters import convolve
import stack_segmentation as stackseg
import hyperopt
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials
from hyperopt import base as hopt_base
from hyperopt.plotting import main_plot_histogram, main_plot_history, main_plot_vars

## boolean params for control flow
TRAIN = True
PREDICT = True
PIMG_ANALYSIS = True
INSTANCE_SEG = True
INSTANCE_ANALYSIS = True
RERUN = not TRAIN  # and PREDICT and PIMG_ANALYSIS and INSTANCE_ANALYSIS)
QUICK = False

## boolean model params
border_weights = False
blur = False

## note: all paths are relative to project home directory

mypath = Path(sys.argv[1])
mypath.mkdir(exist_ok=RERUN)
if not run_from_ipython():
  myfile = Path(__file__)
  print(mypath / myfile.name)
  if not RERUN:
    shutil.copy(myfile, mypath / 'train_and_seg.py')
  sys.stdout = open(mypath / 'stdout.txt', 'w')
  sys.stderr = open(mypath / 'stderr.txt', 'w')


## define input channels
dz = 4
n_channels = 2*(1+2*dz)
ch_mem_xs = 2*dz # original, centered membrane channel
ch_nuc_xs = 2*dz + 1
ch_mem_img = 0
ch_nuc_img = 1
rgbimg = [ch_mem_img, ch_nuc_img, ch_nuc_img]
rgbxs  = [ch_mem_xs, ch_nuc_xs, ch_nuc_xs]

## define classes
n_classes = 5
ch_div = 3
ch_nuc = 1
ch_mem = 0
ch_bg  = 2
ch_ignore = 1
ch_distance = 4
rgbmem = [ch_mem, ch_nuc, ch_bg]
rgbdiv = [ch_div, ch_nuc, ch_bg]
classweights = [1/n_classes,]*n_classes

unet_params = {
  'n_pool' : 3,
  'inputchan' : n_channels,
  'n_classes' : n_classes,
  'n_convolutions_first_layer' : 64,
  'dropout_fraction' : 0.2,
}

# set to None to redo optimization
segmentation_params = {'nuc_mask':0.51, 'nuc_seed':0.99, 'compactness':0.5}
# segmentation_params = None

## if segmentation_params is Noene then optimize this space
segmentation_space = {'nuc_mask' :hp.uniform('nuc_mask', 0.3, 0.7),
                       'nuc_seed':hp.uniform('nuc_seed', 0.9, 1.0),
                       'mem_mask':hp.uniform('mem_mask', 0.0, 0.3),
                       'mem_seed':hp.uniform('mem_seed', 0.0, 0.3),
                       'compactness' : 0, #hp.uniform('compactness', 0, 10),
                       'connectivity': 1, #hp.choice('connectivity', [1,2,3]),
                       }
segmentation_info   = {'name':'watershed_two_chan', 'param0':'mem_mask', 'param1':'mem_seed'}

stack_segmentation_function = lambda x,p : stackseg.watershed_two_chan(x, **p)
if INSTANCE_SEG and segmentation_params is not None and not run_from_ipython():
  mypath_opt = add_numbered_directory(mypath, 'opt')

if TRAIN:
  json.dump(unet_params, open(mypath / 'unet_params.json', 'w'))
  net = unet.get_unet_n_pool(**unet_params)

def condense_labels(lab):
  lab[lab==0]   = ch_bg
  lab[lab==255] = ch_mem
  lab[lab==168] = ch_nuc
  lab[lab==85]  = ch_div
  lab[lab==198] = ch_ignore
  return lab

## load data
img = imread('data/img006.tif')
lab = imread('data/labels_lut.tif')
lab = lab[:,:,0]
lab = condense_labels(lab)
## TODO: this will break once we start labeling XZ and YZ in same volume.
mask_labeled_slices = lab.min((2,3)) < 2
inds_labeled_slices = np.indices(mask_labeled_slices.shape)[:,mask_labeled_slices]

def split_train_vali():
  ys = lab[mask_labeled_slices].copy()
  xs = []
  for t,z in inds_labeled_slices.T:
    res = ll.add_z_to_chan(img[t], dz, ind=z)
    xs.append(res[0])
  xs = np.array(xs)

  # convert ys to categorical
  ys = np_utils.to_categorical(ys).reshape(ys.shape + (-1,))

  # split 400x400 into 16x100x100 patches and reshape back into "SYXC"
  # axes are "SYXC"
  r = 2
  xs = collapse(splt(xs,[r,r],[1,2]), [[0,1,3],[2],[4],[5]])
  ys = collapse(splt(ys,[r,r],[1,2]), [[0,1,3],[2],[4],[5]])

  np.savez(mypath / 'traindat_small', xs=xs[::5], ys=ys[::5])

  ## turn off the `ignore` class
  # mask = ys[...,3]==1
  # ys[mask] = 0

  ## reweight border pixels
  if border_weights:
    a,b,c,d = ys.shape
    mask = ys[...,ch_mem]==1 # ????
    distimg = ~mask
    distimg = np.array([distance_transform_edt(d) for d in distimg])
    distimg = np.exp(-distimg/10)
    # distimg = distimg/distimg.mean((1,2), keepdims=True)
    ys = ys*distimg[...,np.newaxis]
    ys = ys / ys.mean((1,2,3), keepdims=True)

  print(xs.shape,ys.shape,mask_labeled_slices.shape)
  print(ys.max((0,1,2)))

  ## normalize
  xs = xs/xs.mean((1,2), keepdims=True)

  ## shuffle
  inds = np.arange(xs.shape[0])
  np.random.shuffle(inds)
  invers = np.argsort(np.arange(inds.shape[0])[inds])
  xs = xs[inds]
  ys = ys[inds]
  if QUICK:
    xs = xs[:7*6]
    ys = ys[:7*6]

  ## train vali split
  split = 5
  n_vali = xs.shape[0]//split
  xs_train = xs[:-n_vali]
  ys_train = ys[:-n_vali]
  xs_vali  = xs[-n_vali:]
  ys_vali  = ys[-n_vali:]
  return xs_train, xs_vali, ys_train, ys_vali

def show_trainvali():
  sx = [slice(None,5), Ellipsis, rgbxs]
  sy = [slice(None,5), Ellipsis, rgbmem]
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
  res = multicat([[xt,xv,2], [yt,yv,2], 2])
  res = res[:,::2,::2]
  res = merg(res, 0) #[[0,1],[2],[3]])
  return res

def compile_and_callbacks():
  optim = Adam(lr=1e-4)
  loss  = unet.my_categorical_crossentropy(weights=classweights, itd=4)
  net.compile(optimizer=optim, loss=loss, metrics=['accuracy'])

  checkpointer = ModelCheckpoint(filepath=str(mypath / "w001.h5"), verbose=0, save_best_only=True, save_weights_only=True)
  earlystopper = EarlyStopping(patience=30, verbose=0)
  reduce_lr    = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
  callbacks = [checkpointer, earlystopper, reduce_lr]
  return callbacks

def random_augmentation_xy(xpatch, ypatch, train=True):
  if random.random()<0.5:
      xpatch = np.flip(xpatch, axis=1)
      ypatch = np.flip(ypatch, axis=1)
  # if random.random()<0.5:
  #     xpatch = np.flip(xpatch, axis=0)
  #     ypatch = np.flip(ypatch, axis=0)
  if random.random()<0.5:
      # randangle = (random.random()-0.5)*60 # even dist between ± 30
      randangle = 90 * random.randint(0,3)
      xpatch  = rotate(xpatch, randangle, reshape=False, mode='reflect')
      ypatch  = rotate(ypatch, randangle, reshape=False, mode='reflect')
  # if train:
    # if random.random()<0.5:
    #     delta = np.random.normal(loc=0, scale=5, size=(2,3,3))
    #     xpatch = augmentation.warp_multichan(xpatch, delta=delta)
    #     ypatch = augmentation.warp_multichan(ypatch, delta=delta)
    # if random.random()<0.5:
    #     m = random.random()*xpatch.mean() # channels already normed
    #     s = random.random()*xpatch.std()
    #     noise = np.random.normal(m/4,s/4,xpatch.shape).astype(xpatch.dtype)
    #     xpatch += noise
  xpatch = xpatch.clip(min=0)
  xpatch = xpatch/xpatch.mean((0,1))
  return xpatch, ypatch

def train():
  batchsize = 3
  stepsperepoch   = xs_train.shape[0] // batchsize
  validationsteps = xs_vali.shape[0] // batchsize

  f_trainaug = lambda x,y:random_augmentation_xy(x,y,train=False)

  bgen = unet.batch_generator_patches_aug(xs_train, ys_train,
                                    # steps_per_epoch=100,
                                    batch_size=batchsize,
                                    augment_and_norm=f_trainaug,
                                    savepath=mypath)

  f_valiaug = lambda x,y:random_augmentation_xy(x,y,train=False)

  vgen = unet.batch_generator_patches_aug(xs_vali, ys_vali,
                                    # steps_per_epoch=100,
                                    batch_size=batchsize,
                                    augment_and_norm=f_valiaug,
                                    savepath=None)

  history = net.fit_generator(bgen,
                    steps_per_epoch=stepsperepoch,
                    epochs=20,
                    verbose=1,
                    callbacks=callbacks,
                    validation_data=vgen,
                    validation_steps=validationsteps)

  net.save_weights(mypath / 'w002.h5')
  return history
  
def plot_hist_key(k):
  plt.figure()
  y1 = history.history[k]
  y2 = history.history['val_' + k]
  plt.plot(y1, label=k)
  plt.plot(y2, label='val_' + k)
  plt.legend()
  plt.savefig(mypath / (k + '.png'))

def predict_train_vali():
  pred_xs_train = net.predict(xs_train)
  res = collapse(splt(pred_xs_train[:64,...,rgbmem],8,0), [[0,2],[1,3],[4]])
  io.imsave(mypath / 'pred_xs_train.png', res)
  pred_xs_vali = net.predict(xs_vali)
  res = collapse(splt(pred_xs_vali[...,rgbmem],4,0), [[0,2],[1,3],[4]])
  return res

if TRAIN:
  xs_train, xs_vali, ys_train, ys_vali = split_train_vali()
  res = show_trainvali()
  io.imsave(mypath / 'train_vali_ex.png', res)
  callbacks = compile_and_callbacks()
  history = train()
  print(history.history, file=open(mypath / 'history.txt','w'))
  plot_hist_key('loss')
  plot_hist_key('acc')
  ## load best weights from previous training
  net.load_weights(mypath / 'w001.h5')
  res = predict_train_vali()
  io.imsave(mypath / 'pred_xs_vali.png', res)

## Predictions on full dataset!

def predict_on_new():
  # x = perm(img,"tzcyx","tzcyx")
  res = []
  for t in range(img.shape[0]):
    x = img[t]
    if QUICK:
      x = x[[9],:b//4] # downsize for QUICK testing
    x = ll.add_z_to_chan(x, dz, axes="ZCYX")
    x = x / x.mean((1,2), keepdims=True)
    pimg = net.predict(x)
    pimg = pimg.astype(np.float16)
    # qsave(collapse(splt(pimg[:64,::4,::4,rgbdiv],8,0),[[0,2],[1,3],[4]]))
    res.append(pimg)
  res = np.array(res)
  return res

if PREDICT:
  pimg = predict_on_new()
  np.save(mypath / 'pimg', pimg)

def showresults():
  x = img[mask_labeled_slices].transpose((0,2,3,1))[...,rgbimg]
  x[...,2] = 0 # remove blue
  y = lab[mask_labeled_slices]
  y = np_utils.to_categorical(y).reshape(y.shape + (-1,))
  y = y[...,rgbmem]
  z = pimg[mask_labeled_slices][...,rgbmem]
  ss = [slice(None,None,5), slice(None,None,4), slice(None,None,4), slice(None)]
  def f(r):
    r = merg(r[ss])
    r = r / np.percentile(r, 99, axis=(0,1), keepdims=True)
    r = np.clip(r,0,1)
    return r
  x,y,z = f(x), f(y), f(z)
  res = np.concatenate([x,y,z], axis=1)
  return res

def find_divisions():
  x = pimg.astype(np.float32)
  x = x[:,::6] ## downsample z
  div = x[...,ch_div].sum((2,3))
  val_thresh = np.percentile(div.flat, 95)
  n_rows, n_cols = 7, 7
  tz = np.argwhere(div > val_thresh)[:n_rows]
  lst = list(range(x.shape[0]))
  x2 = np.array([x[timewindow(lst, n[0], n_cols), n[1]] for n in tz])
  x2 = collapse(x2, [[0,2],[1,3],[4]]) # '12yxc' -> '[1y][2x][c]'
  x2 = x2[::4,::4,rgbdiv]
  # x2[...,0] *= 40 ## enhance division channel color!
  # x2 = np.clip(x2, 0, 1)
  return x2

if PIMG_ANALYSIS:
  pimg = np.load(mypath / 'pimg.npy')
  res = showresults()
  io.imsave(mypath / 'results.png', res)
  ## max division channel across z
  np.save(mypath / 'max_z_divchan', pimg[:,...,ch_div].max(1))
  res = find_divisions()
  io.imsave(mypath / 'find_divisions.png', res)

## compute instance segmentation statistics  

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

def lab2instance(x):
  x[x!=1] = 0
  x = label(x)[0]
  return x

def optimize(pimg):

  ## optimization params
  
  def risk(params):
    img_instseg = pimg[[0]]
    inds = inds_labeled_slices[:,:-4] # only use first timepoint
    gt_slices = np.array([lab2instance(x) for x in lab[inds[0], inds[1]]])
    print('Evaluating params:',params)
    t0 = time()
    hyp = np.array([stack_segmentation_function(x,params) for x in img_instseg])
    hyp = hyp.astype(np.uint16)
    pred_slices = hyp[inds[0], inds[1]]
    res = np.array([ss.seg(x,y) for x,y in zip(gt_slices, pred_slices)])
    t1 = time()
    val = res.mean()
    return -val

  ## perform the optimization

  trials = Trials()
  best = fmin(risk,
      space=segmentation_space,
      algo=tpe.suggest,
      max_evals=100,
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

  plt.figure()
  main_plot_histogram(trials=trials)
  plt.savefig(mypath_opt / 'hypopt_histogram.pdf')

  plt.figure()
  main_plot_history(trials=trials)
  plt.savefig(mypath_opt / 'hypopt_history.pdf')

  domain = hopt_base.Domain(risk, segmentation_space)

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

  return best['vals']


if INSTANCE_SEG:
  pimg = np.load(mypath / 'pimg.npy')

  if blur:
    pimg = convolve_zyx(pimg)

  if segmentation_params is None:
    print("BEGIN SEGMENTATION OPTIMIZATION")
    params = optimize(pimg)
  else:
    params = segmentation_params

  print("COMPUTE FINAL SEGMENTATION")
  hyp = np.array([stack_segmentation_function(x, params) for x in pimg])
  hyp = hyp.astype(np.uint16)
  np.save(mypath / 'hyp', hyp)

  print("COMPARE SEGMENTATION AGAINST LABELED SLICES")
  inds = inds_labeled_slices[:,:-4] # use full
  gt_slices  = np.array([lab2instance(x) for x in lab[inds[0], inds[1]]])
  pre_slices = hyp[inds[0], inds[1]]
  seg_scores = np.array([ss.seg(x,y) for x,y in zip(gt_slices, pre_slices)])
  print({'seg': seg_scores.mean(), 'std':seg_scores.std()}, file=open(mypath / 'SEG.txt','w'))
  
  print("CREATE DISPLAY GRID OF SEGMENTATION RESULTS")
  img_slices = img[inds[0], inds[1]]
  gspec = matplotlib.gridspec.GridSpec(4, 4)
  gspec.update(wspace=0., hspace=0.)
  fig = plt.figure()
  ids = np.arange(img_slices.shape[0])
  # np.random.shuffle(ids)
  ids = np.concatenate([ids[:24:2],ids[-4:]])
  for i in range(16):
    ax = plt.subplot(gspec[i])
    im1 = img_slices[ids[i],ch_nuc_img,]
    im2 = pre_slices[ids[i]]
    im3 = gt_slices[ids[i]]
    ax = plotting.make_comparison_image(im1,im2,im3,ax=ax)
    # res.append(ax.get_array())
    ax.set_axis_off()
  fig.set_size_inches(10, 10, forward=True)
  plt.savefig(mypath / 'seg_overlay.png',dpi=200, bbox_inches='tight')

if INSTANCE_ANALYSIS:
  hyp = np.load(mypath / 'hyp.npy')
  ## look at cell shape statistics
  print("GENERATE NHLS FROM HYP AND ANALYZE")
  nhls = seglib.labs2nhls(hyp, img[:,:,ch_nuc_img], simple=False)
  pickle.dump(nhls, open(mypath / 'nhls.pkl', 'wb'))

  plt.figure()
  cm = sns.cubehelix_palette(len(nhls))
  for i,nhl in enumerate(nhls):
    areas = [n['area'] for n in nhl]
    areas = np.log2(sorted(areas))
    plt.plot(areas, '.', c=cm[i])
  plt.savefig(mypath / 'cell_sizes.pdf')


#hi coleman! keep coding!
#best, romina :)


