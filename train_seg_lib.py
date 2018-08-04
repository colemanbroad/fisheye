from segtools.defaults.ipython_remote import *
from segtools.defaults.training import *

from contextlib import redirect_stdout
import ipdb
import pandas as pd

from scipy.ndimage.filters import maximum_filter
from scipy.ndimage.morphology import generate_binary_structure, binary_erosion

ss = scores_dense

## utility functions (called internally)

def detect_peaks(image):
  """
  Takes an image and detect the peaks usingthe local maximum filter.
  Returns a boolean mask of the peaks (i.e. 1 when
  the pixel's value is the neighborhood maximum, 0 otherwise)
  """

  # define an 8-connected neighborhood
  neighborhood = generate_binary_structure(3,3)

  #apply the local maximum filter; all pixel of maximal value 
  #in their neighborhood are set to 1
  local_max = maximum_filter(image, footprint=neighborhood)==image
  #local_max is a mask that contains the peaks we are 
  #looking for, but also the background.
  #In order to isolate the peaks we must remove the background from the mask.

  #we create the mask of the background
  background = (image==0)

  #a little technicality: we must erode the background in order to 
  #successfully subtract it form local_max, otherwise a line will 
  #appear along the background border (artifact of the local maximum filter)
  eroded_background = binary_erosion(background, structure=neighborhood, border_value=1)

  #we obtain the final mask, containing only peaks, 
  #by removing the background from the local_max mask (xor operation)
  detected_peaks = local_max ^ eroded_background

  return detected_peaks

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

  stack_segmentation_function = lambda x,p : stack_segmentation.watershed_two_chan(x, **p)
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
  res['n_evals'] = 10 ## must be greater than 2 or hyperopt throws errors
  res['blur'] = False
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

def train(net, trainable, savepath, n_epochs=10, batchsize=3):
  xs_train = trainable['xs_train']
  ys_train = trainable['ys_train']
  ws_train = trainable['ws_train']
  xs_vali  = trainable['xs_vali']
  ys_vali  = trainable['ys_vali']
  ws_vali  = trainable['ws_vali']
  ysem = trainable['ysem']

  stepsperepoch   = xs_train.shape[0] // batchsize
  validationsteps = xs_vali.shape[0] // batchsize

  def print_stats():
    stats = dict()
    stats['batchsize'] = batchsize
    stats['n_epochs']  = n_epochs
    stats['stepsperepoch'] = stepsperepoch
    stats['validationsteps'] = validationsteps
    stats['n_samples_train'] = xs_train.shape[0]
    stats['n_samples_vali'] = xs_vali.shape[0]
    stats['xs_mean'] = xs_train.mean()
    print(stats)
  print_stats()

  current_weight_number = glob_and_parse_filename(str(savepath / "w???.h5"))
  if current_weight_number is None: current_weight_number = 0
  weightname = str(savepath / "w{:03d}.h5".format(current_weight_number + 1))
  checkpointer = ModelCheckpoint(filepath=weightname, verbose=0, save_best_only=True, save_weights_only=True)
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

  net.save_weights(str(savepath / "w{:03d}_final.h5".format(current_weight_number + 1)))
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
      slices = patchmaker.slices_grid(x.shape[:-1], (64,128,128), coverall=False)
      b = 20
      slices_all = patchmaker.make_triplets(slices, (b,b,b))
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

## plotting

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
  keys = history.history.keys()
  for k in keys:
    if 'val_'+k in keys:
      plot_hist_key(k)

## run-style functions requiring

def optimize_segmentation(pimg, rawdata, segparams, mypath_opt):
  lab  = rawdata['lab']
  inds = rawdata['inds_labeled_slices']
  ## recompute gt_slices! don't use gt_slices from rawdata
  gt_slices = rawdata['gt_slices']
  stack_segmentation_function = segparams['function']
  segmentation_space = segparams['space']
  segmentation_info = segparams['info']
  n_evals = segparams['n_evals']
  img_instseg = pimg #[[0]]

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
  inds = rawdata['inds_labeled_slices'] #[:,:-4] # use full
  gt_slices = rawdata['gt_slices']
  # gt_slices  = np.array([lab2instance(x) for x in lab[inds[0], inds[1]]])
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
  gt_slices = rawdata['gt_slices']
  imgsem = rawdata['imgsem']
  ch_nuc_img = imgsem['nuc']

  print("COMPARE SEGMENTATION AGAINST LABELED SLICES")
  seg_scores = compute_seg_on_slices(hyp, rawdata)
  print({'seg': seg_scores.mean(), 'std':seg_scores.std()}, file=open(savepath / 'SEG.txt','w'))
  
  print("CREATE DISPLAY GRID OF SEGMENTATION RESULTS")
  img_slices = img[inds[0], inds[1]]
  # gt_slices  = np.array([lab2instance(x) for x in lab[inds[0], inds[1]]])
  pre_slices = hyp[inds[0], inds[1]]
  # container = np.zeros((1600,1600,3))
  # slices = patchmaker.slices_heterostride((1600,1600),(400,400))
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
    # ipdb.set_trace()
    # container[slices[i]] = ax
    # res.append(ax.get_array())
    ax.set_axis_off()

  fig.set_size_inches(10, 10, forward=True)
  plt.savefig(savepath / 'seg_overlay.png',dpi=200, bbox_inches='tight')
  
  print("INSTANCE SEGMENTATION ANALYSIS COMPLETE")

def build_nhl(hyp, rawdata, savepath):
  print("GENERATE NHLS FROM HYP AND ANALYZE")
  img = rawdata['img']
  ch_nuc_img = imgsem['nuc']
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
Input/Output patchmaker size. (just make sure that it's divisible by 2**n_maxpool)
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

## Tue Jul 17 20:00:41 2018

make n_epochs and batchsize params

TODO:
- [ ] (linearly or isonet) upscale *before* training.


"""