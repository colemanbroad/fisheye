from segtools.defaults.ipython_local import *
# from ipython_remote_defaults import *
from segtools.defaults.training import *
import ipdb

def channel_semantics():
  d = dict()
  d['mem'] = 1
  d['nuc'] = 0
  d['n_channels'] = 2
  ## define input channels
  dz = 2
  d['dz'] = dz
  d['n_channels_xs'] = d['n_channels']*(1+2*dz)
  d['ch_mem_xs'] = 2*dz # original, centered membrane channelk
  d['ch_nuc_xs'] = 2*dz + 1
  d['ch_mem_img'] = 0
  d['ch_nuc_img'] = 1
  d['rgb.img'] = [d['ch_mem_img'], d['ch_nuc_img'], d['ch_nuc_img']]
  d['rgb.xs']  = [d['ch_mem_xs'], d['ch_nuc_xs'], d['ch_nuc_xs']]
  return d

def class_semantics():
  d = dict()
  d['n_classes'] = 3
  d['div'] =  1
  d['nuc'] =  1
  d['mem'] =  0
  d['bg'] =  2
  d['ignore'] =  1
  d['rgb.mem'] =  [d['mem'], d['nuc'], d['bg']]
  d['rgb.div'] =  [d['div'], d['mem'], d['div']]
  return d

def load_rawdata(homedir):
  homedir = Path(homedir)

  chansem = channel_semantics()
  cs = class_semantics()

  def condense_labels(lab):
    d = cs
    lab[lab==0]   = d['bg']
    lab[lab==255] = d['mem']
    lab[lab==168] = d['nuc']
    lab[lab==85]  = d['div']
    lab[lab==198] = d['ignore']
    return lab

  ## load data
  # img = imread(str(homedir / 'data/img006.tif'))
  img = np.load(str(homedir / 'data/img006_noconv.npy'))
  # img = np.load(str(homedir / 'isonet/restored.npy'))
  # img = img[np.newaxis,:352,...]
  # lab = np.load(str(homedir / 'data/labels_iso_t0.npy'))
  lab = imread(str(homedir / 'data/labels_lut.tif'))
  lab = lab[:,:,0]
  # lab = lab[np.newaxis,:352,...]

  lab = condense_labels(lab)
  ## TODO: this will break once we start labeling XZ and YZ in same volume.
  mask_labeled_slices = lab.min((2,3)) < 2
  inds_labeled_slices = np.indices(mask_labeled_slices.shape)[:,mask_labeled_slices]
  def lab2instance(x):
    d = cs
    x[x!=d['nuc']] = 0
    x = label(x)[0]
    return x
  gt_slices = np.array([lab2instance(x) for x in lab[inds_labeled_slices[0], inds_labeled_slices[1]]])

  res = dict()
  res['img'] = img
  res['lab'] = lab
  res['mask_labeled_slices'] = mask_labeled_slices
  res['inds_labeled_slices'] = inds_labeled_slices
  res['gt_slices'] = gt_slices
  return res

def build_trainable(rawdata):
  img = rawdata['img']
  lab = rawdata['lab']
  mask_labeled_slices = rawdata['mask_labeled_slices']
  inds_labeled_slices = rawdata['inds_labeled_slices']

  border_weights = True

  chansem = channel_semantics()
  cs = class_semantics()
  
  ys = lab[mask_labeled_slices].copy()
  xs = []
  for t,z in inds_labeled_slices.T:
    res = unet.add_z_to_chan(img[t], chansem['dz'], ind=z)
    xs.append(res[0])
  xs = np.array(xs)
  xs = perm(xs, "scyx", "syxc")

  # convert ys to categorical
  ys = np_utils.to_categorical(ys).reshape(ys.shape + (-1,))

  # add distance channel?
  if False:
    if dist_channel:
      a,b,c,d = ys.shape
      mask = ys[...,cs['mem']]==1
      distimg = ~mask
      distimg = np.array([distance_transform_edt(d) for d in distimg])
      # distimg = np.exp(-distimg/10)
      mask = ys[...,cs['bg']]==1
      distimg[mask] *= -1
      # distimg = distimg/distimg.mean((1,2), keepdims=True)
      ys = distimg[...,np.newaxis]
      # ys = ys / ys.mean((1,2,3), keepdims=True)

  # split 400x400 into 16x100x100 patches (if r=4) and reshape back into "SYXC"
  # axes are "SYXC"
  r = 2
  xs = collapse(splt(xs,[r,r],[1,2]), [[0,1,3],[2],[4],[5]])
  ys = collapse(splt(ys,[r,r],[1,2]), [[0,1,3],[2],[4],[5]])

  # np.savez(savepath / 'traindat_small', xs=xs[::5], ys=ys[::5])

  ## turn off the `ignore` class
  # mask = ys[...,3]==1
  # ys[mask] = 0

  ## reweight border pixels
  n_sample, n_y, n_x, _ = xs.shape
  ws = np.ones(xs.shape[:-1])
  ws = ws / ws.sum((1,2), keepdims=True)
  if border_weights:
    a,b,c,d = ys.shape
    mask = ys[...,cs['mem']]==1 # ????
    distimg = ~mask
    distimg = np.array([distance_transform_edt(d) for d in distimg])
    distimg = np.exp(-distimg/10)
    ws = distimg/distimg.sum((1,2), keepdims=True)

  classweights = [1/cs['n_classes'],]*cs['n_classes']
  print(xs.shape, ys.shape, mask_labeled_slices.shape)
  print(ys.max((0,1,2)))

  ## normalize
  xs = xs/xs.mean((1,2), keepdims=True)

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
  
  unet_params = {
    'n_pool' : 3,
    'inputchan' : chansem['n_channels_xs'],
    'n_classes' : cs['n_classes'],
    'n_convolutions_first_layer' : 64,
    'dropout_fraction' : 0.2,
    'kern_width' : 3,
  }

  net = unet.get_unet_n_pool(**unet_params)

  # loss  = unet.my_categorical_crossentropy(weights=classweights, itd=0)

  res = dict()
  res['xs_train'] = xs_train
  res['xs_vali'] = xs_vali
  res['ys_train'] = ys_train
  res['ys_vali'] = ys_vali
  res['ws_train'] = ws_train
  res['ws_vali'] = ws_vali
  res['classweights'] = classweights

  res['net'] = net
  # res['loss'] = loss

  return res

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

def train(trainable, savepath):
  xs_train = trainable['xs_train']
  ys_train = trainable['ys_train']
  ws_train = trainable['ws_train']
  xs_vali = trainable['xs_vali']
  ys_vali = trainable['ys_vali']
  ws_vali = trainable['ws_vali']
  classweights = trainable['classweights']
  net = trainable['net']

  batchsize = 3
  n_epochs  = 50

  stepsperepoch   = xs_train.shape[0] // batchsize
  validationsteps = xs_vali.shape[0] // batchsize

  optim = Adam(lr=1e-4)
  loss  = unet.my_categorical_crossentropy(weights=classweights, itd=0)
  net.compile(optimizer=optim, loss=loss, metrics=['accuracy'])
  checkpointer = ModelCheckpoint(filepath=str(savepath / "w001.h5"), verbose=0, save_best_only=True, save_weights_only=True)
  earlystopper = EarlyStopping(patience=30, verbose=0)
  reduce_lr    = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
  callbacks = [checkpointer, earlystopper, reduce_lr]

  f_trainaug = lambda x,y:random_augmentation_xy(x,y,train=True)

  bgen = unet.batch_generator_patches_aug(xs_train, ys_train,
                                    # steps_per_epoch=100,
                                    batch_size=batchsize,
                                    augment_and_norm=f_trainaug,
                                    savepath=savepath)

  f_valiaug = lambda x,y:random_augmentation_xy(x,y,train=False)

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

  net.save_weights(savepath / 'w002.h5')
  return history

def show_trainvali(trainable):
  xs_train = trainable['xs_train']
  xs_vali = trainable['xs_vali']
  ys_train = trainable['ys_train']
  ys_vali = trainable['ys_vali']

  d = channel_semantics()
  c = class_semantics()

  sx = [slice(None,5), Ellipsis, d['rgbxs']]
  sy = [slice(None,5), Ellipsis, c['rgb.mem']]
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
  # io.imsave(savepath / 'train_vali_ex.png', res)
  return res

def plot_history(history, savepath):
  print(history.history, file=open(savepath / 'history.txt','w'))
  def plot_hist_key(k):
    plt.figure()
    y1 = history.history[k]
    y2 = history.history['val_' + k]
    plt.plot(y1, label=k)
    plt.plot(y2, label='val_' + k)
    plt.legend()
    plt.savefig(savepath / (k + '.png'))
  plot_hist_key('loss')
  plot_hist_key('acc')

def run_training(homedir, savepath):
  rawdata = load_rawdata(homedir)
  trainable = build_trainable(rawdata)
  net = trainable['net']
  res = show_trainvali(trainable)
  io.imsave(savepath / 'train_vali_ex.png', res)
  history = train(trainable, savepath)
  plot_history(history, savepath)
  ## load best weights from previous training
  net.load_weights(loadpath / 'w001.h5')
  # loadpath = savepath
  print("TRAIN COMPLETE")

def predict_on_new(net,img,dz):
  assert img.ndim == 5 ## TZYXC
  pimg = []
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
    pimg.append(timepoint)
  pimg = np.array(pimg)
  return pimg

def predict_train_vali(trainable, savepath=None):
  net = trainable['net']
  xs_train = trainable['xs_train']
  xs_vali = trainable['xs_vali']
  rgbmem = class_semantics()['rgb.mem']
  pred_xs_train = net.predict(xs_train)
  rows, cols = rowscols(pred_xs_train.shape[0], 8)
  res1 = collapse(splt(pred_xs_train[:cols*rows,...,rgbmem],rows,0), [[0,2],[1,3],[4]])
  if savepath:
    io.imsave(savepath / 'pred_xs_train.png', res1)
  pred_xs_vali = net.predict(xs_vali)
  rows, cols = rowscols(pred_xs_vali.shape[0], 5)
  res2 = collapse(splt(pred_xs_vali[:cols*rows,...,rgbmem],rows,0), [[0,2],[1,3],[4]])
  if savepath:
    io.imsave(savepath / 'pred_xs_vali.png', res2)
  return res1, res2

def run_predict_on_new(loaddir, savedir):
  unet_params = json.load(open(loaddir / 'unet_params.json', 'r'))
  net = unet.get_unet_n_pool(**unet_params)
  net.load_weights(str(loaddir / 'w001.h5'))
  img = np.load('data/img006_noconv.npy')
  dz = channel_semantics()['dz']
  pimg = predict_on_new(net, img, dz)
  np.save(savedir / 'pimg', pimg)
  loaddir = savedir
  print("PREDICT COMPLETE")

def showresults(pimg, rawdata):
  img = rawdata['img']
  lab = rawdata['lab']
  inds = rawdata['inds_labeled_slices']
  rgbimg = channel_semantics()['rgb.img']
  rgbmem = class_semantics()['rgb.mem']

  x = img[inds[0], inds[1]].transpose((0,2,3,1))[...,rgbimg]
  x[...,2] = 0 # remove blue
  y = lab[inds[0], inds[1]]
  y = np_utils.to_categorical(y).reshape(y.shape + (-1,))
  y = y[...,rgbmem]
  z = pimg[inds[0], inds[1]][...,rgbmem]
  ss = [slice(None,None,5), slice(None,None,4), slice(None,None,4), slice(None)]
  def f(r):
    r = merg(r[ss])
    r = r / np.percentile(r, 99, axis=(0,1), keepdims=True)
    r = np.clip(r,0,1)
    return r
  x,y,z = f(x), f(y), f(z)
  res = np.concatenate([x,y,z], axis=1)
  return res

def find_divisions(pimg):
  ch_div = class_semantics()['div']
  rgbdiv = class_semantics()['rgb.div']
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
  return x2

def run_pimg_analysis(homedir, loaddir, savedir):
  pimg = np.load(loaddir / 'pimg.npy')
  # homedir = './'
  rawdata = load_rawdata(homedir)
  res = showresults(pimg, rawdata)
  io.imsave(savedir / 'results.png', res)
  ## max division channel across z
  ch_div = class_semantics()['div']
  io.imsave(savedir / 'max_z_divchan.png', merg(pimg[:,...,ch_div].max(1),0))
  if pimg.shape[0] > 1:
    res = find_divisions(pimg)
    io.imsave(savedir / 'find_divisions.png', res)
  # loaddir = savedir
  pimg = np.save(savedir / 'pimg.npy', pimg)
  print("PIMG ANALYSIS COMPLETE")



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
Input/Output patch size. (just make sure that it's divisible by 2**n_maxpool)
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
  - 

"""