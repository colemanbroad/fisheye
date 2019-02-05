from clbase import *
import clanalyze
from clanalyze import match_chan

## data loading. layer 1: top

def icp2raw(icp,targetfunc):
  "icp = img,cen name pairs"
  def f(xy):
    x,y = xy
    d1 = build_img(x)
    d2 = build_cen(y)
    d3 = targetfunc(d2)
    dat = {**d1,**d2,**d3}
    dat['weights'] = np.ones(d3['target'][...,0].shape)
    return dat
  return [f(xy) for xy in icp]

## layer 2

def build_img(imgname):
  img = imread(imgname)
  img = gputools.scale(img, (5.5,0.5,0.5))
  img = img[...,np.newaxis]
  img = norm_szyxc_per(img,(0,1,2),pc=[0.2,99.8])
  return {'source':img}

def build_cen(cenname):
  cen = imread(cenname)
  pts = np.array([n['centroid'] for n in nhl_tools.hyp2nhl(cen)]).astype(np.int)
  pts = np.floor((pts+0.5)*[5.5,0.5,0.5]).astype(np.int) 
  pts = pts - [5,0,0] ## adjustment to fix GT ?
  cen = gputools.scale(cen, (5.5,0.5,0.5))
  cen[...] = 0 #np.zeros(img.shape[:-1])
  cen[tuple(pts.T)] = 1
  ncells = pts.shape[0]
  dat = updatekeys(dict(),locals(),['cen','pts','ncells'])
  return dat

@DeprecationWarning
def cen2target_gauss(cendata,w=5):
  pts,cen = cendata['pts'], cendata['cen']
  target,ksum = soft_gauss_target(pts,cen.shape,w=w)
  weights = np.ones(target.shape)
  n_target_pts = pts.shape[0]
  target = target[...,None]
  dat = updatekeys(dict(),locals(),['target', 'n_target_pts', 'weights','ksum','w'])
  return dat

@DeprecationWarning
def cen2target_class(cendata,w=5):
  pts,cen = cendata['pts'], cendata['cen']
  # target1,ksum1 = hard_round_target(pts,cen.shape,w=6)
  target,ksum = hard_round_target(pts,cen.shape,w=w)
  weights = np.ones(target.shape)
  n_target_pts = label(target)[1]
  # weights = weights / weights.mean()
  target = np_utils.to_categorical(target).reshape(target.shape + (-1,))
  dat = updatekeys(dict(),locals(),['target', 'n_target_pts', 'weights','ksum','w'])
  return dat

## layer 3

def hard_round_target(cendata,w=3):
  pts,cen = cendata['pts'], cendata['cen']
  outshape = cen.shape
  k = 3*w
  d = np.indices((k,k,k))
  kern = ((d-k//2)**2).sum(0) < w**2
  target = sum_kernels_at_points(pts,kern,outshape)
  target[target>1] = 1
  target = np_utils.to_categorical(target).reshape(target.shape + (-1,))
  kernsum = kern.sum()
  dat = fromlocals(locals(),['kern','w','target','kernsum'])
  return dat

def soft_exp_target(cendata,w=3):
  pts,cen = cendata['pts'], cendata['cen']
  outshape = cen.shape
  k = 6*w
  d = np.indices((k,k,k))
  kern = np.exp(- 1/w * np.sqrt(((d-k//2)**2).sum(0)))
  target = sum_kernels_at_points(pts,kern,outshape)
  target = target[...,None]
  kernsum = kern.sum()
  dat = fromlocals(locals(),['kern','w','target','kernsum'])
  return dat 

def soft_gauss_target(cendata,w=6):
  pts,cen = cendata['pts'], cendata['cen']
  outshape = cen.shape
  sig = np.array([1,1,1])/w
  wid = w*6
  def f(x): return np.exp(-(sig*x*sig*x).sum()/2)
  kern = math_utils.build_kernel_nd(wid,3,f)
  target = sum_kernels_at_points(pts,kern,outshape)
  target = target[...,None]
  kernsum = kern.sum()
  dat = fromlocals(locals(),['kern','w','target','kernsum'])
  return dat

## layer 4

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



## callback stuff

class GlobalCallback(keras.callbacks.Callback):
  def __init__(self,callbackfunc):
    self.callbackfunc = callbackfunc

  def on_train_begin(self, logs={}):
    self.aucs = []
    self.losses = []

  def on_train_end(self, logs={}):
    return

  def on_epoch_begin(self, epoch, logs={}):
    return

  def on_epoch_end(self, epoch, logs={}):
    self.callbackfunc(self.model, epoch)
    return

  def on_batch_begin(self, batch, logs={}):
    return

  def on_batch_end(self, batch, logs={}):
    return

def build_callbackdata(rawdata):
  # ex_vali  = [(x[0],y[0,...,:-1]) for (x,y) in itertools.islice(vg, 5)]
  tg2 = datagen(rawdata['train'],patch_size=(120,120,120),batch_size=1)
  ex_train = [(x[0],y[0,...,:-1]) for (x,y) in itertools.islice(tg2, 5)]
  xs_train = np.array([x[0] for x in ex_train])
  ys_train = np.array([x[1] for x in ex_train])

  img_final = build_img(imgnames(1)[182])['source']
  cen_final = build_cen(cennames(1)[182])['cen']
  cen_final = np_utils.to_categorical(cen_final).reshape(cen_final.shape + (-1,))

  img_mid = build_img(imgnames(1)[95])['source']
  cen_mid = build_cen(cennames(1)[95])['cen']
  cen_mid = np_utils.to_categorical(cen_mid).reshape(cen_mid.shape + (-1,))

  img_zero = build_img(imgnames(1)[0])['source']
  cen_zero = build_cen(cennames(1)[0])['cen']
  cen_zero = np_utils.to_categorical(cen_zero).reshape(cen_zero.shape + (-1,))

  imgs = fromlocals(locals(),['img_final','cen_final','img_mid','cen_mid','img_zero','cen_zero'])
  scores = []
  scoresbig = []
  current_best = -1

  callbackdata = fromlocals(locals(),['xs_train','ys_train','scores','current_best','imgs','scoresbig'])
  return callbackdata

def cbd_to_cb(callbackdata, rawdata, funcs):
  cbd = callbackdata
  out_channels = rawdata['train'][0]['target'].shape[-1]
  def epochend_callback(net,epoch):

    funcs['f_custom'](net,epoch)

    ## predict on training examples
    yp_train = net.predict(cbd['xs_train'], batch_size=1)
    score_patch = funcs['f_match1'](cbd['ys_train'],yp_train)
    cbd['scores'].append(fromlocals(locals(), ['epoch','score_patch']))
    ## show them
    xs = cbd['xs_train'][...,rawdata['xrgb']]
    ys = cbd['ys_train'][...,rawdata['yrgb']]
    yp = yp_train[...,rawdata['yrgb']]
    res = clanalyze.plotlist([xs,ys,yp],1,c=1) #min(xs.shape[0],1))
    io.imsave(rawdata['epochdir'] / 'train_{:03d}.png'.format(epoch), res)
    print("cp1")
    ## predict on large images (final timepoint) and save weights
    imgs = cbd['imgs']      
    if epoch % 5 == 0:
      print("predict on final...")
      pimg_final  = clanalyze.predict(net, imgs['img_final'][None],out_channels)
      score_final = funcs['f_match2'](imgs['cen_final'][None], pimg_final)
      print("successfully computed final scores...")
      # pimg_mid    = clanalyze.predict(net, imgs['img_mid'][None],out_channels)
      # score_mid   = rawdata['f_match'](imgs['cen_mid'][None], pimg_mid)
      # pimg_zero   = clanalyze.predict(net, imgs['img_zero'][None],out_channels)
      # score_zero  = rawdata['f_match'](imgs['cen_zero'][None], pimg_zero)
      cbd['scoresbig'].append(fromlocals(locals(),['epoch','score_final']))

      if score_final > cbd['current_best']:
        modelname = rawdata['epochdir'] / 'w_match_{:.3f}_{:03d}.h5'.format(score_final, epoch)
        net.save_weights(str(modelname))
        cbd['current_best'] = score_final
        cbd['best_model'] = modelname

    plot_callback(cbd,rawdata['traindir'])

  return epochend_callback

def plot_callback(callbackdata,traindir):
  plt.figure()

  epochs = [d['epoch'] for d in callbackdata['scores']]
  scores = [d['score_patch'] for d in callbackdata['scores']]
  plt.plot(epochs, scores, label='patch')

  epochs = [d['epoch'] for d in callbackdata['scoresbig']]
  scores = [d['score_final'] for d in callbackdata['scoresbig']]
  plt.plot(epochs, scores, label='final')

  plt.legend()
  plt.savefig(traindir / 'callback_traj.pdf')

## network? what network?

def datagen(rawdata_train, patch_size=(48,48,48), batch_size=16):

  borders = np.array([20,20,20])
  padding = [(b,b) for b in borders] + [(0,0)]
  pad = lambda x: np.pad(x, padding, mode='constant')

  srcs = [pad(x['source']) for x in rawdata_train]
  trgs = [pad(x['target']) for x in rawdata_train]

  pad = lambda x: np.pad(x, padding[:-1], mode='constant')
  wghs = [pad(x['weights']) for x in rawdata_train]

  patchshape = np.array(patch_size)

  def patch():
    r = np.random.randint(0,len(srcs))
    src = srcs[r]
    trg = trgs[r]
    wgh = wghs[r]
    u_bound = np.array(src.shape[:-1]) - patchshape
    l_bound = [0,0,0]
    ind = np.array([np.random.randint(l_bound[i],u_bound[i]+1) for i in range(len("ZYX"))])
    ss = patchmaker.se2slices(ind,ind+patchshape)
    y = trg[tuple(ss)]
    w = wgh[tuple(ss)]
    x = src[tuple(ss)]
    # x = x / x.mean((0,1,2), keepdims=True)
    z = cat([y, w[...,np.newaxis]],-1)
    return (x,z)

  while True:
    xys = [patch() for _ in range(batch_size)]
    xs  = np.array([xy[0] for xy in xys])
    ys  = np.array([xy[1] for xy in xys])
    yield (xs,ys)

def build_net(shape_params):
  s_ = shape_params

  unet_params = {
    'n_pool' : 3,
    'n_convolutions_first_layer' : 32,
    'dropout_fraction' : 0.2,
    'kern_width' : 3,
  }

  # mul = 2**unet_params['n_pool']
  # faclist = [factors(x) for x in s_['inshape'][:-1]]
  # for fac in faclist: assert mul in fac

  input0 = Input(s_['inshape'])
  unet_out = unet.get_unet_n_pool(input0, **unet_params)
  output2  = unet.acti(unet_out, s_['out_channels'], last_activation=s_['activation'], name='B')
  net = Model(inputs=input0, outputs=output2)
  optim = Adam(lr=s_['lr'])

  eps = K.epsilon()

  if 'weights' not in s_.keys():
    weights = np.array([1,]*s_['out_channels']) / s_['out_channels']
    weights = K.variable(weights)
  else:
    weights = s_['weights']

  def crossentropy_w(yt,yp):
    ws = yt[...,-1]
    ws = ws[...,np.newaxis]
    yt = yt[...,:-1]
    ce = ws * yt * K.log(yp + eps) #/ K.log(2.0)
    ce = -K.mean(ce,(0,1,2))
    print(K.shape(ce))
    ce = K.sum(ce*weights)
    return ce

  if s_['task']=='regression':

    def loss(yt, yp):
      w  = yt[...,-1]
      yt = yt[...,:-1]
      # lo = losses.mean_squared_error(yt[...,0],yp[...,0])
      lo = losses.mean_absolute_error(yt[...,0],yp[...,0])
      # cellcount = (K.mean(yt) - K.mean(yp))**2
      # return mse # + 10.0 * cellcount
      # ce = losses.binary_crossentropy(yt[...,0],yp[...,0])
      return lo

    def met0(yt, yp):
      w  = yt[...,-1]
      yt = yt[...,:-1]
      # return K.std(yp)
      lo = losses.mean_squared_error(yt[...,0],yp[...,0])
      # lo = losses.mean_absolute_error(yt[...,0],yp[...,0])
      return lo

    def met1(yt,yp):
      w  = yt[...,-1]
      yt = yt[...,:-1]
      cellcount = K.mean(yt) - K.mean(yp)
      return cellcount

  elif s_['task']=='classification':
    def met0(yt,yp):
      return metrics.categorical_accuracy(yt[...,:-1], yp)
  
    loss = crossentropy_w

    def met1(yt,yp):
      w  = yt[...,-1]
      yt = yt[...,:-1]
      cellcount = K.mean(yt[...,1]) - K.mean(yp[...,1])
      return cellcount

  net.compile(optimizer=optim, loss={'B':loss}, metrics={'B':[met0,met1]})
  return net

## combine rawdata, callback data and net params

def traindata_class(savedir,f_rawbuild,w):
  # targetfunc = cen2target_class
  xrgb = [0,0,0]
  yrgb = [1,1,1]
  raw = f_rawbuild(lambda x: hard_round_target(x,w))
  updatekeys(raw,locals(),['xrgb','yrgb'])
  
  weights = np.array([1,1])
  weights[1] = 32.0
  weights = weights / weights.sum()
  weights = K.variable(weights)

  ## callback stuff
  if True:
    callbackdata = build_callbackdata(raw)

    savedir.mkdir(exist_ok=True, parents=True)
    traindir = savedir  / 'train_cp/'; traindir.mkdir(exist_ok=True);
    epochdir = traindir / 'epochs/'; epochdir.mkdir(exist_ok=True);

    updatekeys(raw,locals(),['savedir', 'traindir','epochdir'])

    f_match1 = lambda x,y: match_chan(x,y,(1,1))
    f_match2 = lambda x,y: match_chan(x,y,(1,1))

    weight_decay = [2,1]

    def f_custom(net,epoch):
      w = K.get_value(weights)
      if (epoch % 2 == 0) and epoch > 0 and w[1]>(1/len(w)):
        w = w * weight_decay
        w = w / w.sum()
        print(w,w.shape)
        K.set_value(weights,w)

    funcs = fromlocals(locals(),['f_custom','f_match1','f_match2'])
    epochend_callback = cbd_to_cb(callbackdata,raw,funcs)
    callback = GlobalCallback(epochend_callback)

  out_channels = raw['train'][0]['target'].shape[-1]
  netparams = {'inshape':(None,None,None,1), 'out_channels':out_channels, 'activation':'softmax', 'task':'classification', 'weights':weights, 'lr':1e-4}
  raw['netparams'] = netparams

  data = fromlocals(locals(), ['raw','callback','callbackdata'])

  return data

def traindata_soft(savedir,f_rawbuild, targetfunc):
  xrgb = [0,0,0]
  yrgb = [0,0,0]
  raw = f_rawbuild(targetfunc)
  updatekeys(raw,locals(),['xrgb','yrgb'])
  
  ## callback stuff
  if True:
    callbackdata = build_callbackdata(raw)

    savedir.mkdir(exist_ok=True, parents=True)
    traindir = savedir  / 'train_cp/'; traindir.mkdir(exist_ok=True);
    epochdir = traindir / 'epochs/'; epochdir.mkdir(exist_ok=True);

    updatekeys(raw,locals(),['savedir', 'traindir','epochdir'])

    f_match1 = lambda x,y: match_chan(x,y,(0,0))
    f_match2 = lambda x,y: match_chan(x,y,(1,0))

    def f_custom(net,epoch):
      pass

    funcs = fromlocals(locals(),['f_custom','f_match1','f_match2'])
    epochend_callback = cbd_to_cb(callbackdata,raw,funcs)
    callback = GlobalCallback(epochend_callback)

  out_channels = raw['train'][0]['target'].shape[-1]
  netparams = {'inshape':(None,None,None,1), 'out_channels':out_channels, 'activation':'linear', 'task':'regression', 'lr':1e-4}
  raw['netparams'] = netparams

  data = fromlocals(locals(), ['raw','callback','callbackdata'])

  return data

## which timepoints / datasets to use ?

def bugfixraw(targetfunc):
  times    = [130]
  ns       = [1]
  basedirs = ['Fluo-N3DH-CE']*len(ns)
  
  icp = imgcenpairs({'times':times,'ns':ns,'bases':basedirs})
  raw_train = icp2raw(icp,targetfunc)
  tar = raw_train[0]['target']
  raw_train[0]['source'] = tar + 0.1 * np.random.rand(*tar.shape)

  times    = [100]
  ns       = [1]
  basedirs = ['Fluo-N3DH-CE']*len(ns)
  
  icp = imgcenpairs({'times':times,'ns':ns,'bases':basedirs})
  raw_vali = icp2raw(icp,targetfunc)
  tar = raw_vali[0]['target']
  raw_vali[0]['source'] = tar + 0.1 * np.random.rand(*tar.shape)

  data = {'train':raw_train,'vali':raw_vali}
  return data

def simpleraw(targetfunc):
  times    = [130]
  ns       = [1]
  basedirs = ['Fluo-N3DH-CE']*len(ns)
  
  icp = imgcenpairs({'times':times,'ns':ns,'bases':basedirs})
  raw_train = icp2raw(icp, targetfunc)

  times    = [100]
  ns       = [1]
  basedirs = ['Fluo-N3DH-CE']*len(ns)
  
  icp = imgcenpairs({'times':times,'ns':ns,'bases':basedirs})
  raw_vali = icp2raw(icp, targetfunc)

  data = {'train':raw_train,'vali':raw_vali}
  return data

def buildraw(targetfunc):
  times    = [0,100,150,175,189,189]
  ns       = [2,1,2,1,2,1]
  basedirs = ['Fluo-N3DH-CE']*len(ns)
  
  icp = imgcenpairs({'times':times,'ns':ns,'bases':basedirs})
  raw_train = icp2raw(icp, targetfunc)

  times    = [70,185,185]
  ns       = [1,1,2]
  basedirs = ['Fluo-N3DH-CE']*len(ns)
  
  icp = imgcenpairs({'times':times,'ns':ns,'bases':basedirs})
  raw_vali = icp2raw(icp, targetfunc)

  data = {'train':raw_train,'vali':raw_vali}
  return data

## train new and existing models

def train_model(traindata):
  raw = traindata['raw']

  net = build_net(raw['netparams'])
  params0 = traindata['raw'].get('params0',None)
  if params0:
    net.load_weights(params0)

  tg = datagen(raw['train'],patch_size=(80,80,80),batch_size=4)
  vg = datagen(raw['vali'],patch_size=(120,120,120),batch_size=1)

  callbackdata = traindata['callbackdata']
  callbackdata['scores'] = []
  callbackdata['scoresbig'] = []
  history = ts.train_gen(net, tg, vg, raw['traindir'], n_epochs=300, steps_per_epoch=20, vali_steps=10, callbacks=[traindata['callback']])

  ts.plot_history(history, start=1, savepath=raw['traindir'])
  pickle.dump(callbackdata, open(str(raw['traindir'] / 'callbackdata.pkl'),'wb'))
  plot_callback(callbackdata, raw['traindir'])
  if callbackdata.get('best_model',None) is not None:
    net.load_weights(str(callbackdata['best_model']))

  res = fromlocals(locals(),['net','history','tg','vg'])
  return res

def continuetrain(traindata, res):
  net = res['net']
  tg  = res['tg']
  vg  = res['vg']
  traindir = traindata['raw']['traindir']
  callbackdata = traindata['callbackdata']

  callback = traindata['callback']
  history  = ts.train_gen(net, tg, vg, traindir, n_epochs=4, steps_per_epoch=30, vali_steps=10, callbacks=[callback])
  
  oldhist = res['history']
  for k in ['loss','val_loss','met0','val_met0','met1','val_met1']:
    oldhist[k].extend(history[k])

  ts.plot_history(history, start=1, savepath=traindir)
  plot_callback(callbackdata,traindata['raw']['traindir'])
  pickle.dump(callbackdata, open(str(raw['traindir'] / 'callbackdata.pkl'),'wb'))
  net.load_weights(str(callbackdata['best_model']))