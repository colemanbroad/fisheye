from clbase import *
import clanalyze

## layer 1, top

def icp2raw(icp,cen2target):
  "icp = img,cen name pairs"
  def f(xy):
    x,y = xy
    d1 = build_img(x)
    d2 = build_cen(y)
    d3 = cen2target(d2)
    return {**d1,**d2,**d3}
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

def cen2target_gauss(cendata,w=5):
  pts,cen = cendata['pts'], cendata['cen']
  target,ksum = soft_gauss_target(pts,cen.shape,w=w)
  weights = np.ones(target.shape)
  n_target_pts = pts.shape[0]
  target = target[...,None]
  dat = updatekeys(dict(),locals(),['target', 'n_target_pts', 'weights','ksum','w'])
  return dat

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

def hard_round_target(pts,outshape,w=3):
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

## callbacks n' stuff

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
      mse = losses.mean_squared_error(yt[...,0],yp[...,0])
      cellcount = (K.mean(yt) - K.mean(yp))**2
      # return mse # + 10.0 * cellcount
      # ce = losses.binary_crossentropy(yt[...,0],yp[...,0])
      return mse

    def met0(yt, yp):
      w  = yt[...,-1]
      yt = yt[...,:-1]
      # return K.std(yp)
      mae = losses.mean_absolute_error(yt[...,0],yp[...,0])
      return mae

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
