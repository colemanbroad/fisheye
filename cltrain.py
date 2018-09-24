from clbase import *
import clanalyze

## layer 1, top

def icp2raw(icp,w=4):
  "icp = img,cen name pairs"
  def f(xy):
    x,y = xy
    d1 = build_img(x)
    d2 = build_cen(y)
    d3 = cen2target(d2,w=w)
    return {**d1,**d2,**d3}
  return [f(xy) for xy in icp]

## layer 2

def build_img(imgname):
  img = imread(imgname)
  img = gputools.scale(img, (5.5,0.5,0.5))
  img = img[...,np.newaxis]
  return {'source':img}

def build_cen(cenname):
  dat = dict()
  cen = imread(cenname)
  pts = np.array([n['centroid'] for n in nhl_tools.hyp2nhl(cen)]).astype(np.int)
  pts = np.floor((pts+0.5)*[5.5,0.5,0.5]).astype(np.int) 
  pts = pts - [5,0,0] ## adjustment to fix GT ?
  cen = gputools.scale(cen, (5.5,0.5,0.5))
  cen[...] = 0 #np.zeros(img.shape[:-1])
  cen[tuple(pts.T)] = 1
  ncells = pts.shape[0]
  updatekeys(dat,locals(),['cen','pts','ncells'])
  return dat

def cen2target(cendata,w):
  dat = dict()
  pts,cen = cendata['pts'], cendata['cen']
  target,ksum = make_hard_round_target(pts,cen.shape,w=w)
  n_target_pts = label(target)[1]
  weights = np.ones(target.shape)
  cm=1
  weights[target==1] *= cm
  weights = weights / weights.mean()
  target = np_utils.to_categorical(target).reshape(target.shape + (2,))
  updatekeys(dat,locals(),['cm', 'target', 'n_target_pts', 'weights','ksum','w'])
  return dat

## layer 3

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

## layer 0

def datagen(rawdata_train):

  borders = np.array([20,20,20])
  padding = [(b,b) for b in borders] + [(0,0)]
  pad = lambda x: np.pad(x, padding, mode='constant')

  srcs = [pad(x['source']) for x in rawdata_train]
  trgs = [pad(x['target']) for x in rawdata_train]

  pad = lambda x: np.pad(x, padding[:-1], mode='constant')
  wghs = [pad(x['weights']) for x in rawdata_train]

  patchshape = np.array([120,120,120])

  while True:
    r = random.randint(0,len(srcs)-1)
    src = srcs[r]
    trg = trgs[r]
    wgh = wghs[r]
    u_bound = np.array(src.shape[:-1]) - patchshape
    l_bound = [0,0,0]
    ind = np.array([random.randint(l_bound[i],u_bound[i]) for i in range(len("ZYX"))])
    ss = patchmaker.se2slices(ind,ind+patchshape)
    y = trg[tuple(ss)]
    w = wgh[tuple(ss)]
    x = src[tuple(ss)]
    x = x / x.mean((0,1,2), keepdims=True)
    z = cat([y, w[...,np.newaxis]],-1)
    x = x[None,...]
    z = z[None,...]
    yield (x, z)


class Histories(keras.callbacks.Callback):
  def __init__(self,examples,savedir,weights=None):
    self.examples = examples
    # self.ys = mini_ys
    self.savedir = savedir
    self.weights = None
    if weights is not None:
      self.weights = weights
    self.matching_score = []
    self.bestmodel = None

  def on_train_begin(self, logs={}):
      self.aucs = []
      self.losses = []

  def on_train_end(self, logs={}):
      print("SCORES")
      print(self.matching_score)
      pickle.dump(self.matching_score, open(self.savedir / 'matchingtraj.pkl', 'wb'))
      return

  def on_epoch_begin(self, epoch, logs={}):
      # print("Weights are: ", K.get_value(self.weights))
      return

  def on_epoch_end(self, epoch, logs={}):
      # self.losses.append(logs.get('loss'))
      # y_pred = self.model.predict(self.model.validation_data[0])
      xrgb = [0,0,0]
      yrgb = [1,1,1]
      xs = np.array([x[0] for x in self.examples])
      ys = np.array([x[1] for x in self.examples])
      yp = self.model.predict(xs, batch_size=1)
      
      ns = np.array([clanalyze.match_ratio_from_ytyp(y0,y1) for y0,y1 in zip(ys,yp)])
      ns = ns.T
      print("Match Ratio is: ", 2*ns[0]/(ns[1]+ns[2]))
      score = 2*ns[0].sum()/(ns[1:3].sum())
      print("Avg Score: ", score)
      print("Avg distance", ns[3])
      if len(self.matching_score)==0 or score > max(self.matching_score):
        modelname = self.savedir / 'w_match_{:.3f}_{:03d}.h5'.format(score, epoch)
        self.model.save_weights(str(modelname))
        self.bestmodel = modelname
      self.matching_score.append(score)

      ## show them
      xs = xs[...,xrgb]
      ys = ys[...,yrgb]
      yp = yp[...,yrgb]
      res = clanalyze.plotlist([xs,ys,yp],1,c=1) #min(xs.shape[0],1))
      io.imsave(self.savedir / 'e_{:03d}.png'.format(epoch), res)

      # self.aucs.append(roc_auc_score(self.model.validation_data[1], y_pred))
      
      w = K.get_value(self.weights)
      print(w, w.shape)

      if (epoch % 1 == 0) and epoch > 0 and w[1]>0.5:
        w = w * [1, 0.75]
        w = w / w.sum()
        print("WEIGHT CHANGE")
        print(w,w.shape)
        K.set_value(self.weights,w)

      return

  def on_batch_begin(self, batch, logs={}):
      return

  def on_batch_end(self, batch, logs={}):
      return

def build_net(inshape, out_channels, activation='softmax', weights=None):

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
  # unet_out = unet.get_unet_n_pool_recep(input0, **unet_params)
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
  
    eps = K.epsilon()
    if weights is None:
      weights = np.array([1,]*out_channels) / out_channels
      weights = K.variable(weights)

    def crossentropy_w(yt,yp):
      ws = yt[...,-1]
      ws = ws[...,np.newaxis]
      yt = yt[...,:-1]
      ce = ws * yt * K.log(yp + eps)
      ce = -K.mean(ce,(0,1,2))
      # K.tf.print(ce,[ce])
      print(K.shape(ce))
      ce = K.sum(ce*weights)
      return ce

    loss = crossentropy_w

    def met1(yt,yp):
      w  = yt[...,-1]
      yt = yt[...,:-1]
      cellcount = (K.mean(yt[...,1]) - K.mean(yp[...,1]))**2
      return cellcount

  net.compile(optimizer=optim, loss={'B':loss}, metrics={'B':[met0,met1]})
  return net
