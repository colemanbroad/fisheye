from segtools.defaults.ipython_remote import *
import cltrain as m
import clanalyze
from clbase import *
# import clipy as c


def do_it_all_classifier():
  pass

def load_old_and_reanalyze():
  # raw = buildraw()
  # step1()
  step2()

def step1():
  traindir = Path('training/ce_test5/train_cp/')
  net = net = m.build_net((120,120,120,1), 2, activation='softmax')
  net.load_weights('training/ce_test5/train_cp/epochs/w_match_0.952_097.h5')
  ## do em all
  clanalyze.save_pimgs2(net,traindir / "t1",imgnames(1)[::20])
  clanalyze.save_pimgs2(net,traindir / "t2",imgnames(2)[::20])
  clanalyze.save_pimgs2(net,traindir / "c1",challnames(1)[::20])
  clanalyze.save_pimgs2(net,traindir / "c2",challnames(2)[::20])

def step2():
  traindir = Path('training/ce_test5/train_cp/')
  names1 = imgnames(1)[::20]
  names2 = sorted(glob(str(traindir / "t1" / "*.tif")))
  xy = list(zip(names1, names2))
  png = pallettes_from_pimgs(xy)
  io.imsave(str(traindir / 't1.png'),png)

  names1 = imgnames(2)[::20]
  names2 = sorted(glob(str(traindir / "t2" / "*.tif")))
  xy = list(zip(names1, names2))
  png = pallettes_from_pimgs(xy)
  io.imsave(str(traindir / 't2.png'),png)

  names1 = challnames(1)[::20]
  names2 = sorted(glob(str(traindir / "c1" / "*.tif")))
  xy = list(zip(names1, names2))
  png = pallettes_from_pimgs(xy)
  io.imsave(str(traindir / 'c1.png'),png)

  names1 = challnames(2)[::20]
  names2 = sorted(glob(str(traindir / "c2" / "*.tif")))
  xy = list(zip(names1, names2))
  png = pallettes_from_pimgs(xy)
  io.imsave(str(traindir / 'c2.png'),png)

  # pallettes_from_net(datasets['t1'],net,traindir)
  # pallettes_from_net(datasets['t2'],net,traindir)
  # pallettes_from_net(datasets['c1'],net,traindir)
  # pallettes_from_net(datasets['c2'],net,traindir)

def trainkernels():
  bt = base_train
  data1 = {'times':[100,120,150,180,190], 'ns':[1,2,1,2,1,], 'bases':[bt]*5,}
  icp = imgcenpairs(data1)
  print(icp)
  raw = m.icp2raw(icp,w=4)
  raw_train = [raw[i] for i in [0,1,3,4]]
  raw_vali  = [raw[2]]
  tg = m.datagen(raw_train)
  vg = m.datagen(raw_vali)
  print(next(vg))

def fix189():
  net = build_net((120,120,120,1),2,activation='softmax')
  net.load_weights('training/ce_059/train_cp/epochs/w_match_0.935_132.h5')
  dat = data()
  pimg = predict(net,dat['img'][None,...,None],outchan=2)
  return pimg

def jointotsu():
  space = segmentation.segment_otsu_paramspace()
  optimize_seg_joint(space,'training/ce_test6')

def data():
  img = imread('Fluo-N3DH-CE_challenge/02/t189.tif')
  img = gputools.scale(img, [5.5,.5,.5])
  hyp = imread('training/ce_059/train_cp/pimgs/chall2/watershed/hyp015.tif')
  return updatekeys(dict(),locals(),['img','hyp'])

def tryotsu(data):
  img=data['img']
  print(img.shape)
  img = norm_szyxc(img,(0,1,2))
  segmentation.otsu3d(img)

def allpallettes_scaled():
  segpallette_scaled(datasets['t1'])
  segpallette_scaled(datasets['t2'])
  segpallette_scaled(datasets['c1'])
  segpallette_scaled(datasets['c2'])

def segpallette_scaled(p):
  ts = np.linspace(0,p['n']-1,11).astype(np.int)
  zs = np.linspace(5,p['z']-5,11).astype(np.int) ## ignore five slices at beginning and end. they're just noise.
  a,b,c = p['sh'][1:] + (3,)
  png = np.zeros((len(ts),len(zs),a,b,c))

  params = segmentation.segment_pimg_img_params
  for ti,t in enumerate(ts):
    img = imread(imgnames(p['dind'],base=p['base'])[t])
    img = gputools.scale(img, [5.5,0.5,0.5])
    img = img[...,None]
    pimg = imread("training/ce_059/train_cp/pimgs/{}/pimg_{:03d}.tif".format(p['dset'],t))
    hyp = segmentation.segment_pimg_img({"img":img,"pimg":pimg}, params)
    print(hyp.shape)
    for zi,z in enumerate(zs):
      # ipdb.set_trace()
      # _,mi,ma = norm_szyxc(img[z,...,0],(0,1),return_pc=True)
      png[ti,zi] = seg_compare2gt_rgb(hyp[z], img[z,...,0]) #*(ma-mi) + mi

  png = collapse2(png,"tzyxc","ty,zx,c")
  png = norm_szyxc(png,(0,1,2))
  io.imsave("training/figures/m59seg_{}_{}.png".format(p['dset'],'066'), png)

def shapet():
  def crop(data, scale = (5.5,.5,.5)):
      newshape = _scale_shape(_scale_shape(dshape,scale),tuple(1./s for s in scale))
      return pad_to_shape(data, newshape)
  cropped_data = crop(data, scale = (5.5,.5,.5))

def runny():
  # hyps2movie(hypdir,imgdir,savedir,times)
  img = imread('training/ce_059/train_cp/pimgs/chall1/watershed/hyp015.tif')
  qsave(img)
  print(img.shape)

def scale_all():
  for c in ['c2']:
    p = datasets[c]
    pimgdir = Path('training/ce_059/train_cp/pimgs/{}/'.format(p['dset']))
    img = imread(imgnames(p['dind'],base=p['base'])[0])
    savedir = pimgdir / 'watershed'; savedir.mkdir(exist_ok=True)
    times = range(0,p['n'])
    pimgs2hyps(pimgdir,p,savedir,times)

def testsize():
  p = datasets['c1']
  img = imread(imgnames(p['dind'],base=p['base'])[0])
  print(img.shape)
  p = datasets['c2']
  img = imread(imgnames(p['dind'],base=p['base'])[0])
  print(img.shape)

def pltprofiles():
  """
  does the mean intensity changes drastically after nlm and blurring? no it doesn't.
  ok, let's try the max? or 99%?...
  after running hyperopt it's clear that the optimal function is mean for every z slice.
  """
  def plotimg(img): 
    def f(img): return np.percentile(img,99,axis=(1,2))
    def g(img): return np.percentile(img,50,axis=(1,2))
    def h(img): return np.mean(img,axis=(1,2))
    def i(img): return np.mean(img)
    # plt.plot(f(img))
    plt.plot(g(img))
    plt.plot(h(img))
    plt.axhline(i(img))

  params = bestparams
  plt.figure()
  img = imread(imgnames(1)[100])
  img = gputools.scale(img, [5.5,.5,.5])
  plotimg(img)
  hx = np.array([1,1,1]) / 3
  img2 = gputools.denoise.nlm3(img, sigma=params['sigma'])
  # img3 = img2 / 
  for _ in range(params['nblur']):
    img2 = gputools.convolve_sep3(img2,hx,hx,hx)
  plotimg(img)
  plt.savefig('training/figures/profile.png')

def allpallettes():
  segpallette(datasets['t1'])
  segpallette(datasets['t2'])
  segpallette(datasets['c1'])
  segpallette(datasets['c2'])

def segpallette(p):
  ts = np.linspace(0,p['n']-1,11).astype(np.int)
  zs = np.linspace(5,p['z']-5,11).astype(np.int) ## ignore five slices at beginning and end. they're just noise.
  a,b,c = p['sh'][1:] + (3,)
  png = np.zeros((len(ts),len(zs),a,b,c))

  # def f1(img): return np.percentile(img,99,axis=(1,2),keepdims=True)
  # def f2(img): return np.percentile(img,50,axis=(1,2),keepdims=True)
  def f3(img): return np.mean(img,axis=(1,2),keepdims=True)
  # def f4(img): return np.mean(img)

  params = segment_pimg_img_params
  for ti,t in enumerate(ts):
    img = imread(imgnames(p['dind'],base=p['base'])[t])
    img = gputools.scale(img, [5.5,0.5,0.5])
    img = img[...,None]
    pimg = imread("training/ce_059/train_cp/pimgs/{}/pimg_{:03d}.tif".format(p['dset'],t))
    hyp = segmentation.segment_pimg_img({"img":img,"pimg":pimg}, params)
    print(hyp.shape)
    for zi,z in enumerate(zs):
      # ipdb.set_trace()
      _,mi,ma = norm_szyxc_per(img[z,...,0],(0,1),return_pc=True)
      png[ti,zi] = seg_compare2gt_rgb(hyp[z], img[z,...,0])*(ma-mi) + mi

  png = collapse2(png,"tzyxc","ty,zx,c")
  png = norm_szyxc(png,(0,1,2))
  io.imsave("training/figures/m59seg_{}_{}.png".format(p['dset'],'066'), png)

@DeprecationWarning
def segpallette2(p):
  ts = np.linspace(0,p['n']-1,11).astype(np.int)
  zs = np.linspace(5,p['z']-5,11).astype(np.int) ## ignore five slices at beginning and end. they're just noise.
  a,b,c = p['sh'][1:] + (3,)
  png = np.zeros((len(ts),len(zs),a,b,c))

  def f1(img): return np.percentile(img,99,axis=(1,2),keepdims=True)
  def f2(img): return np.percentile(img,50,axis=(1,2),keepdims=True)
  def f3(img): return np.mean(img,axis=(1,2),keepdims=True)
  def f4(img): return np.mean(img)

  params = segment_pimg_img_params
  for i in range(4):
    params['fmask'] = [f1,f2,f3,f4][i]
    for ti,t in enumerate(ts):
      img = imread(imgnames(p['dind'],base=p['base'])[t])
      img = gputools.scale(img, [5.5,0.5,0.5])
      img = img[...,None]
      pimg = imread("training/ce_059/train_cp/pimgs/{}/pimg_{:03d}.tif".format(p['dset'],t))
      hyp = segmentation.segment_pimg_img({"img":img,"pimg":pimg}, params)
      print(hyp.shape)
      for zi,z in enumerate(zs):
        _,mi,ma = norm_szyxc_per(img[z,...,0],(1,2),return_pc=True)
        png[ti,zi] = seg_compare2gt_rgb(hyp[z], img[z,...,0])*(ma-mi) + mi

    io.imsave("training/figures/m59seg_{}_{}.png".format(p['dset'],i), collapse2(png,"tzyxc","ty,zx,c"))

def mvfiles():
  for f in sorted(glob("training/ce_059/train_cp/pimgs/chall2/watershed/*.tif")):
    f = Path(f)
    # img = imread(f).astype(np.uint16)
    # print(img.shape, img.dtype, img.max())
    if False:
      # img = gputools.scale(img, [1/5.5,2.0,2.0], interpolation='nearest')
      if img.max() < 256:
        imsave(f,img.astype(np.uint8),compress=9)
      else:
        imsave(f,img.astype(np.uint16),compress=9)
    f2 = Path(seg2) / ("mask" + str(f.name)[3:])
    print(f2)
    shutil.copy(f,f2)

def showraw(raw,savedir):
  for i in [1,2,3]:
    res = clanalyze.show_rawdata(raw['vali'],i=i)
    io.imsave(savedir / 'rawdata_vali_{:02d}.png'.format(i), res)





## matching scores

def matching_2cen(cen,pimg,th):
  print(pimg.mean(),np.percentile(pimg,[2,50,90,99]))
  x = clanalyze.cen2pts(cen)
  y = clanalyze.cen2pts(pimg[0,...,0] > th)
  kdt = pyKDTree(y)
  dists, inds = kdt.query(x, k=1, distance_upper_bound=100)
  indices,counts = np.unique(inds[inds<len(y)],return_counts=True)
  return x,y,indices,counts

def match_chan(yts,yps,chs=(1,1)):
  "Samples ZYX C"
  print(yts.shape,yps.shape)
  def single(yt,yp):
    pts0 = clanalyze.cen2pts(yt[...,chs[0]]>0.5)
    pts1 = clanalyze.cen2pts(yp[...,chs[1]]>np.percentile(yp[...,chs[1]],98))
    # print(pts0)
    # print()
    # print(pts1)
    # ipdb.set_trace()
    if len(pts1) > 0:
      kdt = pyKDTree(pts1)
      dists, inds = kdt.query(pts0, k=1, distance_upper_bound=10)
      uinds,counts = np.unique(inds[inds<len(pts1)], return_counts=True)
      return len(uinds), len(pts1), len(pts0)
    return 0,len(pts1),len(pts0)
  scores = np.array([single(yt,yp) for yt,yp in zip(yts,yps)])
  f1 = 2*scores[:,0].sum() / scores[:,[1,2]].sum()
  return f1

## train multiple models consecutively

def run_many():
  for w in [3,5]:
    np.random.seed(0)
    savedir = Path('training/ce_run4/class/w{:d}/'.format(w))
    traindata = build_raw_class(savedir,buildraw,w)
    res = train_model(traindata)

    np.random.seed(0)
    savedir = Path('training/ce_run4/gauss/w{:d}/'.format(w))
    traindata = build_raw_gauss(savedir,buildraw,w)
    # traindata['raw']['params0'] = 'training/ce_run2/gauss/w3/train_cp/w007.h5'
    res = train_model(traindata)

## for training

def bugfixraw(targetfunc):
  times    = [130]
  ns       = [1]
  basedirs = ['Fluo-N3DH-CE']*len(ns)
  
  icp = imgcenpairs({'times':times,'ns':ns,'bases':basedirs})
  raw_train = m.icp2raw(icp,targetfunc)
  tar = raw_train[0]['target']
  raw_train[0]['source'] = tar + 0.1 * np.random.rand(*tar.shape)

  times    = [100]
  ns       = [1]
  basedirs = ['Fluo-N3DH-CE']*len(ns)
  
  icp = imgcenpairs({'times':times,'ns':ns,'bases':basedirs})
  raw_vali = m.icp2raw(icp,targetfunc)
  tar = raw_vali[0]['target']
  raw_vali[0]['source'] = tar + 0.1 * np.random.rand(*tar.shape)

  data = {'train':raw_train,'vali':raw_vali}
  return data

def simpleraw(targetfunc):
  times    = [130]
  ns       = [1]
  basedirs = ['Fluo-N3DH-CE']*len(ns)
  
  icp = imgcenpairs({'times':times,'ns':ns,'bases':basedirs})
  raw_train = m.icp2raw(icp, targetfunc)

  times    = [100]
  ns       = [1]
  basedirs = ['Fluo-N3DH-CE']*len(ns)
  
  icp = imgcenpairs({'times':times,'ns':ns,'bases':basedirs})
  raw_vali = m.icp2raw(icp, targetfunc)

  data = {'train':raw_train,'vali':raw_vali}
  return data

def buildraw(targetfunc):
  times    = [0,100,150,175,189,189]
  ns       = [2,1,2,1,2,1]
  basedirs = ['Fluo-N3DH-CE']*len(ns)
  
  icp = imgcenpairs({'times':times,'ns':ns,'bases':basedirs})
  raw_train = m.icp2raw(icp, targetfunc)

  times    = [70,185,185]
  ns       = [1,1,2]
  basedirs = ['Fluo-N3DH-CE']*len(ns)
  
  icp = imgcenpairs({'times':times,'ns':ns,'bases':basedirs})
  raw_vali = m.icp2raw(icp, targetfunc)

  data = {'train':raw_train,'vali':raw_vali}
  return data

## combine rawdata, callback data and net params

def build_raw_class(savedir,f_rawbuild,w):
  # targetfunc = m.cen2target_class
  xrgb = [0,0,0]
  yrgb = [1,1,1]
  raw = f_rawbuild(lambda x: m.cen2target_class(x,w))
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
    callback = m.GlobalCallback(epochend_callback)

  out_channels = raw['train'][0]['target'].shape[-1]
  netparams = {'inshape':(None,None,None,1), 'out_channels':out_channels, 'activation':'softmax', 'task':'classification', 'weights':weights, 'lr':1e-4}
  raw['netparams'] = netparams

  data = fromlocals(locals(), ['raw','callback','callbackdata'])

  return data

def build_raw_gauss(savedir,f_rawbuild,w):
  # targetfunc = m.cen2target_class
  xrgb = [0,0,0]
  yrgb = [0,0,0]
  raw = f_rawbuild(lambda x : m.cen2target_gauss(x,w=w))
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
    callback = m.GlobalCallback(epochend_callback)

  out_channels = raw['train'][0]['target'].shape[-1]
  netparams = {'inshape':(None,None,None,1), 'out_channels':out_channels, 'activation':'linear', 'task':'regression', 'lr':1e-4}
  raw['netparams'] = netparams

  data = fromlocals(locals(), ['raw','callback','callbackdata'])

  return data

## callback stuff

def build_callbackdata(rawdata):
  # ex_vali  = [(x[0],y[0,...,:-1]) for (x,y) in itertools.islice(vg, 5)]
  tg2 = m.datagen(rawdata['train'],patch_size=(120,120,120),batch_size=1)
  ex_train = [(x[0],y[0,...,:-1]) for (x,y) in itertools.islice(tg2, 5)]
  xs_train = np.array([x[0] for x in ex_train])
  ys_train = np.array([x[1] for x in ex_train])

  img_final = m.build_img(imgnames(1)[189])['source']
  cen_final = m.build_cen(cennames(1)[189])['cen']
  cen_final = np_utils.to_categorical(cen_final).reshape(cen_final.shape + (-1,))

  img_mid = m.build_img(imgnames(1)[95])['source']
  cen_mid = m.build_cen(cennames(1)[95])['cen']
  cen_mid = np_utils.to_categorical(cen_mid).reshape(cen_mid.shape + (-1,))

  img_zero = m.build_img(imgnames(1)[0])['source']
  cen_zero = m.build_cen(cennames(1)[0])['cen']
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

## train models

def train_model(traindata):
  raw = traindata['raw']

  net = m.build_net(raw['netparams'])
  params0 = traindata['raw'].get('params0',None)
  if params0:
    net.load_weights(params0)

  tg = m.datagen(raw['train'],patch_size=(80,80,80),batch_size=4)
  vg = m.datagen(raw['vali'],patch_size=(120,120,120),batch_size=1)

  callbackdata = traindata['callbackdata']
  callbackdata['scores'] = []
  callbackdata['scoresbig'] = []
  history = ts.train_gen(net, tg, vg, raw['traindir'], n_epochs=60, steps_per_epoch=20, vali_steps=10, callbacks=[traindata['callback']])

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

## move

def analyze(traindir,raw,net):
  resultdir = traindir / 'results/'; resultdir.mkdir(exist_ok=True);
  pimgdir   = traindir / 'pimgs/'; pimgdir.mkdir(exist_ok=True);

  results = analyze_cpnet(net,raw,resultdir)
  pickle.dump(results, open(resultdir / 'results.pkl', 'wb'))

  rawgt = build_gt_rawdata()
  results_gt = analyze_cpnet(net,rawgt,resultdir)
  pickle.dump(results_gt, open(resultdir / 'results_gt.pkl', 'wb'))

  # results_gt = pickle.load(open(resultdir / 'results_gt.pkl', 'rb'))
  gtdata = labnames2imgs_cens(labnames(1),1)
  mk_hyps_compute_seg(gtdata, results_gt)

  imgdir = Path('Fluo-N3DH-CE/')
  chaldir = Path('Fluo-N3DH-CE_challenge/')
  save_pimgs(net,pimgdir / 'train1', imgdir / '01', range(0,250))
  save_pimgs(net,pimgdir / 'train2', imgdir / '02', range(0,250))
  save_pimgs(net,pimgdir / 'chall1', chaldir / '01', range(0,190))
  save_pimgs(net,pimgdir / 'chall2', chaldir / '02', range(0,190))
