from segtools.defaults.ipython_remote import *
import cltrain as m
import clanalyze
from clbase import *
homedir = Path("training/ce_test6")
savedir = Path("training/ce_test6")
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

## for training

def bugfixraw():
  times    = [130]
  ns       = [1]
  basedirs = ['Fluo-N3DH-CE']*len(ns)
  
  icp = imgcenpairs({'times':times,'ns':ns,'bases':basedirs})
  raw_train = m.icp2raw(icp)
  tar = raw_train[0]['target']
  raw_train[0]['source'] = tar + 0.1 * np.random.rand(*tar.shape)

  times    = [100]
  ns       = [1]
  basedirs = ['Fluo-N3DH-CE']*len(ns)
  
  icp = imgcenpairs({'times':times,'ns':ns,'bases':basedirs})
  raw_vali = m.icp2raw(icp)
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

def showraw(raw,savedir):
  for i in [1,2,3]:
    res = clanalyze.show_rawdata(raw['vali'],i=i)
    io.imsave(savedir / 'rawdata_vali_{:02d}.png'.format(i), res)

def matching_2cen(cen,pimg,th):
  print(pimg.mean(),np.percentile(pimg,[2,50,90,99]))
  x = clanalyze.cen2pts(cen)
  y = clanalyze.cen2pts(pimg[0,...,0] > th)
  kdt = pyKDTree(y)
  dists, inds = kdt.query(x, k=1, distance_upper_bound=100)
  indices,counts = np.unique(inds[inds<len(y)],return_counts=True)
  return x,y,indices,counts

def a():
  print(x.shape,y.shape)
  g3 = graphmatch.connect_points_digraph_symmetric(x,y,k=1,dub=100)
  return g3

def matching_func_gauss(yt,yp):
  pts0 = clanalyze.cen2pts(yt[...,0]>0.8)
  pts1 = clanalyze.cen2pts(yp[...,0]>0.8)
  # kdt = pyKDTree(y)
  # dists, inds = kdt.query(x, k=k, distance_upper_bound=dub)
  g3 = graphmatch.connect_points_symmetric(pts0,pts1,k=1,dub=100)
  return g3

def match_class(yts,yps):
  "Samples ZYX C"
  print(yts.shape,yps.shape)
  def single(yt,yp):
    pts0 = clanalyze.cen2pts(yt[...,1]>0.5)
    pts1 = clanalyze.cen2pts(yp[...,1]>0.8)
    g3 = graphmatch.connect_points_symmetric(pts0,pts1,k=1,dub=100)
    return len(g3.edges())/len(g3.nodes())
  return np.mean([single(yt,yp) for yt,yp in zip(yts,yps)])

def train_classifier(rawdata,savedir):
  traindir = savedir  / 'train_cp/'; traindir.mkdir(exist_ok=True);
  epochdir = traindir / 'epochs/'; epochdir.mkdir(exist_ok=True);

  weights = np.array([1,1])
  weights[1] = 16.0
  # weights[2] = 16.0
  weights = weights / weights.sum()
  weights = K.variable(weights)
  weight_decay = [2,1] #,.8]

  out_channels = rawdata['train'][0]['target'].shape[-1]
  netparams = {'inshape':(None,None,None,1), 'out_channels':out_channels, 'activation':'softmax', 'task':'classification', 'weights':weights, 'lr':1e-4}
  net = m.build_net(netparams)
  # net.load_weights('training/ce_test6/train_cp/w072_final.h5')

  tg  = m.datagen(rawdata['train'],patch_size=(48,48,48),batch_size=16)
  vg  = m.datagen(rawdata['vali'], patch_size=(120,120,120),batch_size=1)

  def build_callbackdata():
    # ex_vali  = [(x[0],y[0,...,:-1]) for (x,y) in itertools.islice(vg, 5)]
    tg2 = m.datagen(rawdata['train'],patch_size=(120,120,120),batch_size=1)
    ex_train = [(x[0],y[0,...,:-1]) for (x,y) in itertools.islice(tg2, 5)]
    xs_train = np.array([x[0] for x in ex_train])
    ys_train = np.array([x[1] for x in ex_train])

    xrgb = [0,0,0]
    yrgb = [1,1,1]

    img_final = m.build_img(imgnames(1)[189])['source']
    cen_final = m.build_cen(cennames(1)[189])['cen']
    cen_final = np_utils.to_categorical(cen_final).reshape(cen_final.shape + (-1,))

    scores = []

    callbackdata = fromlocals(locals(),['xs_train','ys_train','xrgb','yrgb','scores','img_final','cen_final'])
    return callbackdata

  def cbd_to_cb(callbackdata):
    cbd = callbackdata
    def epochend_callback(net,epoch):

      ## update class weights
      w = K.get_value(weights)
      print(w, w.shape)
      if (epoch % 1 == 0) and epoch > 0 and w[1]>(1/len(w)):
        w = w * weight_decay
        w = w / w.sum()
        print(w,w.shape)
        K.set_value(weights,w)

      ## predict on training examples
      yp_train = net.predict(cbd['xs_train'], batch_size=1)
      score_patch = match_class(cbd['ys_train'],yp_train)
      pimg_final  = clanalyze.predict(net, cbd['img_final'][None],2)
      score_final = match_class(cbd['cen_final'][None], pimg_final)

      scores.append(fromlocals(locals(), ['epoch','score_patch','score_final']))

      # if len(tvs['matching_score'])==0 or score > max(tvs['matching_score']) and trainvali=='vali':
      #   modelname = savedir / 'w_match_{:.3f}_{:03d}.h5'.format(score, epoch)
      #   model.save_weights(str(modelname))
      #   stuff['bestmodel'] = modelname
      # tvs['matching_score'].append(score)

      ## show them
      xs = cbd['xs_train'][...,xrgb]
      ys = cbd['ys_train'][...,yrgb]
      yp = yp_train[...,yrgb]
      res = clanalyze.plotlist([xs,ys,yp],1,c=1) #min(xs.shape[0],1))
      io.imsave(savedir / 'train_{:03d}.png'.format(epoch), res)
    return epochend_callback

  callbackdata = build_callbackdata()
  epochend_callback = cbd_to_cb(callbackdata)
  callback = m.GlobalCallback(epochend_callback)
  history = ts.train_gen(net, tg, vg, traindir, n_epochs=2, steps_per_epoch=15, vali_steps=5, callbacks=[callback])

  ts.plot_history(history, start=1, savepath=traindir)
  # plot_callback(callbackdata)
  # if callbackdata.get('bestmodel',None) is not None:
  #   net.load_weights(str(callbackdata['bestmodel']))
  #   net.save_weights(savedir / 'model_best.net')

  res = fromlocals(locals(),['net','history','tg','vg','savedir','traindir','callbackdata'])
  return res

def train_regressor(rawdata,savedir):
  traindir = savedir  / 'train_cp/'; traindir.mkdir(exist_ok=True);
  epochdir = traindir / 'epochs/'; epochdir.mkdir(exist_ok=True);

  weights = np.array([1,1])
  weights[1] = 16.0
  # weights[2] = 16.0
  weights = weights / weights.sum()
  weights = K.variable(weights)
  weight_decay = [2,1] #,.8]

  out_channels = rawdata['train'][0]['target'].shape[-1]
  # netparams = {'inshape':(120,120,120,1), 'out_channels':out_channels, 'activation':'softmax', 'task':'classification', 'weights':weights, 'lr':1e-4}
  netparams = {'inshape':(None,None,None,1), 'out_channels':out_channels, 'activation':'linear', 'task':'regression', 'lr':5e-4}
  net = m.build_net(netparams)
  # net.load_weights('training/ce_test6/train_cp/w072_final.h5')

  # net = m.build_net((120,120,120,1), 2, activation='softmax', weights=weights)
  # net.load_weights('/projects/project-broaddus/fisheye/training/ce_059/train_cp/epochs/w_match_0.935_132.h5')

  tg = m.datagen(rawdata['train'],patch_size=(48,48,48),batch_size=16)
  tg2 = m.datagen(rawdata['train'],patch_size=(120,120,120),batch_size=1)
  vg = m.datagen(rawdata['vali'],patch_size=(120,120,120),batch_size=1)
  ex_vali = [(x[0],y[0,...,:-1]) for (x,y) in itertools.islice(vg, 5)]
  ex_train = [(x[0],y[0,...,:-1]) for (x,y) in itertools.islice(tg2, 5)]
  xrgb = [0,0,0]
  yrgb = [0,0,0]

  source = m.build_img(imgnames(1)[189])['source']
  cen = m.build_cen(cennames(1)[189])['cen']

  callbackdata = fromlocals(locals(),['ex_train','ex_vali','xrgb','yrgb','epochdir','matching_func','matching_2cen'])
  callbackdata = updatekeys(callbackdata, locals(), ['source','cen','out_channels'])
  callback = m.GlobalCallback(callbackdata)

  history = ts.train_gen(net, tg, vg, traindir, n_epochs=2, steps_per_epoch=15, vali_steps=5, callbacks=[callback])

  ts.plot_history(history, start=1, savepath=traindir)
  plot_callback(callbackdata)
  if callbackdata.get('bestmodel',None) is not None:
    net.load_weights(str(callbackdata['bestmodel']))
    net.save_weights(savedir / 'model_best.net')

  res = fromlocals(locals(),['net','history','tg','vg','savedir','traindir','callbackdata'])
  return res

def plot_callback(callbackdata):
  plt.figure()
  for k in ['train_matching_score', 'val_matching_score', 'stack_scores',]:
    plt.plot(callbackdata[k], label=k)
  plt.legend()
  plt.savefig(callbackdata['epochdir'] / 'callback_traj.pdf')

def continuetrain(res):
  net = res['net']
  tg = res['tg']
  vg = res['vg']
  traindir = res['traindir']
  callbackdata = res['callbackdata']

  callback = m.GlobalCallback(res['callbackdata'])
  history  = ts.train_gen(net, tg, vg, traindir, n_epochs=30, steps_per_epoch=30, vali_steps=10, callbacks=[callback])
  
  ts.plot_history(history, start=1, savepath=traindir)
  plot_callback(callbackdata)
  net.load_weights(str(callbackdata['bestmodel']))
  return res

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
