from segtools.defaults.ipython_remote import *
import cltrain as m
import clanalyze
from clbase import *
# import clipy as c

homedir = Path("training/ce_test6")
savedir = Path("training/ce_test6")




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

def buildraw():
  times    = [70,80,130,160,180,189,189]
  ns       = [1,1,2,1,1,2,1]
  basedirs = ['Fluo-N3DH-CE']*len(ns)
  
  icp = imgcenpairs({'times':times,'ns':ns,'bases':basedirs})
  raw_train = m.icp2raw(icp,w=5)

  times    = [70,185,185]
  ns       = [1,1,2]
  basedirs = ['Fluo-N3DH-CE']*len(ns)
  
  icp = imgcenpairs({'times':times,'ns':ns,'bases':basedirs})
  raw_vali = m.icp2raw(icp,w=5)

  data = {'train':raw_train,'vali':raw_vali}
  return data

def showraw(raw,savedir):
  for i in [1,2,3]:
    res = clanalyze.show_rawdata(raw['vali'],i=i)
    io.imsave(savedir / 'rawdata_vali_{:02d}.png'.format(i), res)

def doitall(rawdata,savedir):
  traindir  = savedir  / 'train_cp/'; traindir.mkdir(exist_ok=True);
  epochdir  = traindir / 'epochs/'; epochdir.mkdir(exist_ok=True);

  weights = np.array([1,1])
  weights[1] = 16.0
  weights = weights / weights.sum()
  weights = K.variable(weights)

  net = m.build_net((120,120,120,1), 2, activation='softmax', weights=weights)
  # net.load_weights('/projects/project-broaddus/fisheye/training/ce_059/train_cp/epochs/w_match_0.935_132.h5')

  tg = m.datagen(rawdata['train'])
  vg = m.datagen(rawdata['vali'])
  examples = [(x[0],y[0,...,:-1]) for (x,y) in itertools.islice(vg, 20)]

  hical = m.Histories(examples, epochdir, weights=weights)

  history = ts.train_gen(net, tg, vg, traindir, n_epochs=100, steps_per_epoch=30, vali_steps=2, callbacks=[hical])
  ts.plot_history(history, start=1, savepath=traindir)
  net.load_weights(str(hical.bestmodel))

  res = updatekeys(dict(),locals(),['net','hical','history'])
  return res

def analyze(traindir,raw,net):
  resultdir = traindir / 'results/'; resultdir.mkdir(exist_ok=True);
  pimgdir   = traindir / 'pimgs/'; pimgdir.mkdir(exist_ok=True);

  results = clanalyze.analyze_cpnet(net,rawdata,resultdir)
  pickle.dump(results, open(resultdir / 'results.pkl', 'wb'))

  rawgt = clanalyze.build_gt_rawdata()
  results_gt = clanalyze.analyze_cpnet(net,rawgt,resultdir)
  pickle.dump(results_gt, open(resultdir / 'results_gt.pkl', 'wb'))

  # results_gt = pickle.load(open(resultdir / 'results_gt.pkl', 'rb'))
  gtdata = labnames2imgs_cens(labnames(1),1)
  clanalyze.mk_hyps_compute_seg(gtdata, results_gt)

  imgdir = Path('Fluo-N3DH-CE/')
  chaldir = Path('Fluo-N3DH-CE_challenge/')
  clanalyze.save_pimgs(net,pimgdir / 'train1', imgdir / '01', range(0,250))
  clanalyze.save_pimgs(net,pimgdir / 'train2', imgdir / '02', range(0,250))
  clanalyze.save_pimgs(net,pimgdir / 'chall1', chaldir / '01', range(0,190))
  clanalyze.save_pimgs(net,pimgdir / 'chall2', chaldir / '02', range(0,190))
