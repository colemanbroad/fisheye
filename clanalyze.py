from clbase import *
import cltrain as m


## matching scores

def matching_2cen(cen,pimg,th):
  print(pimg.mean(),np.percentile(pimg,[2,50,90,99]))
  x = cen2pts(cen)
  y = cen2pts(pimg[0,...,0] > th)
  kdt = pyKDTree(y)
  dists, inds = kdt.query(x, k=1, distance_upper_bound=100)
  indices,counts = np.unique(inds[inds<len(y)],return_counts=True)
  return x,y,indices,counts

def match_chan(yts,yps,chs=(1,1)):
  "Samples ZYX C"
  print(yts.shape,yps.shape)
  def single(yt,yp):
    pts0 = cen2pts(yt[...,chs[0]]>0.5)
    if len(pts0)==0: return 0,0,0
    pts1 = cen2pts(yp[...,chs[1]]>np.percentile(yp[...,chs[1]],98))
    if len(pts1)==0: return 0,0,0
    kdt = pyKDTree(pts1)
    dists, inds = kdt.query(pts0, k=1, distance_upper_bound=10)
    uinds,counts = np.unique(inds[inds<len(pts1)], return_counts=True)
    return len(uinds), len(pts1), len(pts0)

  scores = np.array([single(yt,yp) for yt,yp in zip(yts,yps)])
  f1 = 2*scores[:,0].sum() / np.maximum(scores[:,[1,2]].sum(),1)
  return f1

## do everything

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
  if False:
    save_pimgs(net,pimgdir / 'train1', imgdir / '01', range(0,250))
    save_pimgs(net,pimgdir / 'train2', imgdir / '02', range(0,250))
    save_pimgs(net,pimgdir / 'chall1', chaldir / '01', range(0,190))
    save_pimgs(net,pimgdir / 'chall2', chaldir / '02', range(0,190))

## build data

def build_gtdata():
  "{1,2} x [1..5] x {img,cen,lab,t,z,z2,labname}"
  return {1: labnames2imgs_cens(labnames(1),1), 2: labnames2imgs_cens(labnames(2),2)}

def build_gt_rawdata():
  raw = {'1':times2raw([ 21,  28,  78, 141, 162],1,basedir='Fluo-N3DH-CE'),
         '2':times2raw([10, 12, 106, 120, 126],2,basedir='Fluo-N3DH-CE')}
  return raw

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
      score,n_gt,n_hyp = evalseg(d,params,savedir2)
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
    res.append({'hyp':
      (d,best), 'params':best, 'seg':-seg})

  return res

def seg_and_save_d(d,params,savedir):
  hyp = params['method'](d, params)[d['z2']]
  show = seg_compare2gt_rgb(hyp,d['lab'],d['img'][d['z2'],...,0])
  s1 = scores_dense.seg(d['lab'],hyp)
  # s2,n_gt = scores_dense.seg(d['lab'],hyp, partial_dataset=True)
  io.imsave(savedir / 'seg_1_time_{:03d}_SEG_{:.3f}.png'.format(d['t'], s1), show)
  print("saved ", d['labname'], "with seg = ", s1)

def optimize_seg_joint(space, savedir, data=None, trials=None, max_evals=30):

  if data is None: data  = labnames2imgs_cens(labnames(2),2)
  
  def f_eval(params):
    scores = np.array([evalseg(d,params,savedir) for d in data])
    tot,n_gt,n_hyp = scores.sum(0)
    mu = tot / n_gt
    sig = np.std(scores[:,0]/scores[:,1])
    print(params, mu, sig)
    return -mu

  if trials is None: trials = ho.Trials()

  best = ho.fmin(f_eval,
    space     = space,
    algo      = ho.tpe.suggest,
    max_evals = max_evals,
    trials    = trials)
  best = ho.space_eval(space,best)

  for d in data:
    seg_and_save_d(d,best,savedir)

  pickle.dump(trials, open(savedir / 'trials.pkl', 'wb'))
  print(best)
  res = {'data':data, 'trails':trials}
  return res

def evalseg(d, params, savedir):
  hyp = params['method'](d, params)[d['z2']]
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

def seg_compare2gt_rgb(hyp,img,lab=None,norm=False,clip=False):
  "takes 2d slices [no channels] and makes rgb"
  # hyp = hyp.copy()
  # img = img.copy()
  bor = label_tools.find_boundaries(hyp)
  # hyp[hyp>0] = 1.0
  # hyp[bor] = 0.5
  
  img = img[...,np.newaxis]
  img = img[...,[0,0,0]]
    
  img[...,[0,2]] = 0
  img[bor] = img.max()
  
  if lab is not None:
    bor_gt = label_tools.find_boundaries(lab)
    img[bor_gt] += [-.7*alpha,-.7*alpha,alpha]
  
  if clip: img = img.clip(0,1)

  return img
  # d = updatekeys(dict(),locals(),['img','mi','ma'])
  # return d

## compute seg scores on training datasets

def mk_hyps_compute_seg(data, results_gt):
  pimgs = results_gt['pimg']
  best = bestparams
  best.update({'thi':0.6})
  
  seg,ngt = 0,0
  for d,p in zip(data,pimgs):
    d['pimg'] = p[...,1]
    hyp = segmentation.segment_pimg_img(d, best)
    s,n = scores_dense.seg(d['lab'], hyp[d['z2']], partial_dataset=True)
    seg+=s
    ngt+=n
    print(s,n)
  print(seg, ngt, seg/ngt)

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

## analysis

def analyze_cpnet(net,rawdata,savedir):
  result = dict()
  for k in rawdata.keys():
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
    points = invertdict(data)
    result[k] = dict()
    updatekeys(result[k],locals(),['pimg','thresh','ncells','points','dub','nmatches'])
  return result

def pimg2pts(pimg,th=0.9):
  "axes are 'tzyxc'"
  lab = label(pimg>th)[0]
  nhl = nhl_tools.hyp2nhl(lab)
  pts = np.array([n['centroid'] for n in nhl])
  return pts

def n_matchpoints(pts0,pts1,k=1,dub=3):
  "dub is distance upper bound"
  kdt = pyKDTree(pts1)
  dists, inds = kdt.query(pts0, k=k, distance_upper_bound=dub)
  nmatches = len(np.unique(inds[inds<len(pts1)]))
  return nmatches, dists

def plot_point_connections(pts_gt,pts,filename=None):
  def f(d):
    return n_matchpoints(pts, pts_gt, k=1, dub=d)[0]
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

def match_ratio_from_ytyp(yt,yp):
  pts0 = cen2pts(yt[...,1])
  pts1 = cen2pts(yp[...,1]>0.6)
  if pts1.shape[0] > 0:
    n,dists = n_matchpoints(pts1,pts0,k=1,dub=7)
    m = (dists>0) & (dists<30)
    distmean = np.mean(dists[m])
  else:
    n,distmean = 0, 0
  return n, pts0.shape[0], pts1.shape[0], distmean

def match_ratio_from_pts(pts0,pts1):
  if pts1.shape[0] > 0:
    n,dists = n_matchpoints(pts1,pts0,k=1,dub=7)
    m = (dists>0) & (dists<30)
    distmean = np.mean(dists[m])
  else:
    n,distmean = 0, 0
  return n, pts0.shape[0], pts1.shape[0], distmean

def cen2pts(cen):
  "cen is image with 1-valed blobs at centerpoints and 0 bg"
  pts = np.array([n['centroid'] for n in nhl_tools.hyp2nhl(label(cen)[0])]).astype(np.int)
  return pts


## rendering

def pallettes_from_pimgs(img_pimg_names):
  "list of tuples of filenames. returns y*x*3 rgb ndarray of segmented dataset."
  a,b,c = imread(img_pimg_names[0][1]).shape
  zs = np.linspace(10,a-10,11).astype(np.int)
  png = np.zeros((len(img_pimg_names),len(zs),b,c,3))
  params = segmentation.segment_pimg_img_params
  for ti,xy in enumerate(img_pimg_names):
    img = m.build_img(xy[0])['source']
    pimg = imread(xy[1])
    hyp = segmentation.segment_pimg_img({"img":img,"pimg":pimg}, params)
    for zi,z in enumerate(zs):
      # _,mi,ma = norm_szyxc(img[z,...,0],(0,1),return_pc=True)
      png[ti,zi] = clanalyze.seg_compare2gt_rgb(hyp[z], img[z,...,0]) #*(ma-mi) + mi
      # optional norm?
  png = collapse2(png,"tzyxc","ty,zx,c")
  png = norm_szyxc(png,(0,1,2))
  return png

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

def save_pimgs2(net, savedir, imgnames):
  savedir.mkdir(exist_ok=True)
  counts = []
  for name in imgnames:
    img  = m.build_img(name)['source']
    pimg = predict(net, img[None], outchan=2)
    pimg = pimg[0,...,1].astype(np.float16)
    p = Path(name)
    imsave(str(savedir / ('pimg_' + p.name)), pimg)
    counts.append((name, label(pimg>0.6)[1]))
  print(counts)
  print(counts, file=open(savedir / 'counts.txt','w'))
  return counts

def save_pimgs(net, savedir, imgdir, times):
  savedir.mkdir(exist_ok=True)
  counts = []
  for t in times:
    img = imread(str(imgdir / 't{:03d}.tif'.format(t)))
    img = gputools.scale(img, (5.5,0.5,0.5))

    img = img[np.newaxis,...,np.newaxis]
    pimg = predict(net, img, outchan=2)
    img = img[0,...,0]
    pimg = pimg[0,...,1].astype(np.float16)

    imsave(str(savedir / 'pimg_{:03d}.tif'.format(t)), pimg)
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

def pimgs2movie(pimgdir, imgdir, savedir, times):
  savedir.mkdir(exist_ok=True)
  for t in times:
    img = imread(str(imgdir / 't{:03d}.tif'.format(t)))
    img = gputools.scale(img, (5.5,0.5,0.5))
    pimg = imread(str(pimgdir / 'pimg_{:03d}.tif'.format(t)))
    png = render_frame(img,pimg)
    io.imsave(savedir / 'zcolor{:03d}.png'.format(t), png)

def pimgs2hyps(pimgdir,p,savedir,times):
  savedir.mkdir(exist_ok=True)
  for t in times:
    img = imread(str(p['path'] / 't{:03d}.tif'.format(t)))
    img = gputools.scale(img, (5.5,0.5,0.5))
    img = img[...,np.newaxis]
    pimg = imread(str(pimgdir / 'pimg_{:03d}.tif'.format(t)))
    # png = render_frame(img,pimg)
    hyp = segmentation.segment_pimg_img({'img':img, 'pimg':pimg},segmentation.segment_pimg_img_params)
    up = np.array(p['osh']) / np.array(p['sh'])
    # set0 = set(np.unique(hyp))
    hyp = gputools.scale(hyp.astype(np.uint16),tuple(up),interpolation='nearest')
    # set1 = set(np.unique(hyp))
    # assert set0 == set1
    # io.imsave(savedir / 'zcolor{:03d}.png'.format(t), png)
    imsave(str(savedir / 'hyp{:03d}.tif'.format(t)), hyp)
  
def hyps2movie(hypdir,imgdir,savedir,times):
  savedir.mkdir(exist_ok=True)
  for t in times:
    img = imread(str(imgdir / 't{:03d}.tif'.format(t)))
    img = gputools.scale(img, (5.5,0.5,0.5))
    hyp = imread(str(hypdir / 'hyp{:03d}.tif'.format(t)))
    png = render_frame(img,hyp)
    io.imsave(savedir / 'm_{:03d}.png'.format(t), png)

def viridis(img):
  # img.ndim == 2. can be float or int.
  mi,ma = img.min(), img.max()
  cm = np.array(plt.get_cmap('viridis').colors)
  img2 = np.floor((img-mi)/(ma-mi)*255).astype(np.int)
  rgb = cm[img2.flat].reshape(img2.shape + (3,))
  return rgb

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

def render_frame(img,pimg):
  a,b,c = img.shape #(192, 256, 354)
  rate = 0.01
  decay = np.exp(np.arange(a)*rate).reshape((-1,1,1))
  decay = np.ones(a).reshape((-1,1,1))
  print('img.shape', img.shape)
  print('decay.shape', decay.shape)

  pimg = pimg > 0.6
  # zcolor = 255*viridis((pimg*decay).argmax(0))

  img = norm_szyxc(img,(0,1,2))
  x11 = render.decay(img[:a//2]) #.max(0)
  x11 = x11[...,[0,0,0]]
  x21 = render.decay(img[a//2:]) #.max(0)
  x21 = x21[...,[0,0,0]]
  x12 = viridis((pimg*decay).argmax(0))
  x22 = viridis((pimg[::-1]*decay).argmax(0))
  top = cat([x11,x21],1)
  bot = cat([x12,x22],1)
  full = cat([top,bot],0)
  return full

def render_hyp(img,hyp):
  a,b,c = img.shape
  rate = 0.01
  # decay = np.exp(np.arange(a)*rate).reshape((-1,1,1))
  decay = np.ones(a).reshape((-1,1,1))
  print('img.shape', img.shape)
  print('decay.shape', decay.shape)

  # pimg = pimg > 0.6
  # zcolor = 255*viridis((pimg*decay).argmax(0))

  img = norm_szyxc(img,(0,1,2))
  x11 = render.decay(img[:3*a//4]) #.max(0)
  x11 = x11[...,[0,0,0]]
  x21 = render.decay(img[3*a//4:]) #.max(0)
  x21 = x21[...,[0,0,0]]
  x12 = viridis((hyp>0).argmax(0))
  x22 = viridis((hyp>0).argmax(0))
  # cmap = 
  # x12 = 
  top = cat([x11,x21],1)
  bot = cat([x12,x22],1)
  full = cat([top,bot],0)
  return full

def cell_count_over_time(savedir, a=50,b=80):
  for t in range(a,b):
    pimgname = 'training/ce_013/pimg_{:03d}.tif'.format(t)
    pimg = imread(str(pimgname))


## displaying raw data

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
  borders = (0,10,10,10)
  # patchshape_padded = list(xsem['patchsize'])
  patchshape_padded = [1,120,120,120]
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
    # x = x / x.mean((1,2,3), keepdims=True)
    # x = collapse2(x, 'szyxc','s,y,x,zc')
    container[s3] = net.predict(x)[s2]

  return container


## paper figures

def fig1():
  gtdata = pickle.load(open('training/figures/fig001.pkl','rb'))
  for i in gtdata.keys():
    rgbpanel = []
    score_total, n_gt_total = 0,0
    for j in range(len(gtdata[i])):
      d = gtdata[i][j]
      score_total += d['seg']
      n_gt_total += d['ngt']
      rgbpanel.append(d['rgb'])
    print(score_total, n_gt_total, score_total/n_gt_total)
    rgbpanel = cat(rgbpanel,1)
    plt.figure(figsize=(5*2,1.9))
    plt.imshow(rgbpanel)

    plt.gca().set_aspect('equal', 'datalim')
    plt.gca().set_position([0, 0, 1, 1])
    plt.gca().axes.get_xaxis().set_visible(False)
    plt.gca().axes.get_yaxis().set_visible(False)
    plt.gca().set_frame_on(False)
    for j in range(5):
      d = gtdata[i][j]
      plt.text(j*d['rgb'].shape[1]+120, -20, "{:.3f}".format(d['seg']/d['ngt']))
    # io.imsave('training/figures/fig001_{:d}.png'.format(i),rgbpanel)
    plt.savefig('training/figures/fig001_{:d}.png'.format(i),dpi=300,bbox_inches='tight',pad_inches=0)

def fig1pkl(gtdata):
  for i in gtdata.keys():
    pimgdir = Path('training/ce_014/pimgs{:d}/'.format(i))
    rgbpanel = []
    score_total, n_gt_total = 0,0
    for j in range(len(gtdata[i])):
      t = gtdata[i][j]['t']
      img = gtdata[i][j]['img']
      lab = gtdata[i][j]['lab']
      pimg = imread(str(pimgdir / 'pimg_{:03d}.tif'.format(t)))
      hyp = segmentation.segment_pimg_img({'img':img,'pimg':pimg}, segmentation.segment_pimg_img_params)
      z2 = gtdata[i][j]['z2']
      segscore, n_gt = scores_dense.seg(lab, hyp[z2], partial_dataset=True)
      print(segscore/n_gt)
      score_total += segscore
      n_gt_total += n_gt
      rgb = seg_compare2gt_rgb(hyp[z2],lab,img[z2,...,0])
      rgbpanel.append(rgb)
      gtdata[i][j].update({'pimg':pimg,'hyp':hyp,'seg':segscore,'ngt':n_gt,'rgb':rgb})
    print(score_total, n_gt_total, score_total/n_gt_total)
  pickle.dump(gtdata, open('training/figures/fig001.pkl','wb'))

def fig3pkl():
  pklname = Path("training/figures/fig003.pkl")
  dubs = [1,3,7,10]
  data = dict()
  for ds in ['t1','t2']:
    data[ds] = []
    for t in range(0,195):
      cen = imread(cennames(datasets[ds]['dind'])[t])
      pimg = imread('training/ce_059/train_cp/pimgs/{}/pimg_{:03d}.tif'.format(datasets[ds]['dset'],t))
      mask = pimg>0.6
      # mask = distance_transform_edt(mask)
      # mask = mask > 5
      pts = np.array([n['centroid'] for n in nhl_tools.hyp2nhl(label(mask)[0])]).astype(np.int)
      pts_gt = np.array([n['centroid'] for n in nhl_tools.hyp2nhl(cen)]).astype(np.int)
      pts_gt = np.floor((pts_gt+0.5)*[5.5,0.5,0.5]).astype(np.int) 
      n_match = 0 if len(pts)==0 else [n_matchpoints(pts,pts_gt,k=1,dub=d)[0] for d in dubs]
      n_pr = pts.shape[0]
      n_gt = pts_gt.shape[0]
      data[ds].append(updatekeys(dict(),locals(),['t','n_gt','n_pr','n_match','dubs']))
    data[ds] = invertdict(data[ds])
  pickle.dump(data,open(pklname,"wb"))

def fig3():
  pklname = Path("training/figures/fig003.pkl")
  data = pickle.load(open(pklname,'rb'))
  plt.figure()
  plt.plot(data['t2']['t'], data['t2']['n_gt'], label="GT2")
  plt.plot(data['t2']['t'], data['t2']['n_pr'], label="PRED2")
  plt.plot(data['t1']['t'], data['t1']['n_gt'], label="GT1")
  plt.plot(data['t1']['t'], data['t1']['n_pr'], label="PRED1")
  plt.legend()
  plt.savefig("training/figures/fig003.pdf")
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

def fig7pkl():
  thresh = np.linspace(0,1,100)
  data = []
  for i,t in enumerate([1,10,50,100,150,194]):
    pimg = imread('training/ce_014/pimgs1/pimg_{:03d}.tif'.format(t))
    cen = imread(cennames(1)[t])
    ngt = label(cen)[1]
    y = [label(pimg>th)[1] for th in thresh]
    data.append({'t':t, 'x':thresh, 'y':y, 'ngt':ngt})
  pickle.dump(data, open('training/figures/fig007.pkl','wb'))

def fig7():
  data = pickle.load(open('training/figures/fig007.pkl','rb'))
  plt.figure(figsize=(10,8))
  gs = matplotlib.gridspec.GridSpec(3,2)
  for i,d in enumerate(data):
    ax = plt.subplot(gs[i//2, i%2])
    ax.plot(d['x'],d['y'],label=str(d['t']))
    ax.plot(d['x'],[d['ngt'] for _ in range(len(d['x']))], label="n cells")
    plt.legend()
  plt.savefig('training/figures/fig007.pdf')
