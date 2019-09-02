
import matplotlib.pyplot as plt
import numpy as np
from segtools import cell_view_lib as view
from segtools import scores_dense

from segtools.defaults.ipython_local import *
import gputools
import ipdb
from subprocess import run

from sklearn.mixture import GaussianMixture
from segtools import label_tools

find_boundaries = label_tools.find_boundaries

def build_rawdata(homedir):
  imgname = '/net/fileserver-nfs/stornext/snfs2/projects/myersspimdata/Mauricio/for_coleman/ph3_labels_trainingdata_07_10_2018/trainingdata_ph3labels_hspph2bandbactinlap2bline_fish6_justph3andh2b.czi'
  img = czifile.imread(imgname)
  img = img[0,0,0,0,0,:,0,:,:,:,0] # CZYX
  img = perm(img, "CZYX","ZYXC")
  shrink = 8
  sh = np.array(img.shape)[:-1] // shrink
  ss = patchmaker.se2slices(sh,(shrink-1)*sh)
  img2 = img[ss]
  img2 = norm(img2)

  # img2 = img[ss]
  # img2 = img2 / img2.max(axis=(0,1,2), keepdims=True)

  sig = 20
  wid = 40
  x = np.linspace(-3,3,40)
  y = np.exp((-x*x)/2)
  y = y / y.sum()
  def f(x): return np.exp(-(x*x).sum()/(2*sig**2))
  kern = math_utils.build_kernel_nd(wid,3,f)
  kern = kern[::5] ## anisotropic kernel matches img
  kern = kern / kern.sum()

  img[...,0] = fftconvolve(img[...,0], kern, mode='same')
  # img = img / img.mean(axis=(0,1,2), keepdims=True)

  r = 1 ## xy downsampling factor
  imgsem = {'axes':"ZYXC", 'ph3':0, 'h2b':1, 'n_channels':2, 'r':r} ## image semantics

  res = dict()
  res['img'] = img[:,::r,::r]
  res['imgsem'] = imgsem
  return res

def lineplot_one_image(img2d, n=25):
  # f = plt.figure()
  view.imshowme(img2d)
  # def lineplot(row):
  y,x = img2d.shape
  img2d = img2d / img2d.max()
  def lineprof(i): plt.plot(np.r_[0:x], -img2d[i,:]*10+i)
  for i in np.r_[n:y:n]:
    lineprof(i)

def build_rawdata(homedir, savedir):
  homedir = Path(homedir)
  savedir = Path(savedir)

  labnames = glob(str(homedir / 'Fluo-N3DH-CE/01_GT/SEG/man_seg_*'))

  print(labnames)
  cat = np.concatenate
  def imread2(x): return imread(str(x))

  imgs = []
  cens = []
  labs = []
  times_zs = []
  hyps = []
  for i in range(len(labnames)):
    i = 2
    ln = labnames[i]
    tz = (int(ln[-11:-8]), int(ln[-7:-4]))
    t,z = tz
    lab = imread2(ln)
    img = imread2(homedir / 'Fluo-N3DH-CE/01/t{:03d}.tif'.format(t))
    cen = imread2(homedir / 'Fluo-N3DH-CE/01_GT/TRA/man_track{:03d}.tif'.format(t))
    
    img = gputools.scale(img, (5.5,0.5,0.5))
    lab = zoom(lab, (0.5,0.5), order=0)
    lab = label(lab)[0]
    # cen = label(cen)[0].astype(np.uint16)
    cen = cen>0
    cen = gputools.scale(cen, (5.5,0.5,0.5), interpolation='linear')
    cen = label(cen>0.5)[0]
    img2 = gputools.denoise.nlm3(img, sigma=20)
    hyp = watershed(img2, cen, mask=img2>img2.mean()*1.0, compactness=1000)

    z2 = int(z*5.5)
    print(ln)
    print(scores_dense.seg(lab, hyp[z2]))
    # ipdb.set_trace()
    return locals()

def a():
  def norm(x): return (x-x.min())/(x.max()-x.min())
  out = [norm(x) for x in [img2[z2],lab,cen[z2-3:z2+3].max(0),hyp[z2]]]
  out = cat(out,1)
  io.imsave(savedir / "img{:03d}.png".format(i), out)

def pngopen():
  res = run(['rsync efal:pngsave.png .'], shell=True)
  print(res)
  img = io.imread('pngsave.png')
  open_in_preview(img)
  return img

def testconv(shape):
  pts = np.random.rand(100,3)
  shape = np.array(shape)

  wid = 6*8
  pts = (pts*(shape-(2*wid))).astype(np.int) + wid
  sig = np.array([1,1,1])/8
  def f(x): return np.exp(-(sig*x*sig*x).sum()/2)
  kern = math_utils.build_kernel_nd(wid,3,f)
  st = np.array(kern.shape)
  weights = np.zeros(shape)
  sh = np.array(shape)

  for p in pts:
    start = np.floor(p - wid//2).astype(np.int)
    end   = np.floor(p + wid//2).astype(np.int)
    ss = patchmaker.se2slices(start,end)
    weights[ss] += kern

  return pts, weights

def runalex():
  times = range(100)
  data = np.ones((100,10,4))
  imgs = np.ones((100,10,7,100,100))
  for t in times:
    for i,th in enumerate(np.arange(10)/20 + 0.05):
      img = imread('/Users/broaddus/Downloads/Archive 2/predGT/predGT_{:d}.tif'.format(t))
      pimg1 = imread('/Users/broaddus/Downloads/Archive 2/predSynth/predSynth_{:d}.tif'.format(t))
      pimg2 = imread('/Users/broaddus/Downloads/Archive 2/predRaw/predRaw_{:d}.tif'.format(t))
      raw = imread('/Users/broaddus/Downloads/raw/raw_{:d}.tif'.format(t))
      lab_gt = label(1-img)[0]
      lab1  = label(pimg1 < th)[0]
      lab2  = label(pimg2 < th)[0]
      # imsave('lab1/lab_synth_{:d}.tif'.format(t), lab1)
      # imsave('lab2/lab_raw_{:d}.tif'.format(t), lab2)
      pr1 = scores_dense.precision(lab_gt,lab1)
      pr2 = scores_dense.precision(lab_gt,lab2)
      sg1 = scores_dense.seg(lab_gt,lab1)
      sg2 = scores_dense.seg(lab_gt,lab2)
      # imgs[t,i] = [raw,img,pimg1,pimg2,lab_gt,lab1,lab2]
      data[t,i] = [pr1,pr2,sg1,sg2]
  return data,imgs

def otsu3d(img,bounds=None):
  counts, bins = np.histogram(img,bins=256)
  centers = bins[:-1] + (bins[1]-bins[0])/2

  def f(i):
    n1 = counts[:i].sum()
    n2 = counts[i:].sum()
    p1 = counts[:i] / n1
    p2 = counts[i:] / n2
    w1 = centers[:i]
    w2 = centers[i:]
    var1 = (p1*w1**2).sum() - (p1*w1).sum()**2
    var2 = (p2*w2**2).sum() - (p2*w2).sum()**2
    otsu = (n1*var1 + n2*var2)/(n1+n2)
    # otsu = np.log2(otsu)
    # print(otsu)
    return otsu
  dat = np.array([f(i) for i in range(1,len(centers))])
  am = dat.argmin()
  print(centers[am], dat[am])
  return centers[am]

def regionotsu(img3):
  # threshimg = np.ones(img3.shape)
  sh = (50,1,2)
  threshimg  = np.zeros(sh)
  patchshape = [ceil(1*img3.shape[i]/sh[i]) for i in range(img3.ndim)]
  # res = patchmaker.patchtool({'img':img3.shape,'patch':(170//2,256//4,356//4),'overlap_factor':(2,2,2)})
  res = patchmaker.patchtool({'grid':sh,'img':img3.shape,'patch':patchshape})
  for i in range(len(res['starts'])):
    ss = res['slices'][i]
    ind = res['inds'][i]
    th = otsu3d(img3[ss])
    # th = np.percentile(img3[ss],90) #*0.75 #.mean()*2.5
    # th = img3[ss].max
    threshimg[tuple(ind)] = th
  scale = [img3.shape[i]/threshimg.shape[i] for i in range(img3.ndim)]
  threshimg = gputools.scale(threshimg.astype(np.float32),scale)
  return threshimg

def boundimg(img3,iss):
  threshimg = regionotsu(img3)
  bds = find_boundaries(img3>threshimg)
  iss.stack = img3.copy()
  iss.stack[bds] = 0


def invertdict(lod, f=lambda x: np.array(x)):
  d2 = dict()
  for k in lod[0].keys():
    d2[k] = f([x[k] for x in lod])
  return d2

def revertdict(dol):
  res = []
  for i in range(len(list(dol.values())[0])):
    print(i)
    res.append({k:v[i] for k,v in dol.items()})
  return res

def plot_trajectories():
  pass

