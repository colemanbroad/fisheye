from segtools.defaults.ipython import *
from segtools.defaults.training import *

cat = np.concatenate
import gputools
import scipy.ndimage.morphology as morph
from sklearn.mixture import GaussianMixture

from skimage.morphology import disk         
from skimage.filters import threshold_otsu, rank


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
    return otsu
  dat = np.array([f(i) for i in range(1,len(centers))])
  am = dat.argmin()
  print(centers[am], dat[am])
  return centers[am]


bestparams1 = {'compactness': 500,  'm_mask': 1.519235931075873,  'nblur': 1, 'sigma': 15.106708203111125}
bestparams1 = {'compactness': 10,   'm_mask': 1.6140250787144337, 'nblur': 0, 'sigma': 43.14356469542666}
bestparams  = {'compactness': 1000, 'm_mask': 1.4583022667432466, 'nblur': 0, 'sigma': 21.53564789198287}
best065 = {'compactness': 10, 'fmask': lambda img: np.mean(img,axis=(1,2),keepdims=True), 'm_mask': 1.4957586697544223, 'nblur': 0, 'sigma': 26.705392772329496}
best066 = {'compactness': 10, 'fmask': lambda img: np.mean(img,axis=(1,2),keepdims=True), 'm_mask': 1.506104652944799, 'nblur': 0, 'sigma': 27.21768359449283}
bestOtsu = {'compactness': 10, 'fmask': lambda img: otsu3d(img), 'm_mask': 1., 'nblur': 0, 'sigma': 27.21768359449283}
# best067 = {'compactness': 1000, 'm_mask': 85.45836136315079, 'nblur': 0, 'sigma': 28.40330819537189}

def segment_otsu(d, params):
  img2 = gputools.denoise.nlm3(d['img'][...,0], sigma=params['sigma'])
  hx = np.array([1,1,1]) / 3
  for _ in range(params['nblur']):
    img2 = gputools.convolve_sep3(img2,hx,hx,hx)
  img2 = norm_szyxc(img2,(0,1,2))
  # local_otsu = np.array([rank.otsu(img2[i], disk(params['m_mask']))/255 for i in range(img2.shape[0])])
  otsu = otsu3d(img2)
  hyp = watershed(-img2, label(d['cen']>0)[0], mask=img2>otsu, compactness=params['compactness'])
  return hyp

def segment_otsu_paramspace():
  space = dict()
  space['sigma']       = 27 #ho.hp.uniform('sigma',      1,50)
  space['nblur']       = 0 #ho.hp.choice('nblur',       [0,1,2,3])
  # space['m_mask']      = ho.hp.uniform('m_mask',     -1,1)
  space['compactness'] = ho.hp.choice('compactness', [0.1, 1, 10, 100, 1000])
  # space['fmask']       = ho.hp.choice('fmask', [f2,f3,f4])
  space['method'] = segment_otsu
  return space

def segment(d, params):
  img2 = gputools.denoise.nlm3(d['img'][...,0], sigma=params['sigma'])
  hx = np.array([1,1,1]) / 3
  for _ in range(params['nblur']):
    img2 = gputools.convolve_sep3(img2,hx,hx,hx)
  hyp = watershed(-img2, label(d['cen']>0)[0], mask=img2>img2.mean((1,2),keepdims=True)*params['m_mask'], compactness=params['compactness'])
  return hyp

def segment_paramspace():
  # def f1(img): return np.percentile(img,99,axis=(1,2),keepdims=True)
  # def f2(img): return np.percentile(img,50,axis=(1,2),keepdims=True)
  # def f3(img): return np.mean(img,axis=(1,2),keepdims=True)
  # def f4(img): return np.mean(img)
  
  space = dict()
  space['sigma']       = ho.hp.uniform('sigma',      1,50)
  space['nblur']       = 0 #ho.hp.choice('nblur',       [0,1,2,3])
  space['m_mask']      = ho.hp.uniform('m_mask',     0,2)
  space['compactness'] = ho.hp.choice('compactness', [10, 100, 500, 1000])
  # space['fmask']       = ho.hp.choice('fmask', [f2,f3,f4])
  return space

def segment_pimg_img(d, params):
  img = d['img'][...,0].astype(np.float32)
  pimg = d['pimg']
  hx = np.array([1,1,1]) / 3
  img2 = gputools.denoise.nlm3(img, sigma=params['sigma'])
  for _ in range(params['nblur']):
    img2 = gputools.convolve_sep3(img2,hx,hx,hx)
  hyp = watershed(-img2, label(pimg > params['thi'])[0], mask=img2>params['fmask'](img2)*params['m_mask'], compactness=params['compactness'])
  return hyp

segment_pimg_img_params = {'thi':0.6, **best066}

def norm_szyxc(img,axs=(1,2,3)):
  mi,ma = img.min(axs,keepdims=True), img.max(axs,keepdims=True)
  img = (img-mi)/(ma-mi)
  return img

def norm_szyxc_per(img,axs=(1,2,3),pc=[2,99.9],return_pc=False):
  mi,ma = np.percentile(img,pc,axis=axs,keepdims=True)
  img = (img-mi)/(ma-mi)
  img = img.clip(0,1)
  if return_pc:
    return img,mi,ma
  else:
    return img
