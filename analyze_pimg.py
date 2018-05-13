# %load_ext autoreload
# %autoreload 2
from ipython_remote_defaults import *

from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard, ReduceLROnPlateau

import unet
import lib as ll
from lib import splt, merg, collapse, multicat
import augmentation
from segtools import lib as seglib
from segtools import segtools_simple as ss
from scipy.ndimage.filters import convolve


QUICK = False #True

## note: all paths are relative to project home directory
mypath = Path("training/t025/")
mypath.mkdir(exist_ok=True)
myfile = Path(__file__)
print(mypath / myfile.name)
shutil.copy(myfile, mypath / 'analyze_pimg.py')
sys.stdout = open(mypath / 'stdout_analysis.txt', 'w')
sys.stderr = open(mypath / 'stderr_analysis.txt', 'w')


def condense_labels(lab):
  lab[lab==0] = 2   # background
  lab[lab==255] = 0 # nuclear membrane
  lab[lab==168] = 1 # nucleus
  lab[lab==85] = 3  # divisions
  lab[lab==198] = 1 # 4 # unknown
  return lab

# for making rgb colored images later
rgbmem = [0,1,2]
rgbdiv = [3,1,2]
divchannel = 3
nucchannel = 1
memchannel = 0

# define channels and classes
memchannel_img = 0
nucchannel_img = 1
rgbimg = [memchannel_img, nucchannel_img, nucchannel_img]
dz = 2
n_channels = 2*(1+2*dz) # xs.shape[-1]
n_classes  = 4  # len(np.unique(ys))
classweights = [1/n_classes,]*n_classes

## load data
img = imread('data/img006.tif')
lab = imread('data/labels_lut.tif')
lab = lab[:,:,0]
lab = condense_labels(lab)
mask_train_slices = lab.min((2,3)) < 2


pimg = np.load(mypath / 'pimg.npy')

def showresults():
  x = img[mask_train_slices].transpose((0,2,3,1))[...,rgbimg]
  x[...,2] = 0 # remove blue
  y = lab[mask_train_slices]
  y = np_utils.to_categorical(y).reshape(y.shape + (-1,))
  y = y[...,rgbmem]
  z = pimg[mask_train_slices][...,rgbmem]
  ss = [slice(None,None,5), slice(None,None,4), slice(None,None,4), slice(None)]
  def f(r):
    r = merg(r[ss])
    r = r / np.percentile(r, 99, axis=(0,1), keepdims=True)
    r = np.clip(r,0,1)
    return r
  x,y,z = f(x), f(y), f(z)
  res = np.concatenate([x,y,z], axis=1)
  return res
res = showresults()
io.imsave(mypath / 'results.png', res)

## max division channel across z

np.save(mypath / 'max_z_divchan', pimg[:,...,divchannel].max(1))


def find_divisions():
  x = pimg[:,:,:,:,:]
  a,b,c,d,e = x.shape
  x = x.reshape((a,b,c,d,e))
  dz = 6 ## downsample z
  div = x[:,::dz,...,divchannel].sum((2,3))
  val_thresh = np.percentile(div.flat, 95)
  n_rows_max = 7 
  tz = np.argwhere(div > val_thresh)[:n_rows_max]
  n_cols = 7 ## n columns
  lst = list(range(x.shape[0]))
  x2 = x[:,::dz]
  x2 = np.array([x2[timewindow(lst, n[0], n_cols), n[1]] for n in tz])
  a,b,c,d,e = x2.shape
  x2.shape
  x2 = perm(x2,'12yxc','1y2xc')
  a,b,c,d,e = x2.shape
  x2 = x2.reshape((a*b,c*d,e))
  x2 = x2[::4,::4,rgbdiv]
  # x2[...,0] *= 40 ## enhance division channel color!
  x2 = np.clip(x2, 0, 1)
  return x2

res = find_divisions()
io.imsave(mypath / 'find_divisions.png', res)

## quit early if QUICK

if QUICK:
  sys.exit(0)

## compute instance segmentation statistics

BLUR = False
if BLUR:
  weights = np.full((3, 3, 3), 1.0/27)
  pimg = [[convolve(pimg[t,...,c], weights=weights) for c in range(pimg.shape[-1])] for t in range(pimg.shape[0])]
  pimg = np.array(pimg)
  pimg = pimg.transpose((0,2,3,4,1))
  pimg = [[convolve(pimg[t,...,c], weights=weights) for c in range(pimg.shape[-1])] for t in range(pimg.shape[0])]
  pimg = np.array(pimg)
  pimg = pimg.transpose((0,2,3,4,1))

def instance_seg(x):
  x1 = x[...,nucchannel] # nuclei
  x2 = x[...,memchannel] # borders
  poten = 1 - x1 - x2
  mask = (x1 > 0.5) & (x2 < 0.15)
  seed = (x1 > 0.8) & (x2 < 0.1)
  res  = watershed(poten, label(seed)[0], mask=mask)
  return res
hyp = np.array([instance_seg(x) for x in pimg])
hyp = hyp.astype(np.uint16)
np.save(mypath / 'hyp', hyp)
# hyp = np.load(mypath / 'hyp.npy')

y_pred_instance = hyp[mask_train_slices]

def lab2instance(x):
  x[x!=1] = 0
  x = label(x)[0]
  return x
y_gt_instance = np.array([lab2instance(x) for x in lab[mask_train_slices]])

res = [ss.seg(x,y) for x,y in zip(y_gt_instance, y_pred_instance)]
print({'mean' : np.mean(res), 'std' : np.std(res)}, file=open(mypath / 'SEG.txt', 'w'))

nhls = seglib.labs2nhls(hyp, img[:,:,1], simple=False)
pickle.dump(nhls, open(mypath / 'nhls.pkl', 'wb'))

plt.figure()
cm = sns.cubehelix_palette(len(nhls))
for i,nhl in enumerate(nhls):
  areas = [n['area'] for n in nhl]
  areas = np.log2(sorted(areas))
  plt.plot(areas, '.', c=cm[i])
plt.savefig(mypath / 'cell_sizes.pdf')

#hi coleman! keep coding!
#best, romina :)
