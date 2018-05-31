import numpy as np
import re
import pandas
from segtools import lib as seglib
from segtools import voronoi
from scipy.ndimage import label
from ipython_defaults import perm, flatten, merg, splt, collapse

from sortedcontainers import SortedSet, SortedDict


def autocorr(x):
    result = np.correlate(x, x, mode='full')
    print(result.shape, result.size)
    return result[result.size//2:]

def autocorr_multi(img):
  xs = np.arange(0, 400, 40, np.int)
  ys = np.arange(0, 400, 40, np.int)
  zs = np.arange(0, 70, 7, np.int)
  ts = np.arange(0, 11, 1, np.int)  
  x  = img[0,0,0,:,i]

def mkpoints():
  pointanno = pandas.read_csv("~/Desktop/Projects/fisheye/anno/point_anno.csv", header=0, sep=' ')
  points = pointanno[['X', 'Y', 'Slice']]
  points.X *= 10
  points.X = np.ceil(points.X)
  points.Y *= 10
  points.Y = np.ceil(points.Y)
  points = points.as_matrix().astype(np.int)
  points = points[:,[2,1,0]]
  points[:,0] -= 1 # fix stupid 1-based indexing
  return points


def add_z_to_chan(img, dz, ind=None, axes="ZCYX"):
  assert img.ndim == 4

  ## by default do the entire stack
  if ind is None:
    ind = np.arange(img.shape[0])

  ## pad img
  img = perm(img, axes, "ZCYX")
  pad = [(dz,dz)] + [(0,0)]*3
  img = np.pad(img, pad, 'reflect')

  ## allow single ind
  if not hasattr(ind, "__len__"):
    ind = [ind]

  def add_single(i):
    res = img[i-dz:i+dz+1]
    a,b,c,d = res.shape
    res = res.reshape((a*b,c,d))
    res = perm(res, "CYX", "YXC")
    return res

    ind = np.array(ind) + dz
    res = np.stack([add_single(i) for i in ind], axis=0)
    res = perm(res, "ZYXC", axes)

  return res


@DeprecationWarning
def fixlabels(imgWlab):
  """
  Interesting numpy trivia.
  You can use a single slice on the left side of an assignment, but not a slice of a slice!
  We have to break this into two separate assignments.
  """
  inds = imgWlab[:,:,0].max((2,3))
  x0 = imgWlab[inds>0]
  x = x0[:,0]
  x[x==0] = 2   # background
  x[x==255] = 0 # nuclear membrane
  x[x==168] = 1 # nucleus
  x[x==198] = 3 # unknown
  x[x==85] = 4  # divisions
  x0[:,0] = x
  return inds, x0

@DeprecationWarning
def labeled_slices_to_xsys(img, mask, dz=0, axes="TZCYX"):
  """
  mask is 2D over T and Z channels
  return slices from img only where labeling exists
  add a few of the surrounding planes to the channels dimension of xs
  """
  assert mask.ndim == 2
  img = perm(img, axes, "TZCYX")
  pad = [(0,0)] + [(dz,dz)] + [(0,0)]*3
  inds = np.indices(mask.shape)
  xs = []
  imgpad = np.pad(img, pad, 'reflect')
  for t,z in inds[:,mask].T:
    xs.append(add_z_to_chan(imgpad[t],z+dz,dz))
  xs = np.array(xs) # results in "XYXC"
  return xs


def add_z_to_chan(img, dz, ind=None, axes="ZCYX"):
  assert img.ndim == 4

  ## by default do the entire stack
  if ind is None:
    ind = np.arange(img.shape[0])

  ## pad img
  img = perm(img, axes, "ZCYX")
  pad = [(dz,dz)] + [(0,0)]*3
  img = np.pad(img, pad, 'reflect')

  ## allow single ind
  if not hasattr(ind, "__len__"):
    ind = [ind]

  def add_single(i):
    res = img[i-dz:i+dz+1]
    a,b,c,d = res.shape
    res = res.reshape((a*b,c,d))
    res = perm(res, "CYX", "YXC")
    return res

  ind = np.array(ind) + dz
  res = np.stack([add_single(i) for i in ind], axis=0)
  res = perm(res, "ZYXC", axes)

  return res


@DeprecationWarning
def merge_into_cols(*args, n=10):
  a,b,c,d = args[0].shape
  sx = [slice(None,n), Ellipsis, [1,1,0]]
  rows = []
  for arg in args:
    ar = arg[ss].reshape([n*b,c,3])
    ar = ar / ar.max((0,1), keepdims=True)
    rows.append(ar)
  rows = np.concatenate(rows, axis=1)
  return rows

@DeprecationWarning
def merge_into_rows(*args, n=10):
  a,b,c,d = args[0].shape
  ss = [slice(None,n), Ellipsis, [1,1,0]]
  rows = []
  for arg in args:
    ar = arg[ss]
    ar = perm(ar, 'syxc', 'ysxc').reshape([b,n*c,3])
    ar = ar / ar.max((0,1), keepdims=True)
    rows.append(ar)
  rows = np.concatenate(rows, axis=0)
  return rows

@DeprecationWarning
def tilez(img, ncols=8, ds=1):
  img = img[::ds,::ds,::ds]
  nz = img.shape[0]
  nrows,rem = divmod(nz,ncols)
  if rem>0: nrows+=1
  ny,nx = img.shape[1], img.shape[2]
  print(nrows, nz,ny,nx)
  if img.ndim==3:
      res = np.zeros((nrows*ny, ncols*nx))
  elif img.ndim==4:
      res = np.zeros((nrows*ny, ncols*nx, img.shape[3]))
  print(res.shape)
  for i in range(nz):
      r,c = divmod(i,ncols)
      sy = slice(r*ny, (r+1)*ny)
      sx = slice(c*nx, (c+1)*nx)
      if img.ndim==3:
          ss = [sy,sx]
      elif img.ndim==4:
          ss = [sy,sx,slice(None)]
      res[ss] = img[i]
  return res



def labels2nhl(lab):
  lab[lab!=1] = 2
  lab[lab==2] = 0
  hyp = label(lab)[0]
  nhl = seglib.hyp2nhl_2d(hyp)
  return hyp,nhl

def join_pimg_to_imgwlabs(imgwlabs, pimg):
  pimg = pimg.transpose((0,1,4,2,3))
  res = np.concatenate([imgwlabs, pimg], axis=2)
  return res



def broadcast_nonscalar_op(op, arr, subaxes, axes_full=None):
  "op must preserve shape. less general than broadcast_nonscalar_func, but probs faster."
  
  arr = arr.copy()
  
  N = arr.ndim
  M = len(subaxes)
  if axes_full is None:
    axes_full = axes2str(range(N))
  subaxes = axes2str(subaxes)
  newaxes = move_axes_to_end(axes_full, subaxes)
  arr = perm(arr, axes_full, newaxes)

  for idx in np.ndindex(arr.shape[:N-M]):
    arr[idx] = op(arr[idx])

  arr = perm(arr, newaxes, axes_full)
  return arr

def broadcast_nonscalar_func(func, arr, subaxes, axes_full=None):
  "func must preserve ndim, but not necessarily shape."
  N = arr.ndim
  M = len(subaxes)
  if axes_full is None:
    axes_full = axes2str(range(N))
  subaxes = axes2str(subaxes)
  newaxes = move_axes_to_end(axes_full, subaxes)
  arr = perm(arr, axes_full, newaxes)

  res = np.empty(arr.shape[:N-M],np.ndarray)
  for idx in np.ndindex(arr.shape[:N-M]):
    res[idx] = func(arr[idx]).tolist()
  res = np.array(res.tolist())

  res = perm(res, newaxes, axes_full)
  return res



def ax2dict(axes):
  d = SortedDict()
  for i,a in enumerate(axes):
    # d[a] = i
    d[i] = a
  return d



def axes2str(axes):
  "idempotent."
  return ''.join([str(x) for x in axes])

def move_axes_to_end(allaxes, subaxes):
  s1 = allaxes.translate({ord(i):None for i in subaxes})
  s2 = s1 + subaxes
  return s2



# def move_axes_to_end(arr, axes)