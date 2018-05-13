import numpy as np
import re
import pandas
from segtools import lib as seglib
from scipy.ndimage import label
from ipython_defaults import perm, flatten

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

def highlight_segmentation_borders(img, hyp):
  # hyp2 = lib.remove_nucs_hyp(nhl[:-2], hyp)
  # iss = view.ImshowStack(hyp2)
  # return iss
  # iss = view.ImshowStack([img, hyp2])
  # mask = lib.mask_nhl(nhl[-1], hyp)
  # mask = hyp==nhl[-1]['label']
  borders = voronoi.lab2binary_neibs3d(hyp)
  img[borders<6] = img.max() + hyp2[borders<6]/hyp2.max()*img.max()
  return img

def sorted_nicely( l ): 
  """ Sort the given iterable in the way that humans expect.
      taken from https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python
  """ 
  import re
  convert = lambda text: int(text) if text.isdigit() else text 
  alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
  return sorted(l, key = alphanum_key)

def getxyslice(iss):
  x0,x1 = iss.fig.axes[0].get_xlim()
  x0,x1 = int(x0), int(x1)
  y1,y0 = iss.fig.axes[0].get_ylim() # y is inverted!
  y0,y1 = int(y0), int(y1)
  ss = [slice(None),] * iss.stack.ndim
  ss[-2] = slice(y0,y1)
  ss[-1] = slice(x0,x1)
  return ss

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

def add_z_to_chan(X, i, dz, axes="ZCYX"):
    x2 = X[i-dz:i+dz+1]
    a,b,c,d = x2.shape
    x2 = x2.reshape((a*b,c,d))
    x2 = perm(x2, "cyx", "yxc")
    return x2

def collapse(arr, axes=[[0,2],[1,3]]):
  sh = arr.shape
  perm = flatten(axes)
  arr = arr.transpose(perm)
  newshape = [np.prod([sh[i] for i in ax]) for ax in axes]
  arr = arr.reshape(newshape)
  return arr

def merg(arr, ax=0):
  "given a list of axes, merge each one with it's successor."
  if type(ax) is list:
    assert all(ax[i] <= ax[i+1] for i in range(len(ax)-1))
    for i,axis in zip(range(100),ax):
      arr = merg(arr, axis-i)
  else: # int type  
    assert ax < arr.ndim-1
    sh = list(arr.shape)
    n = sh[ax]
    del sh[ax]
    sh[ax] *= n
    arr = arr.reshape(sh)
  return arr

def splt(arr, s1=10, ax=0):
  """
  split an array into more dimensions
  takes a list of split values and a list of axes and divides each axis into two new axes,
  where the first has a length given by the split value (which must by an even divisor of the original axis length)
  res = arange(200).reshape((2,100))
  res = splt(res, 5, 1)
  res.shape == (4,5,20)

  res = arange(3*5*7*11).reshape((3*5,7*11))
  res = splt(res, [3,7],[0,1])
  res.shape == (3,5,7,11)

  you can even list the same dimension multiple times
  res = arange(3*5*7*11).reshape((3*5*7,11))
  res = splt(res, [3,5],[0,0])
  res.shape == (3,5,7,11)
  """
  sh = list(arr.shape)
  if type(s1) is list and type(ax) is list:
    assert all(ax[i] <= ax[i+1] for i in range(len(ax)-1))
    for i,spl,axis in zip(range(100),s1, ax):
      arr = splt(arr, spl, axis+i)
  elif type(s1) is int and type(ax) is int:
    s2,r = divmod(sh[ax], s1)
    assert r == 0
    sh[ax] = s2
    sh.insert(ax, s1)
    arr = arr.reshape(sh)
  return arr

def multicat(lst):
  if type(lst[0]) is list:
    # type of list is list of list of...
    # apply recursively to each element. then apply to result.
    # apply recursively to every element except last
    res = [multicat(l) for l in lst[:-1]] + [lst[-1]]
    res = multicat(res)
  else:
    # lst is list of ndarrays. return an ndarray.
    res = np.concatenate(lst[:-1], axis=lst[-1])
  return res

def multistack(lst):
  if type(lst[0]) is list:
    # type of list is list of list of...
    # apply recursively to each element. then apply to result.
    # apply recursively to every element except last
    res = [multistack(l) for l in lst[:-1]] + [lst[-1]]
    res = multistack(res)
  # elif type(lst[0]) is int:
  #   # lst is list of ndarrays. return an ndarray.
  #   res = np.stack(lst[:-1], axis=lst[-1])
  else:
    res = np.stack(lst[:-1], axis=lst[-1])
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


