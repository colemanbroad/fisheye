from segtools.defaults.ipython_local import *
from scipy.ndimage import convolve
import string
import random


def k_random_inds(n,k):
  "n choose k. returns an ndarray of indices. choose without replacement. k>n equiv to k=n."
  inds = np.arange(n)
  np.random.shuffle(inds)
  return inds[:k]

def buildkern():
  kern = np.exp(- (np.arange(10)**2 / 2))
  kern /= kern.sum()
  return kern

def downsample_slice(img, axis=0, kern=buildkern()):
  assert img.ndim==2
  kern_shape = [1,1]
  kern_shape[axis] = len(kern)
  kern = kern.reshape(kern_shape)
  img = convolve(img, kern)  
  ss = list(np.s_[:,:])
  ds = 71*5
  ss[axis] = slice(None,ds,5)
  img = img[ss]
  # img = imresize(img, newshape, interp='nearest')
  return img

def downsample_stack(img, axis=0, factor=5, kern=buildkern()):
  assert img.ndim==3
  # kern_shape = [1,1]
  # kern_shape[axis] = len(kern)
  # kern = kern.reshape(kern_shape)
  img = convolve(img, kern)
  ss = list(np.s_[:,:])
  # ds = 71*factor
  ss[axis] = slice(None,None,factor)
  img = img[ss]
  # img = imresize(img, newshape, interp='nearest')
  return img

def original():
  img = np.load('data/img006_noconv.npy')
  img = perm(img, 'tzcyx', 'tczyx')

  if False:
    ## take a random subset of 20% of z slices and 3 timepoints
    inds = np.arange(img.shape[1])
    np.random.shuffle(inds)
    zi = inds[:10]

    ss = [slice(None)]*5
    ss[0] = [0,4,8]

  img = img[[0,4,8]]
  s = img.shape
  
  def subsample_ind(img, axis, nsample):
    ss = [slice(None)]*img.ndim
    ss[axis] = k_random_inds(img.shape[axis], nsample)
    res = img[ss]
    return res

  img_z = subsample_ind(img, 2, 8)
  img_z = perm(img_z, "tczyx", "tczyx")
  img_y = subsample_ind(img, 3, 8)
  img_y = perm(img_y, "tczyx", "tcyxz")
  img_x = subsample_ind(img, 4, 8)
  img_x = perm(img_x, "tczyx", "tcxyz")

  # kernel = 
  def makecat():
    img_z_down_x = broadcast_nonscalar_func(lambda x: downsample_slice(x,0), img_z, (3,4))
    img_z_down_x = np.swapaxes(img_z_down_x, -1, -2)
    img_z_down_y = broadcast_nonscalar_func(lambda x: downsample_slice(x,1), img_z, (3,4))
    cat = np.concatenate([img_z_down_x, img_x, img_z_down_y, img_y], axis=-1)
    return cat

  iss = Stack(makecat())
  return iss




history = """

## Stuff to learn

learn the following functions:
  np.piecewise
  np.frompyfunc
  np.vectorize

  np.apply_along_axis
  np.apply_over_axes

  np.fromfunctiopn

  np.s_
  np.index_exp
  np.ndindex

  _end = '_end_'

  
## Mon Jun 18 14:42:20 2018

Refactoring into functions. We want to apply the psfs coming from `look_at_psf.py`.
We can run the script. Now we want to use the new psfs.

Currently the point spread function in `buildkern` returns an array with ndim=1. We need to apply psfs with ndim=3! This can still be done with `convolve`. 

`downsamp` only works with slices. This is fundamentally wrong. We want to apply our psf to the full 3D image, then cut it into slices!

The correct order of operations is
1. load psfs (anisotropic component only)
2. rotate psfs
3. apply to full 3D image
4. downsample
5. view image slices

The current problem is there's no way to know if your downsampled slices look like the real slices.
The `iss` panel shows img_z_down_x and img_x, but they have no meaningful correspondence.


"""