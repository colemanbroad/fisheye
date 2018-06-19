from segtools.defaults.ipython_local import *
from scipy.ndimage import convolve
import string
import random
from segtools import cell_view_lib as view

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

def select_specific_indices(arr, slices):
  for sli in slices:
    arr = arr[sli]
  return arr

def load():
  img = np.load('data/img006_noconv.npy')
  img = perm(img, 'tzcyx', 'tczyx')
  img = img[[0,4,8]]
  img = img[[0]]
  return img

def load_img__apply_psf():
  img = load()
  kernel = np.load('data/measured_psfs.npy')

  def func(ind):
    t,c = ind
    res = convolve(img[t,c], kernel[c])
    return res
  kernel = rotate(kernel, 90, axes=(1,3))
  img = broadcast_nonscalar_func(func, img, '234')

  if False:
    ## take a random subset of 20% of z slices and 3 timepoints
    inds = np.arange(img.shape[1])
    np.random.shuffle(inds)
    zi = inds[:10]

    ss = [slice(None)]*5
    ss[0] = [0,4,8]
  
  return img

def compare1(img=None, img2=None):
  if img is None:
    img  = load()
  if img2 is None:
    img2 = load_img__apply_psf()
  iss1  = Stack(np.swapaxes(img, 2, 4)) # swap z,x
  iss2  = Stack(img[...,::5]) # downsample x
  iss3  = Stack(np.swapaxes(img2, 2, 4)) # swap z,x in blurred image
  iss4  = Stack(img2[...,::5]) # downsample x
  return iss1, iss2, iss3, iss4

def compare2(img, img2):
  "img2 = convolve(img, kernel)"
  img = img[0]
  img2 = img2[0]
  x = np.swapaxes(img,  1, 3)
  y = np.swapaxes(img2, 1, 3)
  iss1 = Stack([x,y]) # swap x,z

  x = img[...,::5]
  y = img2[...,::5]
  iss2 = Stack([x,y])
  return iss1, iss2

def compare3(img, img2):
  t,c = 0,0
  x_ind = random.randint(0,400)
  x1 =  img[t,c,: ,x_ind] # true xz
  z_ind = random.randint(0,71)
  x2 =  img[t,c,z_ind,:    ,::5].T # artifical. downsampled x
  x3 = img2[t,c,z_ind,:    ,::5].T # artificial. aniso blur + downsampled x.
  print(x1.shape, x2.shape, x3.shape)
  cat = np.concatenate
  view.imshowme(cat([x1,x2,x3],0))

def downsample_and_plot(img):
  def random_inds_from_axis(img, axis, nsample):
    ss = [slice(None)]*img.ndim
    ss[axis] = k_random_inds(img.shape[axis], nsample)
    res = img[ss]
    return res

  img_z = random_inds_from_axis(img, 2, 8)
  img_z = perm(img_z, "tczyx", "tczyx")
  img_y = random_inds_from_axis(img, 3, 8)
  img_y = perm(img_y, "tczyx", "tcyxz")
  img_x = random_inds_from_axis(img, 4, 8)
  img_x = perm(img_x, "tczyx", "tcxyz")

  ss = [slice(None)]*img.ndim
  ss[3] = slice(None,None,5)
  img_z_down_x = img_z[ss]
  img_z_down_x = np.swapaxes(img_z_down_x, -1, -2)
  img_z_down_x = img_z_down_x[...,:71]

  ss = [slice(None)]*img.ndim
  ss[4] = slice(None,None,5)
  img_z_down_y = img_z[ss]
  img_z_down_y = img_z_down_y[...,:71]

  cat = np.concatenate([img_z_down_x, img_x, img_z_down_y, img_y], axis=-1)

  iss = Stack(cat)
  return iss

def run():
  img = load_img__apply_psf()
  iss = downsample_and_plot(img)
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

Now we apply the convolution to the entire stack...
And since this is costly we split original into load_img__apply_psf and downsample_and_plot.

`compare1` shows how applying the convolution makes x way too blurry.
Simple downsampling of x without blurring gives the most convincing yz lookalike.
1. The convolution is applied to make the effective psf isotropic, but this is done on top of the image noise.
2. This has the effect of hiding noise and the high frequency signal.
3. Ideally we would first denoise, then apply the pure-anisotropic psf, then *renoise*!
    This assumes noise happens mostly at the end, after the psf has been applied.
    Which is True, for finite sampling noise and detector noise?

## Tue Jun 19 11:00:16 2018

We should not just apply random operations to the psf to make iss3 and iss4 look less blurry.
Potential causes for this problem are:
1. We should denoise img, apply psf, renoise img for reasons outlined above.
2. We should denoise *the psf* (at what stage? before separating anisotropic part?)
    Is this a real explanation? Does the additive noise have the effect of broadening our psfs?

Let's try denoising the psfs first when they are created. See `look_at_psf.py`.
Denoising + background subtraction leads to nice psfs.
But the true xz and artificial xz slices still don't look identical.
And in fact the artificial xz made by downsampling *without* anisotropic blur look better!
This is because they have the proper noise statistics. See `res063` = compare3().

"""