from segtools.defaults.ipython_local import *
from scipy.ndimage import convolve
from scipy.signal import fftconvolve
import string
import random
from segtools import cell_view_lib as view
import ipdb
from scipy.ndimage.interpolation import zoom 

from segtools.patchmaker import grow, translate, slice_from_shape, slices_perfect_covering, apply_tiled_kernel

## utilities

def k_random_inds(n,k):
  "n choose k. returns an ndarray of indices. choose without replacement. k>n equiv to k=n."
  inds = np.arange(n)
  np.random.shuffle(inds)
  return inds[:k]

def random_inds_slice(shape, axis, nsample):
  ss = [slice(None)]*len(shape)
  ss[axis] = k_random_inds(shape[axis], nsample)
  return ss

def buildkern():
  kern = np.exp(- (np.arange(10)**2 / 2))
  kern /= kern.sum()
  return kern

def select_specific_indices(arr, slices):
  for sli in slices:
    arr = arr[sli]
  return arr

def test_trips(res,border):
  res2 = np.pad(res, [(b,b) for b in border], mode='constant')

## load data

def load_img():
  img = np.load('data/img006_noconv.npy')
  img = perm(img, 'tzcyx', 'tczyx')
  img = img[[0,4,8]]
  img = img[[0]]
  return img

def load_img__apply_psf():
  kernel = load_kernel__rotate__scale()
  def func(ind):
    t,c = ind
    return fftconvolve(img[t,c], kernel[c], mode='same')
  img = broadcast_nonscalar_func(func, img, '234')
  return img

def load_kernel__rotate__scale():
  img = load_img().astype(np.float)
  kernel = np.load('data/measured_psfs.npy')

  ## rotate in x,z plane. Is this the same as swapaxes?
  ## no. 90deg rotation maps x->z and z->-x ! swapaxes maps z->x !
  ## if our psf is mirror symmetric about x=0 then this doesn't matter.
  ## but it's measured data and not exactly symmetric... see [^1].
  kernel = rotate(kernel, 90, axes=(1,3))
  ## then downsample z (used to be x) s.t. voxel sizes agree w image
  ## take 0.1um z dim -> 0.5um to match image. see [^2].
  kernel = zoom(kernel,(1,1./5,1,1,1), order=0)
  kernel = kernel / kernel.sum((1,2,3), keepdims=True)
  print(kernel.shape)
  return kernel

## show results. These names are fixed. append only.

def compare1(img=None, img2=None):
  if img is None: img  = load_img()
  if img2 is None: img2 = load_img__apply_psf()
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

def compare3(img=None, img2=None):
  if img is None: img  = load_img()
  if img2 is None: img2 = load_img__apply_psf()
  t,c = 0,0
  x_ind = random.randint(0,400)
  x1 =  img[t,c,:,x_ind] # true xz
  z_ind = random.randint(0,71)
  x2 =  img[t,c,z_ind,:,::5].T # artifical. downsampled x
  x3 = img2[t,c,z_ind,:,::5].T # artificial. aniso blur + downsampled x.
  print(x1.shape, x2.shape, x3.shape)
  cat = np.concatenate
  view.imshowme(cat([x1,x2,x3],0))

def downsample_and_plot(img):
  ss = random_inds_from_axis(img.shape, 2, 8)
  img_z = img[ss].copy()
  img_z = perm(img_z, "tczyx", "tczyx")
  ss = random_inds_from_axis(img.shape, 3, 8)
  img_y = img[ss].copy()
  img_y = perm(img_y, "tczyx", "tcyxz")
  ss = random_inds_from_axis(img.shape, 4, 8)
  img_x = img[ss].copy()
  img_x = perm(img_x, "tczyx", "tcxzy")

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

## original

def run():
  img = load_img__apply_psf()
  iss = downsample_and_plot(img)
  return iss

## Deprecated

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
Note how the data itself is anisotropic.
using aniso blur kernel can only account for one of three types of anisotropy

1. anisotropic psf. we can measure psf and account for this.
2. anisotropic sampling. we know the sampling whenever we make an image. isonet is designed to help restore isotropic sampling.
3. anisotropic data. nothing we can do except for re-image.
what about anisotropic illumination? it's not just that the psf itself is anisotropic, but the psf varies with space!
also, there are bright regions of the image where the laser first comes in contact with the sample. 

Notice how the nuclei in `res063` are wider in top panel and brighter towards the bottom (lower z)
Also, the tissue curves at the bottom and the nuclei are oriented differently in these regions.

## Wed Jun 20 11:44:00 2018

It is now this module's responsibility to scale/subsample and rotate psfs s.t. they are appropriate for our images.

[^1]: Should we rotate or swapaxes? Rotate! Simply applying swapaxes changes the chirality of our psf,
making it non-physical. That psf doesn't exist IRL!

[^2]: How does CSBDeep work with voxel sizes? We have to provide a subsampling value, which describes
the sampling anisotropy. Is the convolution of the stack done *before* or *after* the subsampling?
1. The voxel size doesn't change, because subsampling is immediately followed by simple interpolation.
2. This subsampling + interpolation is done on x and y, not on the undersampled dimension.
Does the convolution get applied to full 3D stacks? Or just to slices?
Is there any way to have a kernel that varies w space or time?

problem: applying the 16x80x80 kernel takes several minutes... 
easy solution is to use scipy.signal.fftconvolve(...,mode='same')

fixed bug in load_img__apply_psf.
Now we rotate and downsample the psf s.t. the voxel sizes agree with image.
we have to re-normalize after this procedure otherwise image is dark!

PROBLEM: After fixing the voxel size the slices look very blurry again!
I have no way of knowing if I've got the right psf just by visual inspection.
We have to make line plots?

## Thu Jun 21 15:11:28 2018

I've checked the kernels after rotation and downsampling. They appear correct.
x width is long. y is medium. z is very short.

let's make line plots to look at the differences between artificial and real XZ slices.



"""