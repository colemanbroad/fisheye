from segtools.defaults.ipython_local import *
from segtools import cell_view_lib as view
from segtools.render import max_three_sides
from skimage.measure import block_reduce
from scipy.ndimage.interpolation import shift
# from segtools.patchmaker import centered_slice
from segtools import cell_view_lib as view
from segtools import render
from segtools.patchmaker import slice_from_shape
import ipdb

## utils

def diginorm(img):
  "quantile based normalization. remap intensity values to index of quantile."
  perc = np.percentile(img, np.linspace(0,100,min(1000,img.size//10)))
  img = np.digitize(img, perc)
  return img

def entropy(psf):
  print("Nans: ", np.isnan(psf).sum())
  # ipdb.set_trace()
  return -np.where(psf!=0, psf * np.log2(psf), 0).sum()

def print_stats(psfs):
  header = "{:>11s}  "*3
  line = "{:+11.4E}  "*3
  f = lambda mn,mx: (mx - mn) / mx
  print(header.format("max", "min", "ratio"))
  for r in psfs:
    mx, mn = r.max(), r.min()
    print(line.format(mx, mn, f(mn, mx)))
    print("Entropy : ", entropy(r))

## plotting

def plot_xyzmax(img,ax=None):
  gspec = matplotlib.gridspec.GridSpec(1,3)
  if ax is None:
    fig = plt.figure()
  for i in [0,1,2]:
    ax = plt.subplot(gspec[i])
    ax.imshow(img.max(i))

def show_psfs(psfs):
  cat = np.concatenate
  a = render.max_three_sides(psfs[0], axis=0)
  b = render.max_three_sides(psfs[1], axis=0)
  c = cat([a,b], 1)
  view.imshowme(c)

## single psf functions

def split_psf(h0, gamma=1.):
  """
  function originally by Martin Weigert
  splits h into a isotropic and anisotropic part
  h is the measured psf
  Use the anisotropic part as the input for anisotropic CARE training.
  eg:
  _, h_aniso = split_psf(measured_psf, gamma = mygamma)
  """

  orig_shape = h0.shape
  assert orig_shape[0]==orig_shape[1]==orig_shape[2]
  assert orig_shape[0]%2==0

  pads = tuple(s % 2 for s in orig_shape)
  slice_pads = tuple(slice(p, s + p) for s, p in zip(orig_shape, pads))

  h = np.pad(h0, tuple((p, 0) for p in pads), mode="constant")

  h = 1. * h / np.sum(h)

  h_iso = (h * h.transpose(1, 2, 0) * h.transpose(2, 0, 1)) ** (1. / 3)

  h_iso_f = np.fft.rfftn(np.fft.fftshift(h_iso))
  h_f = np.fft.rfftn(np.fft.fftshift(h))

  h_aniso_f = h_f * h_iso_f.conjugate() / (gamma + abs(h_iso_f) ** 2)

  h_aniso = np.fft.ifftshift(np.fft.irfftn(h_aniso_f))[slice_pads]
  h_iso = h_iso[slice_pads]
  return h_iso, h_aniso
  
def center_crop_martin(x,dshape):
  x_blur = blur(x,4)
  ind = np.unravel_index(np.argmax(x_blur),x.shape)
  ss = tuple(slice(i-s//2,i-s//2+s) for s,i in zip(dshape,ind))
  return x[ss].copy()

def recenter_psf(psf, w=40):
  "recenter and crop to valid region."
  image_center    = np.array(psf.shape) / 2 - 0.5
  brightest_pixel = np.argwhere(psf == psf.max())[0] #+ 0.5
  dr  = image_center - brightest_pixel
  psf = shift(psf, dr)
  psf = psf.clip(min=0)
  # ss  = slice_from_shape(psf.shape)
  # w   = min(np.array(psf.shape) / 2 - np.abs(dr))
  # w += w%2
  # assert w > 10
  image_center    = np.array(psf.shape) / 2 - 0.5
  ss  = patch.centered_slice(image_center, w)
  psf = psf[ss]
  return psf

def set_boundary_to_zero_and_normalize(psf):
  ss = slice_from_shape(psf.shape)
  mask = np.ones(psf.shape, dtype=np.bool)
  ss2 = patch.grow(ss, -1)
  mask[ss2] = 0
  psf = psf - psf[mask].max()
  psf = psf.clip(min=0)
  psf = psf / psf.sum(keepdims=True)
  return psf

def reshape_and_process_psf(psf, w=40):
  ## first make voxel size isotropic
  psf = block_reduce(psf, (2,1,1), np.mean)
  ## denoise hevily ;)
  psf = gputools.denoise.nlm3(psf, 15, 2, 3)
  ## then remove background
  psf = set_boundary_to_zero_and_normalize(psf)
  # psf = np.maximum(psf-np.percentile(psf,30),0)
  ## recenter psf 
  psf = recenter_psf(psf, w=w)
  ## split into iso/anio, only keep aniso
  _, psf = split_psf(psf)
  psf = psf.clip(min=0)
  ## then scale psf so voxel size matches x,y dimensions of img006.
  # ds = 5
  # psf = block_reduce(psf,(ds,ds,ds),np.mean)
  psf /= psf.sum()
  return psf

## multi-channel psfs functions

 
## global entry

def load_memimg():
  "just for fun... full quantile norm really highlights background noise!"
  img = imread('../carine_smFISH_seg/imgs/20150915_timeseries_aldob_apoeb_1_29_MembraneMiddle_predict__two.tif')
  mask = np.isnan(img)
  img[mask] = 0
  img = img[0]
  img = diginorm(img)
  return img

@DeprecationWarning
def load_crop_run():
  """
  First load the data. normalize. center, crop, get_anisotropic for each channel independently.
  Then downsample to get anisotropic voxel size...
  """
  psfs = imread('data/settingsColemandataset_4_crop.tif')
  psfs = np.moveaxis(psfs,0,1)

  # a = 170
  # ss = np.s_[:,:a,:a,:a]
  # psfs = psfs[ss]

  ## background subtraction, channel independent.
  psfs = psfs - np.percentile(psfs, 2, axis=(1,2,3), keepdims=True)
  ## normalize
  psfs = psfs / psfs.sum(axis=(1,2,3), keepdims=True)
  ## first make voxel size isotropic
  psfs = np.array([block_reduce(psfs[i],(2,1,1),np.sum) for i in [0,1]])
  ## then get anisotropic part of psf
  f = lambda idx: process_single_psf(psfs[idx[0]], w=40)
  psfs = broadcast_nonscalar_func(f, psfs, '123')
  ## then scale psf so voxel size matches x,y dimensions of img006.
  ds = 5
  psfs = np.array([block_reduce(psfs[i],(ds,ds,ds),np.sum) for i in [0,1]])
  np.save('data/measured_psfs', psfs)
  return psfs

def load():
  psfs = imread('data/settingsColemandataset_4_crop.tif')
  psfs = np.moveaxis(psfs,0,1)
  return psfs

def try_various_bg_subtraction():
  if False:
    ## percentile
    psfs = load()
    psfs = psfs - np.percentile(psfs, 2, axis=(1,2,3), keepdims=True)
    psfs = psfs.clip(min=0)
    psfs = run_after_bg_subtraction(psfs)
    print_stats(psfs)

    ## global spatial min. channel independent.
    psfs = load()
    psfs = psfs - psfs.min(axis=(1,2,3), keepdims=True)
    psfs = run_after_bg_subtraction(psfs)
    print_stats(psfs)

  ## set boundary
  psfs = load()
  # psfs = [set_boundary_to_zero_and_normalize(psfs[c]) for c in [0,1]]
  # psfs = np.array(psfs)
  psfs = run_after_bg_subtraction(psfs)
  print_stats(psfs)
  show_psfs(psfs)
  np.save('data/measured_psfs.npy', psfs)
  return psfs

## original script before refactor

def original():
  psf = imread('data/settingsColemandataset_4_crop.tif')
  psf = np.moveaxis(psf,0,1)

  a = 170
  ss = np.s_[:,:a,:a,:a]
  psf = psf[ss]

  psf = psf / psf.sum(axis=(1,2,3), keepdims=True)

  h_iso, h_aniso = split_psf(psf[1], gamma=1.0)
  res = max_three_sides(h_iso, catax=1)
  view.imshowme(res)
  res = max_three_sides(h_aniso, catax=1)
  view.imshowme(res)

  info = """
    Let's inspect the psf we got from one of the beads in:
    `/Volumes/myersspimdata/Mauricio/for_coleman/PSF_20x/settingsColemandataset_4_crop.tif`.

    How should we turn this measured psf (from imaging beads) into a psf we can use for ISONET training?
    """

  h_iso, h_aniso = split_psf(psf[1], gamma=1.0)

  debug = """
    h_aniso doesn't look like a psf, and it is not normalized (sum==0.499999) while h_iso is.
    Is the problem my choice of gamma?
    Let's plot multiple gamma values simultaneously.
    How do I know when h_aniso looks good?
    Guess.
    """

  gammas = 2.0**np.arange(-13,-3)
  res = [[max_three_sides(stk) for stk in split_psf(psf[1], g)] for g in gammas]
  res = np.array(res)
  ## axes are [gamma, iso/aniso, yx/zx,zy, ...]

  comments = """
    The proper gamma looks to be somewhere around 2^-10 or 2^-11.
    The xy and yz max projections are identical for the isotropic component.
    Is that by design?
    Using a psf with width 171 gives h_aniso with a very bright stripe along the center lines:
      x=0,y=0 (xy view)
      y=0,z=0 (yz view)
      x=0,z=0 (xz view)
    Apparently this is a know issue.
    The results look nice when the input has even width.
    """

  res = split_psf(psf[1], gamma=5e-4)
  h_aniso = h_aniso.clip(min=0)
  h_aniso = h_aniso / h_aniso.sum()

  comments = """
    Now we have to take the anisotropy of our voxel size into account.
    The size of our voxels are 0.5,1,1 µm.
    We have to adjust for this.
    We could start by simply avg-pooling along the z dimension with poolsize 2.
    Then our voxel size is an isotropic 1µm.
    Then we have to re-crop and rerun h_anioso generation.
  """

  psf = imread('data/settingsColemandataset_4_crop.tif')
  psf = np.moveaxis(psf,0,1)

  ## const padding will be cropped away later
  psf = np.array([block_reduce(psf[i,:170],(2,1,1),np.sum) for i in [0,1]])

  plot_xyzmax(h_iso)
  plot_xyzmax(psf[0])
  plot_xyzmax(psf[1])

  ## we can measure the bead's 3D position, and crop it so that it's in the center.
  # psf = psf[:,:77,:2*83, :2*77]

  ## let's center these psfs automatically with subpixel accuracy
  ## first the 0 dimension
  psf0 = psf[0]
  psf0 = psf0 / psf0.sum()
  plot_xyzmax(psf0)

  image_center    = np.array(psf0.shape) / 2 - 0.5
  center_of_mass  = (np.indices(psf0.shape)*psf0).sum((1,2,3)) #+ 0.5
  brightest_pixel = np.argwhere(psf0 == psf0.max())[0] #+ 0.5

  p0 = matplotlib.patches.Circle((image_center[2],image_center[1]),radius=1, color='r', fill=False)
  p1 = matplotlib.patches.Circle((center_of_mass[2],center_of_mass[1]),radius=1, color='g', fill=False)
  p2 = matplotlib.patches.Circle((brightest_pixel[2],brightest_pixel[1]),radius=1, color='b', fill=False)

  ax = plt.figure(1).axes[0]
  ax.add_patch(p0)
  ax.add_patch(p1)
  ax.add_patch(p2)

  results = """
    From the above we can see that the center_of_mass can't find the peak at all! Why is that?
    If the peak is off center, and the peak is not much higher than the backgound, then it may miss.
    """


  psf0_shifted = shift(psf0, image_center - brightest_pixel)

  w = [30,]*3
  def sd(i):
    return slice(floor(image_center[i]-w[i]), ceil(image_center[i]+w[i]))
  ss = [sd(i) for i in [0,1,2]]
  ## Adjust w until you can't see the const-padding from `shift`.

  plot_xyzmax(psf0_shifted[ss])

  psf_aniso_centered_cropped = broadcast_nonscalar_func(center_psf_crop_and_get_aniso, psf, '123')
  np.save('data/measured_psfs', psf_aniso_centered_cropped)

history = """

## Tue Jun 19 11:09:13 2018

Our images are too blurry after applying these psfs, so we're going to denoise them first.
We guess the additive noise has the effect of making the psfs appear more flat and broad.
We need to either denoise or do background subraction.

We have three techniques for bg subtraction:
1. subtract global min
2. subtract 2nd percentile and clip min to zero
3. subtract max over boundary pixels and clip min to zero

After consulting w Martin he recommended using denoising AND bg subtraction.
This produces reasonable looking psfs.

Now we have to think about how to do the scaling at the very end.
Really, this should be the responsibility of whoever wants to use this psf for their images.
They just have to know the voxel size of the psf, then scale/subsample as appropriate for their images.

Avg pooling or subsampling?

First, we shouldn't be averaging or subsampling for img006, because the isotropic voxel size of our
psfs already matches up with the voxel size of the x,y dimensions of img006.

*Do we want to make the z-dimension match up with the z-dimension of img006?*

Yes, probably. And we should use downsampling, not avg-pooling, because this is more in line with
what happens during microscopy.

## Wed Jun 20 11:39:25 2018

Remove 5x downsampling.
The data should be *subsampled* s.t. the voxel shapes of the psf and the image align.
The voxel shapes of the *rotated* psf and the image should align!
The image img006_noconv has 0.1x0.1x0.5 µm voxel spacing. Our psf should have 0.5x0.1x0.1 before rotation.
This problem is now exclusively in the hands of `generate_artificial_xz_slices.py`. 
We'll provide exclusively isotropic 0.1um, unrotated psfs.

res069 and res070 look like psfs measured from the two channels. They are not aligned! This shows chromatid aberration!

"""