from segtools.defaults.ipython_local import *
psf = imread('data/settingsColemandataset_4_crop.tif')
psf = np.moveaxis(psf,0,1)
a = 170
ss = np.s_[:,:a,:a,:a]
psf = psf[ss]

psf = psf / psf.sum(axis=(1,2,3), keepdims=True)

from segtools import cell_view_lib as view
from segtools.render import max_three_sides

res = max_three_sides(h_iso, catax=1)
view.imshowme(res)
res = max_three_sides(h_aniso, catax=1)
view.imshowme(res)

info = """
  Let's inspect the psf we got from one of the beads in:
  `/Volumes/myersspimdata/Mauricio/for_coleman/PSF_20x/settingsColemandataset_4_crop.tif`.

  How should we turn this measured psf (from imaging beads) into a psf we can use for ISONET training?
  """

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
  The results look nice for 
  """

res = split_psf(psf[1], gamma=5e-4)
h_aniso = h_aniso.clip(min=0)
h_aniso = h_aniso / h_aniso.sum()













