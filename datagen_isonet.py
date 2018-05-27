from __future__ import print_function, unicode_literals, absolute_import, division
from ipython_remote_defaults import *
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import os
from tifffile import imread, TiffFile
from csbdeep import data
from csbdeep.plot_utils import plot_some
from scipy.ndimage import rotate

# from csbdeep.utils import download_and_extract_zip_file
# download_and_extract_zip_file(
#     url = 'https://cloud.mpi-cbg.de/index.php/s/Vu0rN1G33z9hQa4/download',
#     provides = ('raw_data/retina/cropped_farred_RFP_GFP_2109175_2color_sub_10.20.tif',)
# )

# x = imread('raw_data/retina/cropped_farred_RFP_GFP_2109175_2color_sub_10.20.tif')
# x = imread('data/img006_noconv.tif')
x = np.load('data/img006_noconv.npy')

# with TiffFile('data/img006.tif') as tif:
#     images = tif.asarray()
#     for page in tif:
#         for tag in page.tags.values():
#             t = tag.name, tag.value
#             print(t)
#         image = page.asarray()

mypath = Path('isonet')
mypath.mkdir(exist_ok=True)

x = x[3]
axes = 'ZCYX'
subsample = 5.0 #10.2
print('image size       =', x.shape)
print('image axes       =', axes)
print('subsample factor =', subsample)

plt.switch_backend('agg')

plt.figure(figsize=(15,15))
plot_some(np.moveaxis(x,1,-1)[[5,-5]], title_list=[['xy slice','xy slice']], pmin=2,pmax=99.8);
plt.savefig(mypath / 'datagen_1.png')

plt.figure(figsize=(15,15))
plot_some(np.moveaxis(np.moveaxis(x,1,-1)[:,[50,-50]],1,0), title_list=[['xz slice','xz slice']], pmin=2,pmax=99.8, aspect=subsample);
plt.savefig(mypath / 'datagen_2.png')

from csbdeep.data import RawData
def gimmeit_gen():
    yield x,x,axes,None

raw_data = RawData(gimmeit_gen, 1, "this is great!")

# raw_data = data.get_tiff_pairs_from_folders (
#     basepath    = 'data',
#     source_dirs = ['isonet'],
#     target_dir  = 'isonet',
#     axes        = axes,
# )

def buildkern():
    kern = np.exp(- (np.arange(10)**2 / 2))
    kern /= kern.sum()
    kern = kern.reshape([1,1,-1,1])
    kern = np.stack([kern,kern],axis=1)
    return kern

psf_kern = buildkern()





psf_aniso = imread('data/psf_aniso_NA_0.8.tif')
psf_channels = np.stack([psf_aniso,]*2, axis=1)
# psf_channels  = psf_aniso[:,np.newaxis]
iso_transform = data.anisotropic_distortions(
    subsample = subsample,
    psf       = None,
    # psf       = rotate(psf_channels, 90, axes=(0,2)),
)

X, Y, XY_axes = data.create_patches (
    raw_data            = raw_data,
    patch_size          = (1,2,128,128),
    #patch_axes          = axes,
    n_patches_per_image = 512,
    transforms          = [iso_transform],
)

assert X.shape == Y.shape
print("shape of X,Y =", X.shape)
print("axes  of X,Y =", XY_axes)

# remove dummy z dim to obtain multi-channel 2D patches
X = X[:,:,0,...]
Y = Y[:,:,0,...]
XY_axes = XY_axes.replace('Z','')

assert X.shape == Y.shape
print("shape of X,Y =", X.shape)
print("axes  of X,Y =", XY_axes)

np.savez(mypath / 'my_training_data.npz', X=X, Y=Y, axes=XY_axes)

from csbdeep.plot_utils import plot_some

for i in range(2):
    plt.figure(figsize=(16,4))
    sl = slice(8*i, 8*(i+1))
    plot_some(np.moveaxis(X[sl],1,-1),np.moveaxis(Y[sl],1,-1),title_list=[np.arange(sl.start,sl.stop)])
    plt.savefig(mypath / 'datagen_panel_{}.png'.format(i))


