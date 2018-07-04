from __future__ import print_function, unicode_literals, absolute_import, division
from segtools.defaults.ipython_remote import *
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'
from csbdeep.utils.plot_utils import plot_some

import os
from tifffile import imread, TiffFile
from csbdeep import data
from scipy.ndimage import rotate

from csbdeep.data import RawData

def original():
    x = np.load('data/img006_noconv.npy')
    x = np.moveaxis(x, 1,2)
    ## initial axes are TZCYX, but we
    axes = 'CZYX'

    mypath = Path('isonet_psf_1')
    mypath.mkdir(exist_ok=True)

    subsample = 5.0 #10.2
    print('image size       =', x.shape)
    print('image axes       =', axes)
    print('subsample factor =', subsample)

    plt.switch_backend('agg')

    if False:
        plt.figure(figsize=(15,15))
        plot_some(np.moveaxis(x[0],1,-1)[[5,-5]], title_list=[['xy slice','xy slice']], pmin=2,pmax=99.8);
        plt.savefig(mypath / 'datagen_1.png')

        plt.figure(figsize=(15,15))
        plot_some(np.moveaxis(np.moveaxis(x[0],1,-1)[:,[50,-50]],1,0), title_list=[['xz slice','xz slice']], pmin=2,pmax=99.8, aspect=subsample);
        plt.savefig(mypath / 'datagen_2.png')

    def gimmeit_gen():
        ## iterate over time dimension
        for i in range(x.shape[0]):
            yield x[i],x[i],axes,None

    raw_data = RawData(gimmeit_gen, x.shape[0], "this is great!")

    ## initial idea
    if False:
        def buildkernel():
            kern = np.exp(- (np.arange(10)**2 / 2))
            kern /= kern.sum()
            kern = kern.reshape([1,1,-1,1])
            kern = np.stack([kern,kern],axis=1)
            return kern
        psf_kern = buildkernel()

    ## use Martins theoretical psf
    if False:
        psf_aniso = imread('data/psf_aniso_NA_0.8.tif')
        psf_channels = np.stack([psf_aniso,]*2, axis=1)

    def buildkernel():
        kernel = np.zeros(20)
        kernel[7:13] = 1/6
        ## reshape into CZYX. long axis is X.
        kernel = kernel.reshape([1,1,-1])
        ## repeate same kernel for both channels
        kernel = np.stack([kernel,kernel],axis=0)
        return kernel

    psf = buildkernel()
    print(psf.shape)

    ## use theoretical psf
    if False:
        psf_channels = np.load('data/measured_psfs.npy')
        psf = rotate(psf_channels, 90, axes=(1,3))

    iso_transform = data.anisotropic_distortions(
        subsample = subsample,
        psf       = psf,
    )

    X, Y, XY_axes = data.create_patches (
        raw_data            = raw_data,
        patch_size          = (2,1,128,128),
        n_patches_per_image = 256,
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

    for i in range(2):
        plt.figure(figsize=(16,4))
        sl = slice(8*i, 8*(i+1))
        plot_some(np.moveaxis(X[sl],1,-1),np.moveaxis(Y[sl],1,-1),title_list=[np.arange(sl.start,sl.stop)])
        plt.savefig(mypath / 'datagen_panel_{}.png'.format(i))

if __name__ == '__main__':
  original()


history = """
## Mon Jun 25 18:18:52 2018


"""