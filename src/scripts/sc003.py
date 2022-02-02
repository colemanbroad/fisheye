import numpy as np
import sys

import skimage.io as io
import matplotlib.pyplot as plt
plt.ion()
from scipy.ndimage import zoom, label
from scipy.signal import gaussian
from scipy.ndimage.morphology import binary_dilation

from tabulate import tabulate
import pandas
from skimage.morphology import watershed
from segtools import voronoi

import gputools
import spimagine

from segtools import cell_view_lib as view
from segtools import lib

ys_avgd = np.load('training/t005/ys_avgd.npy')
ys_unet_dense_0 = np.load('data_predict/ys_unet_dense_0.npy')
ys_unet_dense_0_upscaled = zoom(ys_unet_dense_0, (356/71,1,1,1))

ys_unet_dense_0_upscaled = lib.normalize_percentile_to01(ys_unet_dense_0_upscaled, 0, 100)

iss = view.ImshowStack(ys_avgd, colorchan=True)
w = spimagine.volshow(ys_avgd[...,0], interpolation="nearest")

