from segtools import cell_view_lib as view # must be first matplotlib import
from segtools import color

import lib as ll
import sys

import numpy as np
from tifffile import imread, imsave
import scipy.ndimage as nd
import matplotlib.pyplot as plt
plt.ion()
import networkx as nx
# import pycpd
from glob import glob
import skimage as si

img = np.load('/Volumes/projects/project-broaddus/fisheye/prediction/p002/out001.npy')
img = np.load('/Volumes/projects/project-broaddus/fisheye/training/t009/img6pred.npz.npy')
img = img.transpose((0,3,1,2))