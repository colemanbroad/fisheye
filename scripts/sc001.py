import numpy as np
import sys
sys.path.insert(0, '/Users/colemanbroaddus/Desktop/Projects/nucleipix/')

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

# import gputools
# import spimagine

from segtools import cell_view_lib as view
from segtools import lib

from keras.utils import np_utils
import keras.backend as K

img6  = io.imread('data_train/img006.tif')
img6  = img6.transpose((0,1,3,4,2))

ys_rf_sparse = np.load('data_train/img006_Probabilities.npy')
ys_unet_sparse_0 = np.load('data_predict/ys_predict.npy')
ys_unet_sparse_1 = np.load('data_predict/ys_predict_1.npy')
ys_unet_dense_0 = np.load('data_predict/ys_unet_dense_0.npy')
ys_unet_dense_1 = np.load('data_predict/ys_predict_1_dense.npy')
lab6_sparse = np.load('data_train/labels006.npy')
lab6_sparse = lab6_sparse[0,...,0]
mask_sparse = lab6_sparse != 0
lab6_dense = np.load('data_train/lab6_dense.npy')
ys_avgd = np.load('ys_avgd.npy')

# from unet import my_categorical_crossentropy
# f_cross_ent = my_categorical_crossentropy(weights=[1/3,1/3,1/3], itd=0, BEnd=np)

ys = lab6_dense.copy()
# permute labels so as to be consistent with previous classifiers
ys[ys>1]  = 3
ys[ys==1] = 2
ys[ys==3] = 1
ys = np_utils.to_categorical(ys).reshape(ys.shape + (-1,))

def numpy_ce(y_true, y_pred, weights=(1/3, 1/3, 1/3), axis=0):
    ce = y_true * np.log(y_pred + K.epsilon())
    ce = np.mean(ce, axis=axis)
    result = weights * ce
    result = -np.sum(result)
    return result

print("DENSE GT")
print("RF Sparse")
print(numpy_ce(ys, ys_rf_sparse[0,:10], axis=(0,1,2)))
print("Unet Sparse")
print(numpy_ce(ys, ys_unet_sparse_0[:10], axis=(0,1,2)))
print("Unet Dense")
print(numpy_ce(ys, ys_unet_dense_0[:10], axis=(0,1,2)))

ys = lab6_sparse.copy()
ys = np_utils.to_categorical(ys).reshape(ys.shape + (-1,))
# permute labels so as to be consistent with previous classifiers
# ys[mask][:,[1,2,3]]
print("SPARSE GT")
y_true = ys[mask_sparse][:,[1,2,3]]
print("RF Sparse")
print(numpy_ce(y_true, ys_rf_sparse[0][mask_sparse]))
print("Unet Sparse")
print(numpy_ce(y_true, ys_unet_sparse_0[mask_sparse]))
print("Unet Dense")
print(numpy_ce(y_true, ys_unet_dense_0[mask_sparse]))

def accuracy(y_true, y_pred):
  nclasses = y_pred.shape[-1]
  y_true = y_true.argmax(-1)
  y_pred = y_pred.argmax(-1)
  table = []
  for i in range(nclasses):
    m = y_true==i
    v = [(y_pred[m]==j).sum() for j in range(nclasses)]
    table.append(v)
  print("Confusion")
  print(tabulate(table))
  mat = np.array(table)
  # dimension 0 is True class, dimension 1 is predicted class
  print("Accuracy per predicted class")
  print(mat / mat.sum(0, keepdims=True))
  print("Accuracy per true class")
  print(mat / mat.sum(1, keepdims=True))

# Print confusion matricies for each case
accuracy(y_true, ys_rf_sparse[0][mask_sparse])
accuracy(y_true, ys_unet_sparse_0[mask_sparse])
accuracy(y_true, ys_unet_dense_0[mask_sparse])

ys = lab6_dense.copy()
# permute labels so as to be consistent with previous classifiers
ys[ys>1]  = 3
ys[ys==1] = 2
ys[ys==3] = 1
ys = np_utils.to_categorical(ys).reshape(ys.shape + (-1,))

# Print confusion matricies for each case
accuracy(ys, ys_rf_sparse[0,:10])
accuracy(ys, ys_unet_sparse_0[:10])
accuracy(ys, ys_unet_dense_0[:10])

