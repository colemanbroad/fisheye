from __future__ import print_function, unicode_literals, absolute_import, division
from ipython_remote_defaults import *
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

sys.stdout = open('iso_predict_stdout.txt', 'w')
sys.stderr = open('iso_predict_stderr.txt', 'w')

plt.switch_backend('agg')

import os
from tifffile import imread
from csbdeep.models import IsotropicCARE
from csbdeep.predict import PercentileNormalizer, PadAndCropResizer
from csbdeep.plot_utils import plot_some

# from csbdeep.utils import download_and_extract_zip_file
# download_and_extract_zip_file(
#     url = 'https://cloud.mpi-cbg.de/index.php/s/Vu0rN1G33z9hQa4/download',
#     provides = ('raw_data/retina/cropped_farred_RFP_GFP_2109175_2color_sub_10.20.tif',)
# )

# x = imread('raw_data/retina/cropped_farred_RFP_GFP_2109175_2color_sub_10.20.tif')
x = imread('data/img006.tif')
x = x[4]
axes = 'ZCYX'
subsample = 5.0
print('image size       =', x.shape)
print('image axes       =', axes)
print('subsample factor =', subsample)


plt.figure(figsize=(15,15))
plot_some(np.moveaxis(x,1,-1)[[5,-5]], title_list=[['xy slice', 'xy slice']], pmin=2,pmax=99.8);
plt.savefig('ex.png')

plt.figure(figsize=(15,15))
plot_some(np.moveaxis(np.moveaxis(x,1,-1)[:,[50,-50]],1,0), title_list=[['xz slice','xz slice']], pmin=2,pmax=99.8, aspect=subsample);
plt.savefig('ex.png')

model = IsotropicCARE(config=None, name='my_model_old')
model.load_weights()

normalizer = PercentileNormalizer(1,99.8)
resizer = PadAndCropResizer()

restored = model.predict(x, axes, subsample, normalizer, resizer)
np.save('restored', restored)

print('input  (%s) = %s' % (axes, str(x.shape)))
print('output (%s) = %s' % (axes, str(restored.shape)))
print()

plt.figure(figsize=(15,15))
plot_some(np.moveaxis(restored,1,-1)[[5,-5]], title_list=[['xy slice','xy slice']], pmin=2,pmax=99.8);

plt.figure(figsize=(15,15))
plot_some(np.moveaxis(np.moveaxis(restored,1,-1)[:,[50,-50]],1,0), title_list=[['xz slice','xz slice']], pmin=2,pmax=99.8);
