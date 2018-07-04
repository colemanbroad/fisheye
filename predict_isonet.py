from __future__ import print_function, unicode_literals, absolute_import, division
from segtools.defaults.ipython_remote import *
import numpy as np
import matplotlib.pyplot as plt
plt.switch_backend('agg')

import os
from tifffile import imread
from csbdeep.models import IsotropicCARE
from csbdeep.data import PercentileNormalizer, PadAndCropResizer, NoNormalizer
# from csbdeep.plot_utils import plot_some
from csbdeep.utils.plot_utils import plot_some

cat = np.concatenate
import gputools

import ipdb
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'


def original():
  mypath = Path('isonet')
  mypath.mkdir(exist_ok=True)

  # sys.stdout = open(mypath / 'predict_stdout.txt', 'w')
  # sys.stderr = open(mypath / 'predict_stderr.txt', 'w')

  # ipdb.set_trace()
  print(mypath)

  x = np.load('data/img006_noconv.npy')
  ## initial axes are TZCYX, but we move them
  x = np.moveaxis(x, 1,2)
  axes = 'CZYX'
  subsample = 5.0
  print('image size       =', x.shape)
  print('image axes       =', axes)
  print('subsample factor =', subsample)

  xy = x[:,:,[0,10,20,30,40,70]][:,[0,0,1]]
  xy[:,[0,1]] = 0.5 * xy[:,[0,1]]
  io.imsave(mypath / 'predict_xy.png', collapse(xy, [[0,3],[2,4],[1]]))
  xz = x[:,:,:,[0,10,40,50,200,300,399]][:,[0,0,1]]
  xz[:,[0,1]] = 0.5 * xz[:,[0,1]]
  io.imsave(mypath / 'predict_xz.png', collapse(xz, [[0,2],[3,4],[1]]))

  if False:
    def plot(x):
      plt.figure(figsize=(15,15))
      plot_some(np.moveaxis(x,1,-1)[[5,-5]], title_list=[['xy slice', 'xy slice']], pmin=2, pmax=99.8);
      print('predict_1')
      plt.savefig(mypath / 'predict_1.png')

      plt.figure(figsize=(15,15))
      plot_some(np.moveaxis(np.moveaxis(x,1,-1)[:,[50,-50]],1,0), title_list=[['xz slice','xz slice']], pmin=2,pmax=99.8, aspect=subsample);
      plt.savefig(mypath / 'predict_2.png')
      print('predict_2')
    ipdb.set_trace()
    plot(x[2])

  model = IsotropicCARE(config=None, name=str(mypath / 'my_model'))
  model.load_weights()

  ## normalize input everywhere
  normalizer = PercentileNormalizer(1,99.8, do_after=True)
  x = normalizer.before(x, "TCZYX")

  def norm_and_predict(x, axes):
    nonorm = NoNormalizer()
    resizer = PadAndCropResizer()
    restored = model.predict(x, axes, subsample, nonorm, resizer)
    return restored

  restored_alltime = []
  for t, xt in enumerate(x):
    restored = norm_and_predict(xt, axes)
    # np.save(mypath / 'restored{:03d}'.format(t), restored)
    print('input  (%s) = %s' % (axes, str(xt.shape)))
    print('output (%s) = %s' % (axes, str(restored.shape)))
    print()
    restored_alltime.append(restored)
  restored_alltime = np.array(restored_alltime)
  imsave(mypath / 'restored_alltime.tif', restored_alltime)

  ## intesity dist (over space, time, and channels!)
  plt.figure()
  plt.plot(np.percentile(x, np.linspace(0,100,100)))
  plt.plot(np.percentile(restored_alltime, np.linspace(0,100,100)))
  plt.savefig(mypath / 'indensity_dist.png')

  res_xy, res_xz = compare_in_out2(x, restored_alltime, subsample)
  io.imsave(mypath / 'compare_xy.png', res_xy)
  io.imsave(mypath / 'compare_xz.png', res_xz)

  return restored_alltime

def compare_in_out(arr_in, arr_out, subsample):
  "axes are TCZYX. zdim is longer in arr_out."

  rgb   = [0,1,1] # red is membrane, green and blue are nucleus
  tinds = [0,5,8]
  zinds = [0,10,40,70]
  yinds = [0,150,300,399]

  def normrgb(img):
    "channels are last dimension."
    img = img / img.mean((0,1), keepdims=True)
    return img

  def panel_xy(arr):
    xy = arr[tinds][:,:,zinds][:,rgb]
    xy = collapse(xy, [[0,3],[2,4],[1]])
    xy = normrgb(xy)
    return xy
  
  def panel_xz(arr):
    xz = arr[tinds][:,:,:,yinds][:,rgb]
    xz = collapse(xz, [[0,2],[3,4],[1]])
    xz = normrgb(xz)
    return xz

  xy_in = panel_xy(arr_in)
  xy_out = panel_xy(arr_out)
  xz_in = panel_xz(arr_in)
  xz_in = zoom(xz_in, (subsample,1,1), order=1)
  xz_out = panel_xz(arr_out)

  res_xy = cat([xy_in, xy_out], 1)
  res_xz = cat([xz_in, xz_out], 1)

  def norm(img):
    "minmax must be in [0,1) for png imsave."
    mi, ma = np.percentile(img, [2,99.5])
    img = (img-mi)/(ma-mi)
    img = img.clip(0,1)
    return img

  res_xy = norm(res_xy)
  res_xz = norm(res_xz)

  return res_xy, res_xz

def compare_in_out2(arr_in, arr_out, subsample):
  "axes are TCZYX. zdim is longer in arr_out."

  rgb   = [0,1,1] # red is membrane, green and blue are nucleus
  tinds = [0,5,8]
  zinds = [0,10,40,70]
  yinds = [0,150,300,399]

  def norm_over_chan(img, chandim=-1):
    "iverts the usual axes argument."
    axes = list(range(img.ndim))
    del axes[chandim]
    img = img / img.mean(tuple(axes), keepdims=True)
    return img
  
  ## 
  arr_in = arr_in[tinds]
  arr_out = arr_out[tinds]

  ## tczyx
  def f(x):
    # t,c = ind
    # x = arr_in[t,c]
    x = gputools.scale(x, (subsample,1,1), interpolation='linear')
    return x
  arr_in = broadcast_nonscalar_func(f, arr_in, '234')

  xy_in  = arr_in[:,:,zinds][:,rgb]
  xy_out = arr_out[:,:,zinds][:,rgb]
  xy = np.stack([xy_in, xy_out], 0)
  xy = norm_over_chan(xy, chandim=2)
  ## ITCZYX -> TY,ZIX,C
  xy = collapse(xy, [[1,4],[3,0,5],[2]])

  xz_in  = arr_in[:,:,:,yinds][:,rgb]
  xz_out = arr_out[:,:,:,yinds][:,rgb]
  xz = np.stack([xz_in, xz_out], 0)
  xz = norm_over_chan(xz, chandim=2)
  ## ITCZYX -> TZ,YIX,C
  xz = collapse(xz, [[1,3],[4,0,5],[2]])

  def norm(img):
    "minmax must be in [0,1) for png imsave."
    mi, ma = np.percentile(img, [2,99.5])
    img = (img-mi)/(ma-mi)
    img = img.clip(0,1)
    return img

  xy = norm(xy)
  xz = norm(xz)

  return xy, xz

if __name__ == '__main__':
  original()