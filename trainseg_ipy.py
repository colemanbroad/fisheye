from segtools.defaults.ipython_remote import *
import train_seg_lib as ts
from segtools import unet

homedir = Path('./')
loaddir = Path('training/test/')
savedir = Path('training/test/') 

rawdata = ts.load_rawdata(homedir)
# trainable = ts.build_trainable2D(rawdata)
# trainable = ts.build_trainable3D(rawdata)
trainable = ts.build_trainable_dist2cen(rawdata)
ts.show_trainvali(trainable, savepath=savedir)

## custom stuff for running script

if __name__ == '__main__':
  loaddir.mkdir(exist_ok=True)
  shutil.copy(__file__, savedir)
  shutil.copy('train_seg_lib.py', savedir)
  mypath_opt = savedir

  # unet_params = {
  #   'n_pool' : 3,
  #   'inputchan' : 10,
  #   'n_classes' : 3,
  #   'n_convolutions_first_layer' : 64,
  #   'dropout_fraction' : 0.2,
  #   'kern_width' : 3,
  #   'ndim' : 2,
  # }
  # net = unet.get_unet_n_pool(**unet_params)
  # trainable['net'] = net

## train net and plot trajectories

history = ts.train(trainable, savedir)
ts.plot_history(history, savedir)

## load weights after network optimization and view results

trainable['net'].load_weights(loaddir / 'w001.h5')
ts.predict_train_vali(trainable, savedir)
pimg = ts.predict_on_new_2D(trainable['net'], rawdata['img'], trainable['xsem'])
ts.max_z_divchan(pimg, savedir)
ts.show_results(pimg, rawdata, savedir)
ts.find_divisions(pimg, savedir)

## optimize the segmentation

segparams = ts.segparams()
best = ts.optimize_segmentation(pimg, rawdata, segparams, mypath_opt)
hyp = np.array([segparams['function'](x, best) for x in pimg])
seg_scores = ts.compute_seg_on_slices(hyp, rawdata)
ts.analyze_hyp(hyp, rawdata, segparams, savedir)

nhls = ts.build_nhl(hyp, rawdata, savedir)
tr = ts.track_nhls(nhls, savedir)
