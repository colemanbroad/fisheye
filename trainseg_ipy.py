from segtools.defaults.ipython_remote import *

import train_seg_lib as ts

homedir = Path('./')
loaddir = Path('training/test/')
savedir = Path('training/test/')
mypath_opt = savedir

rawdata = ts.load_rawdata(homedir)
# trainable = ts.build_trainable(rawdata)
trainable = ts.build_trainable3D(rawdata)

## train net and plot trajectories

history = ts.train(trainable, savedir)
ts.plot_history(history, savedir)

## load weights after network optimization and view results

trainable['net'].load_weights(loaddir / 'w001.h5')
ts.predict_train_vali(trainable, savedir)
pimg = ts.predict_on_new(trainable['net'], rawdata['img'])
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

