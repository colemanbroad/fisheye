from segtools.defaults.ipython_remote import *

homedir = Path('./')
loaddir = Path('training/test/')
savedir = Path('training/test/')
mypath_opt = savedir

rawdata = ts.load_rawdata('./')
trainable = ts.build_trainable(rawdata)
history = ts.train(trainable, savedir)
ts.plot_history(history, savedir)

## load weights after network optimization

trainable['net'].load_weights(loaddir / 'w001.h5')
ts.predict_train_vali(trainable, savedir)
pimg = ts.predict_on_new(trainable['net'], rawdata['img'])
## look at pimg results
ts.max_z_divchan(pimg)
ts.show_results(pimg, rawdata, savedir)
ts.find_divisions(pimg, savedir)

segparams = ts.segparams()
best = ts.optimize_segmentation(pimg, rawdata, segparams, mypath_opt)
hyp = np.array([segparams['function'](x, best) for x in pimg])
seg_scores = ts.compute_seg_on_slices(hyp, rawdata)
ts.analyze_hyp(hyp, rawdata, segparams, savedir)

nhls = ts.build_nhl(hyp, rawdata, savedir)
tr = ts.track_nhls(nhls, savedir)