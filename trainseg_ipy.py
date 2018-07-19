from segtools.defaults.ipython_remote import *
import train_seg_lib as ts
import unet_dist2center as dc

homedir = Path('./')
loaddir = Path('training/test/')
savedir = Path('training/test/')

rawdata = dc.build_rawdata(homedir)
trainable = dc.build_trainable(rawdata)

## custom stuff for running script

if __name__ == '__main__':
  loaddir = Path(sys.argv[1])
  savedir = Path(sys.argv[2])
  print(loaddir, savedir)
  loaddir.mkdir(exist_ok=True)
  savedir.mkdir(exist_ok=True)
  # shutil.copy(__file__, savedir)
  shutil.copy('train_seg_lib.py', savedir)
  shutil.copy('unet_dist2center.py', savedir)
  mypath_opt = savedir

dc.show_trainvali(trainable, savepath=savedir)

## train net and plot trajectories

histories = []
for i in range(20):
  net = dc.build_net(trainable['xsem'], trainable['ysem'])
  history = ts.train(net, trainable, savedir, n_epochs=10, batchsize=3)
  print("HISTORY: ", i, " ", min(history.history['val_loss']))
  histories.append(history.history)

ts.plot_history(history, savedir)
json.dump(histories, open(savedir / 'histories.json', 'w'), indent=2)

## load weights after network optimization and view results

net.load_weights(loaddir / 'w001.h5')
dc.predict_trainvali(net, trainable, savedir)
pimg = dc.predict(net, rawdata['img'], trainable['xsem'], trainable['ysem'])
np.save(savedir / 'pimg', pimg)

sys.exit(0)

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
