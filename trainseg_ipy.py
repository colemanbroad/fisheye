from segtools.defaults.ipython_remote import *
import train_seg_lib as ts
import unet_dist2center as dc
import unet_3d_pixclass as u3
import unet_2d_pixclass as u2

homedir = Path('./')
loaddir = Path('training/u3_007/')
savedir = Path('training/u3_007/')
mypath_opt = savedir

print("HOME:", homedir.absolute())


m = u3

rawdata = m.build_rawdata(homedir)
trainable = m.build_trainable(rawdata)

## custom stuff for running script
def rm_if_exists_and_copy(p):
  p = Path(p)
  if (savedir / p).exists():
    os.remove(savedir / p)
  shutil.copy(p, savedir)

if __name__ == '__main__':
  loaddir = Path(sys.argv[1])
  savedir = Path(sys.argv[2])
  print("loaddir, savedir", loaddir, savedir)

  savedir.mkdir(exist_ok=True)
  # shutil.copy(__file__, savedir)
  for p in ['train_seg_lib.py', 'unet_dist2center.py', 'unet_2d_pixclass.py', 'unet_3d_pixclass.py']:
      rm_if_exists_and_copy(p)

m.show_trainvali(trainable, savepath=savedir)

## train net and plot trajectories

if True:
  savedir = savedir / "remove"

  val_losses = []
  for i in range(6):
    trainable = m.build_trainable(rawdata)
    savedir = savedir.parent / "t{:03d}".format(i)
    savedir.mkdir(exist_ok=True)
    net = m.build_net(trainable['xsem'], trainable['ysem'])
    history = ts.train(net, trainable, savedir, n_epochs=30, batchsize=3)
    best_val = min(history.history['val_loss'])
    val_losses.append(best_val)
    print("HISTORY: ", i, " ", best_val)
    ts.plot_history(history, savedir)
    json.dump(history.history, open(savedir / 'history.json', 'w'), indent=2, cls=NumpyEncoder)

  savedir = savedir.parent

  print("best validation losses:", val_losses, file=open(savedir/'val_losses.txt','w'))
  idx_best = np.argmin(val_losses)
  print("idx_best", idx_best)

## load weights after network optimization and view results

if True:
  net.load_weights(str(loaddir / "t{:03d}/w001.h5".format(idx_best)))
  m.predict_trainvali(net, trainable, savedir)
  pimg = m.predict(net, rawdata['img'], trainable['xsem'], trainable['ysem'])
  np.save(savedir / 'pimg', pimg)
else:
  pimg = np.load(loaddir / 'pimg.npy')

m.show_results(pimg, rawdata, trainable, savepath=savedir)

# ts.max_z_divchan(pimg, savedir)
# ts.show_results(pimg, rawdata, savedir)
# ts.find_divisions(pimg, savedir)

## optimize the segmentation

segparams = ts.segparams()
gt_patches = dict()
gt_patches['gt_slices'] = rawdata['gt_slices'][:-4]
gt_patches['inds_labeled_slices'] = rawdata['inds_labeled_slices'][:, :-4]
best = ts.optimize_segmentation(pimg[[0]], {**rawdata, **gt_patches}, segparams, mypath_opt)
hyp = np.array([segparams['function'](x, best) for x in pimg])
seg_scores = ts.compute_seg_on_slices(hyp, rawdata)
print(seg_scores)
ts.analyze_hyp(hyp, rawdata, segparams, savedir)
nhls = ts.build_nhl(hyp, rawdata, savedir)
tr = ts.track_nhls(nhls, savedir)

history = """

## Mon Jul 30 00:35:38 2018

Recently added loop over training.
Now regen trianing data and retrain model each iteration.
BUGFIX. now we should be able to train multiple models and predict from best.

3 deep 16 starting features 5 conv-width gives by far the best results.
chose best over ten models trained. The variability of the best val loss is large!
best val loss in range [0.15, 0.22] with many 0.18 and 0.19 values.
seg is 0.805 ! best so far.

*Does the variability come from the train/vali split, model initialization or SGD?*




"""