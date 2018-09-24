# import unet_dist2center as dc
# import unet_3d_pixclass as u3
# import unet_2d_pixclass as u2
# import unet_ph3regress as ph3
# import unet_dist2bg as bg
from segtools.defaults.ipython_remote import *
import train_seg_lib as ts
import unet_3d_cele as ce
savedir = Path('training/ce_test6/'); savedir.mkdir(exist_ok=True);
# homedir = Path('./')
# loaddir = Path('training/ce_022/')
# mypath_opt = savedir
m = ce


def rm_if_exists_and_copy(p):
  p = Path(p)
  if (savedir / p).exists():
    os.remove(savedir / p)
  shutil.copy(p, savedir)

if __name__ == '__main__':
  loaddir = Path(sys.argv[1])
  savedir = Path(sys.argv[2])
  mypath_opt = savedir
  print("loaddir, savedir", loaddir, savedir)

  savedir.mkdir(exist_ok=True)
  # shutil.copy(__file__, savedir)
  for p in ['train_seg_lib.py', 'unet_3d_cele.py', 'unet_dist2center.py', 'unet_2d_pixclass.py', 'unet_3d_pixclass.py']:
      rm_if_exists_and_copy(p)


# trials = m.optimize_seg_joint(savedir, max_evals=200)
m.fig3pkl()
sys.exit(0)


if False:
  m.doitall(savedir)

if False:
  imgdirs = ["Fluo-N3DH-CE/01/", "Fluo-N3DH-CE/02/", "Fluo-N3DH-CE_challenge/01/", "Fluo-N3DH-CE_challenge/02/"]
  pimgextensions = ['train1/', 'train2/', 'chall1/', 'chall2']
  # imgdirs = ["Fluo-N3DH-CE/02/", "Fluo-N3DH-CE_challenge/01/", "Fluo-N3DH-CE_challenge/02/"]
  # pimgextensions = ['train2/', 'chall1/', 'chall2']
  ws = [3,4,5,6]
  for i in range(len(pimgextensions)):
    # plt.figure()
    for t in range(29,31):
      imgdir = Path(imgdirs[i])
      pimgdir = Path("training/ce_{:03d}/train_cp/pimgs/".format(t))
      extdir = pimgdir / pimgextensions[i]
      # cellcounts = eval((extdir / 'counts.txt').open().read())
      plt.plot([c[1] for c in cellcounts], label=str(ws[t-27]))
      zcolordir = extdir / 'zcolor'
      n = 250 if i == 0 else 190
      m.pimgs2movie(extdir, imgdir, zcolordir, range(0,n))
      watersheddir = extdir / 'watershed'
      savedir = watersheddir / 'movie'
      m.hyps2movie(watersheddir,imgdir,savedir,range(0,n))
    # plt.legend()
    # plt.title(pimgextensions[i])
    # plt.savefig(str(pimgdir / 'traj{:02d}.png'.format(ws[i])))

  sys.exit(0)

## make pimgs

if True:
  imgdirs = ["Fluo-N3DH-CE/01/", "Fluo-N3DH-CE/02/", "Fluo-N3DH-CE_challenge/01/", "Fluo-N3DH-CE_challenge/02/"]
  pimgextensions = ['train1/', 'train2/', 'chall1/', 'chall2']

  net = m.build_net((120,120,120,1), 2, activation='softmax')
  net.load_weights('training/ce_059/train_cp/epochs/w_match_0.935_132.h5')
  pimgdir = Path('training/ce_059/train_cp/pimgs/')
  m.save_pimgs(net, pimgdir/'train1', Path(imgdirs[0]), range(0,250)) #challenge=False,n=1)
  m.save_pimgs(net, pimgdir/'train2', Path(imgdirs[1]), range(0,250)) #challenge=False,n=2)
  m.save_pimgs(net, pimgdir/'chall1', Path(imgdirs[2]), range(0,190)) #challenge=True,n=1)
  m.save_pimgs(net, pimgdir/'chall2', Path(imgdirs[3]), range(0,190)) #challenge=True,n=2)

  print('success')
  sys.exit(0)

## dist2bg model

if False:
  # rawdata = m.build_rawdata(homedir)

  rawgt = pickle.load(open('training/ce_test/rawgt.pkl','rb'))

  net = m.build_net((120,120,120,1),1,activation='linear')

  # lod = np.array(m.revertdict(rawgt['gt']))
  # rawgt['train'] = m.invertdict(lod[[0,3,4]])
  # rawgt['vali'] = m.invertdict(lod[[1,2]])

  tg = m.datagen(rawgt['train'], targetname='target2')
  vg = m.datagen(rawgt['vali'], targetname='target2')

  traindir = savedir / 'train_d2bg/'
  traindir.mkdir(exist_ok=True)

  examples = [(x[0],y[0,...,:-1]) for (x,y) in itertools.islice(vg, 5)]
  hical = m.Histories(examples, traindir)

  history = ts.train_gen(net, tg, vg, traindir, n_epochs=80, steps_per_epoch=30, callbacks=[hical])
  ts.plot_history(history, start=0, savepath=traindir)
  net.load_weights(history['weightname'])

  pimg = m.predict(net, rawgt['vali']['source'], outchan=1)
  res  = m.show_rawdata(rawgt['vali'],pimg,1)
  io.imsave(traindir / 'result.png', res)

  # m.optimize_seg_separate_net(net,homedir,savedir)

  print('success')
  sys.exit(0)

## render movies from a pretrained classifier

if False:
  # traindir = savedir / 'train_cp'; traindir.mkdir(exist_ok=True);
  # resultdir = traindir / 'results'; resultdir.mkdir(exist_ok=True);
  pimgdir = Path('training/ce_014/pimgs2/'); pimgdir.mkdir(exist_ok=True);
  # renderdir = pimgdir / 'figs'; renderdir.mkdir(exist_ok=True);

  net = m.build_net((120,120,120,1), 2, activation='softmax')
  net.load_weights('training/ce_012/train_cp/w001_final.h5')
  m.save_pimgs(net,pimgdir)
  # m.make_movie(net,renderdir,0,250)

  sys.exit(0)

## train and evaluate classifier

if True:
  traindir  = savedir  / 'train_cp/'; traindir.mkdir(exist_ok=True);
  resultdir = traindir / 'results/'; resultdir.mkdir(exist_ok=True);
  epochdir  = traindir / 'epochs/'; epochdir.mkdir(exist_ok=True);
  pimgdir  = traindir / 'pimgs/'; pimgdir.mkdir(exist_ok=True);

  rawdata = {'train':m.times2raw(range(0,195,10),1,homedir), 'vali':m.times2raw(range(5,195,10),1,homedir)}
  # rawdata = {'train':m.times2raw([185],1,homedir), 'vali':m.times2raw([155],1,homedir)}
  # rawdata = {'train':m.times2raw([40, 100],1,homedir), 'vali':m.times2raw([30, 105],1,homedir)}
  # rawdata = {'train':times2raw([10,20,50,100,150],1,homedir), 'vali':times2raw([25,105],1,homedir)}
  # rawdata = {'train':times2raw([10, 60, 140, 185],1,homedir), 'vali':times2raw([20, 75, 155, 175],1,homedir)}

  for i in [1,2,3]:
    res = m.show_rawdata(rawdata['vali'],i=i)
    io.imsave(traindir / 'rawdata_vali_{:02d}.png'.format(i), res)

  weights = np.array([1,1])
  weights[1] = 16.0
  weights = weights / weights.sum()
  weights = m.K.variable(weights)


  # net = m.build_net((120,120,120,1),2,activation='softmax',weights=weights)
  net = m.build_net((120,120,120,1),2,activation='softmax')

  tg = m.datagen(rawdata['train'])
  vg = m.datagen(rawdata['vali'])

  examples = [(x[0],y[0,...,:-1]) for (x,y) in itertools.islice(vg, 5)]
  hical = m.Histories(examples, epochdir, weights=weights)

  history = ts.train_gen(net, tg, vg, traindir, n_epochs=60, steps_per_epoch=20, callbacks=[hical])
  ts.plot_history(history, start=1, savepath=traindir)
  net.load_weights(history['weightname'])

  results = m.analyze_cpnet(net,rawdata,resultdir)
  pickle.dump(results, open(resultdir / 'results.pkl', 'wb'))

  rawgt = m.build_gt_rawdata(homedir)
  results_gt = m.analyze_cpnet(net,rawgt,resultdir)
  pickle.dump(results_gt, open(resultdir / 'results_gt.pkl', 'wb'))

  print('success')
  sys.exit(0)

## analysis

results_gt = pickle.load(open(resultdir / 'results_gt.pkl', 'rb'))
gtdata = m.labnames2imgs_cens(m.labnames(1),1)
m.mk_hyps_compute_seg(gtdata, results_gt)

## old training

rawdata = m.build_rawdata(homedir)
# pickle.dump(rawdata, open(savedir / 'rawdata.pkl','wb'))
trainable = m.build_trainable(rawdata)
m.show_trainvali(trainable, savedir)

net = m.build_net(trainable)
ns = trainable['vali']['xs'].shape[0]
ns = floor(np.sqrt(ns))

traindir = savedir / 'train/'
traindir.mkdir()
hical = m.Histories(trainable['vali']['xs'][::ns], trainable['vali']['ys'][::ns], traindir)

history = ts.train(net, trainable, traindir, n_epochs=40, batchsize=1,callbacks=[hical])
history = history.history
# history = eval(open(loaddir / 'history.txt','r').read())

ts.plot_history(history, savedir)
net.load_weights(history['weightname'])
m.predict_trainvali(net, trainable, savedir)
pimg = m.predict(net, rawdata['vali']['source'])
np.save(savedir / 'pimg', pimg)

sys.exit(0)
# m.show_trainvali(trainable, savepath=savedir)

## train net and plot trajectories

savedir = savedir / "remove"
val_losses = []
for i in range(10):
  trainable = m.build_trainable(rawdata)
  savedir = savedir.parent / "t{:03d}".format(i)
  savedir.mkdir(exist_ok=True)
  net = m.build_net(trainable)
  # net.load_weights('training/ph3_009/t000/w001.h5')
  traindir = savedir / 'epochs'
  traindir.mkdir()
  history = ts.train(net, trainable, traindir, n_epochs=150, batchsize=1)
  history = history.history
  best_val = min(history['val_loss'])
  val_losses.append(best_val)
  print("HISTORY: ", i, " ", best_val)
  ts.plot_history(history, savedir)
  m.predict_trainvali(net, trainable, savedir)
  json.dump(history, open(savedir / 'history.json', 'w'), indent=2, cls=NumpyEncoder)

savedir = savedir.parent

print("best validation losses:", val_losses, file=open(savedir/'val_losses.txt','w'))
idx_best = np.argmin(val_losses)
print("idx_best", idx_best)

## load weights after network optimization and view results

net.load_weights(str(loaddir / "t{:03d}/w001.h5".format(idx_best)))
m.predict_trainvali(net, trainable, savedir)
pimg = m.predict(net, rawdata['img'], trainable['xsem'], trainable['ysem'])
np.save(savedir / 'pimg', pimg)

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