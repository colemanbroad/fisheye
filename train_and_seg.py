# %load_ext autoreload
# %autoreload 2
from ipython_remote_defaults import *

from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard, ReduceLROnPlateau

import unet
import lib as ll
import augmentation
from segtools import lib as seglib
from segtools import segtools_simple as ss
from scipy.ndimage.filters import convolve


QUICK = False #True

## note: all paths are relative to project home directory
mypath = Path("training/t018/")
mypath.mkdir(exist_ok=True)
myfile = Path(__file__)
print(mypath / myfile.name)
shutil.copy(myfile, mypath / 'train_and_seg.py')
sys.stdout = open(mypath / 'stdout.txt', 'w')
sys.stderr = open(mypath / 'stderr.txt', 'w')

## load data
img = imread('data/img006.tif')
lab = imread('data/labels_lut.tif')
lab = lab[:,:,0]
lab = ll.condense_labels(lab)
mask_train_slices = lab.max((2,3)) > 0
ys = lab[mask_train_slices].copy()
xs = ll.labeled_slices_to_xsys(img, mask_train_slices, axes="TZCYX")

# np.savez(mypath / 'traindat', xs=xs, ys=ys)
np.savez(mypath / 'traindat_small', xs=xs[::50], ys=ys[::50])

sys.exit(0)

print(xs.shape,ys.shape,mask_train_slices.shape)

# define classweights
n_channels = xs.shape[-1]
n_classes  = len(np.unique(ys))
classweights = [1/n_classes,]*n_classes

# convert ys to categorical
ys = np_utils.to_categorical(ys).reshape(ys.shape + (-1,))

# split 400x400 into 16x100x100 patches
nz,ny,nx,nc = xs.shape
ny4,nx4 = ny//4, nx//4
xs = xs.reshape((nz,4,ny4,4,nx4,nc))
xs = perm(xs,"z1y2xc","z12yxc")
xs = xs.reshape((-1,ny4,nx4,nc))
nz,ny,nx,nc = ys.shape
ys = ys.reshape((nz,4,ny4,4,nx4,nc))
ys = perm(ys,"z1y2xc","z12yxc")
ys = ys.reshape((-1,ny4,nx4,nc))

## turn off the `ignore` class
mask = ys[...,3]==1
ys[mask] = 0

## reweight border pixels
a,b,c,d = ys.shape
distimg = np.zeros((a,b,c), dtype=np.float16)
mask = ys[...,2]==1
distimg[~mask] = 1
distimg = np.array([distance_transform_edt(d) for d in distimg])
distimg = np.exp(-distimg/10)
distimg = distimg/distimg.mean((1,2), keepdims=True)
ys = ys*distimg[...,np.newaxis]

print(ys.max((0,1,2)))

## normalize
xs = xs/xs.mean((1,2), keepdims=True)

## shuffle
inds = np.arange(xs.shape[0])
np.random.shuffle(inds)
invers = np.argsort(np.arange(inds.shape[0])[inds])
xs = xs[inds]
ys = ys[inds]
if QUICK:
  xs = xs[:7*6]
  ys = ys[:7*6]

## train vali split
split = 7
n_vali = xs.shape[0]//split
xs_train = xs[:-n_vali]
ys_train = ys[:-n_vali]
xs_vali  = xs[-n_vali:]
ys_vali  = ys[-n_vali:]

net = unet.get_unet_n_pool(n_pool=2,
                            inputchan=n_channels,
                            n_classes=n_classes,
                            n_convolutions_first_layer=32,
                            dropout_fraction=0.2)

# net.load_weights('unet_model_weights_checkpoint.h5')

optim = Adam(lr=1e-4)
loss  = unet.my_categorical_crossentropy(weights=classweights, itd=4)
net.compile(optimizer=optim, loss=loss, metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=str(mypath / "w001.h5"), verbose=0, save_best_only=True, save_weights_only=True)
earlystopper = EarlyStopping(patience=30, verbose=0)
reduce_lr    = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
callbacks = [checkpointer, earlystopper, reduce_lr]

def random_augmentation_xy(xpatch, ypatch, train=True):
  # if random.random()<0.5:
  #     xpatch = np.flip(xpatch, axis=1)
  #     ypatch = np.flip(ypatch, axis=1)
  # if random.random()<0.5:
  #     xpatch = np.flip(xpatch, axis=0)
  #     ypatch = np.flip(ypatch, axis=0)
  # if random.random()<0.5:
  #     randangle = (random.random()-0.5)*60 # even dist between ± 30
  #     xpatch  = rotate(xpatch, randangle, reshape=False, mode='reflect')
  #     ypatch  = rotate(ypatch, randangle, reshape=False, mode='reflect')
  if train:
    if random.random()<0.5:
        delta = np.random.normal(loc=0, scale=5, size=(2,3,3))
        xpatch = augmentation.warp_multichan(xpatch, delta=delta)
        ypatch = augmentation.warp_multichan(ypatch, delta=delta)
    if random.random()<0.1:
        m = random.random()*xpatch.mean() # channels already normed
        s = random.random()*xpatch.std()
        noise = np.random.normal(m/4,s/4,xpatch.shape).astype(xpatch.dtype)
        xpatch += noise
  xpatch = xpatch.clip(min=0)
  xpatch = xpatch/xpatch.mean((0,1))
  return xpatch, ypatch

batchsize = 3
stepsperepoch = xs_train.shape[0] // batchsize
validationsteps = xs_vali.shape[0] // batchsize

f_trainaug = lambda x,y:random_augmentation_xy(x,y,train=False)

bgen = unet.batch_generator_patches_aug(xs_train, ys_train,
                                  # steps_per_epoch=100,
                                  batch_size=batchsize,
                                  augment_and_norm=f_trainaug,
                                  savepath=mypath)


f_valiaug = lambda x,y:random_augmentation_xy(x,y,train=False)

vgen = unet.batch_generator_patches_aug(xs_vali, ys_vali,
                                  # steps_per_epoch=100,
                                  batch_size=batchsize,
                                  augment_and_norm=f_valiaug,
                                  savepath=None)

history = net.fit_generator(bgen,
                  steps_per_epoch=stepsperepoch,
                  epochs=20,
                  verbose=1,
                  callbacks=callbacks,
                  validation_data=vgen,
                  validation_steps=validationsteps)

net.save_weights(mypath / 'w002.h5')

print(history.history, file=open(mypath / 'history.txt','w'))
def plot_hist_key(k):
  plt.figure()
  y1 = history.history[k]
  y2 = history.history['val_' + k]
  plt.plot(y1, label=k)
  plt.plot(y2, label='val_' + k)
  plt.legend()
  plt.savefig(mypath / (k + '.png'))
plot_hist_key('loss')
plot_hist_key('acc')

## reqd: img, QUICK, 



## Predictions!

x = perm(img,"tzcyx","tzcyx")
a,b,c,d,e = x.shape
if QUICK:
  x = x[[9],:b//4] # downsize for QUICK testing

a,b,c,d,e = x.shape
x = x.reshape((a*b,c,d,e))
x = x/x.mean((1,2), keepdims=True)
x.shape
pgen = unet.batch_generator_pred_zchannel(x, batch_size=1)
x3 = np.stack([x2 for x2 in pgen])
x3 = x3[:,0]
# pimg = net.predict_generator(pgen, steps=x.shape[0], verbose=1)
pimg = net.predict(x3)
pimg = pimg.astype(np.float16)
pimg = pimg[np.newaxis,...]
a,b,c,d,e = pimg.shape
# pimg = pimg.reshape((a,b,c,d,pimg.shape[-1]))
np.save(mypath / 'pimg', pimg)

## max division channel across z

np.save(mypath / 'max_z_divchan', pimg[:,...,4].max(1))

## find divisions

def find_divisions():
  x = pimg[:,:,:,:,:]
  a,b,c,d,e = x.shape
  x = x.reshape((a,b,c,d,e))
  dz = 6 ## downsample z
  div = x[:,::dz,...,4].sum((2,3))

  val_thresh = np.percentile(div.flat, 95)

  n_rows_max = 7 
  tz = np.argwhere(div > val_thresh)[:n_rows_max]

  n_cols = 7 ## n columns
  lst = list(range(x.shape[0]))
  x2 = x[:,::dz]
  x2 = np.array([x2[timewindow(lst, n[0], n_cols), n[1]] for n in tz])
  a,b,c,d,e = x2.shape
  x2.shape
  x2 = perm(x2,'12yxc','1y2xc')
  a,b,c,d,e = x2.shape
  x2 = x2.reshape((a*b,c*d,e))
  x2 = x2[::4,::4,[4,1,2]]
  x2[...,0] *= 40 ## enhance division channel color!
  x2 = np.clip(x2, 0, 1)
  io.imsave(mypath / 'find_divisions.png', x2)

find_divisions()

## quit early if QUICK

if QUICK:
  sys.exit(0)

## compute instance segmentation statistics

BLUR = False
if BLUR:
  weights = np.full((3, 3, 3), 1.0/27)
  pimg = [[convolve(pimg[t,...,c], weights=weights) for c in range(pimg.shape[-1])] for t in range(pimg.shape[0])]
  pimg = np.array(pimg)
  pimg = pimg.transpose((0,2,3,4,1))
  pimg = [[convolve(pimg[t,...,c], weights=weights) for c in range(pimg.shape[-1])] for t in range(pimg.shape[0])]
  pimg = np.array(pimg)
  pimg = pimg.transpose((0,2,3,4,1))

def f(x):
  x1 = x[...,1] # nuclei
  x2 = x[...,2] # borders
  mask = (x1 > 0.5) #& (x2 < 0.15)
  res  = watershed(1-x1,label(x1>0.8)[0], mask = mask)
  # res = label(mask)[0]
  return res
hyp = np.array([f(x) for x in pimg])
hyp = hyp.astype(np.uint16)
np.save(mypath / 'hyp', hyp)

y_pred_instance = hyp[mask_train_slices]

def f(x):
  x[x!=1] = 0
  x = label(x)[0]
  return x
y_gt_instance = np.array([f(x) for x in traindata[:,0].copy()])

res = [ss.seg(x,y) for x,y in zip(y_gt_instance, y_pred_instance)]
print({'mean' : np.mean(res), 'std' : np.std(res)}, file=open(mypath / 'SEG.txt', 'w'))

nhls = seglib.labs2nhls(hyp, img[:,:,1], simple=False)
pickle.dump(nhls, open(mypath / 'nhls.pkl', 'wb'))

plt.figure()
cm = sns.cubehelix_palette(len(nhls))
for i,nhl in enumerate(nhls):
  areas = [n['area'] for n in nhl]
  areas = np.log2(sorted(areas))
  plt.plot(areas, '.', c=cm[i])
plt.savefig(mypath / 'cell_sizes.pdf')

#hi coleman! keep coding!
#best, romina :)


