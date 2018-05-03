from ipython_remote_defaults import *

from keras.optimizers import Adam, SGD
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, EarlyStopping, TensorBoard, ReduceLROnPlateau

import unet
import lib as ll
import augmentation
from segtools import lib as seglib



## build home directory to save output
mypath = Path("training/t010/")
mypath.mkdir(exist_ok=True)

## load data
img = imread('data/img006.tif')
imglut = imread('data/labels_lut.tif')
img_w_labs = np.concatenate([imglut[:,:,[0]], img], axis=2)
inds, traindata = ll.fixlabels(img_w_labs)

## arrange into xs and ys
xs_xy = traindata[:,[1,2]].copy()
ys_xy = traindata[:,0].copy()
xs = perm(xs_xy,'zcyx','zyxc')
ys = ys_xy

# define classweights
n_classes = len(np.unique(ys))
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
xs = xs[inds] # [:7*6] downsize for local testing
ys = ys[inds] # [:7*6] downsize for local testing

## train vali split
split = 7
n_vali = xs.shape[0]//split
xs_train = xs[:-n_vali]
ys_train = ys[:-n_vali]
xs_vali  = xs[-n_vali:]
ys_vali  = ys[-n_vali:]

net = unet.get_unet_n_pool(n_pool=2,
                            inputchan=2,
                            n_classes=n_classes,
                            n_convolutions_first_layer=32,
                            dropout_fraction=0.2)

# net.load_weights('unet_model_weights_checkpoint.h5')

optim = Adam(lr=1e-4)
loss = unet.my_categorical_crossentropy(weights=classweights, itd=4)
net.compile(optimizer=optim, loss=loss, metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath=str(mypath / "w001_2.h5"), verbose=0, save_best_only=True, save_weights_only=True)
earlystopper = EarlyStopping(patience=30, verbose=0)
reduce_lr    = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, verbose=1, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
callbacks = [checkpointer, earlystopper, reduce_lr]

def random_augmentation_xy(xpatch, ypatch, train=True):
  if random.random()<0.5:
      xpatch = np.flip(xpatch, axis=1)
      ypatch = np.flip(ypatch, axis=1)
  if random.random()<0.5:
      xpatch = np.flip(xpatch, axis=0)
      ypatch = np.flip(ypatch, axis=0)
  # if random.random()<0.5:
  #     delta = np.random.normal(loc=0, scale=5, size=(2,3,3))
  #     xpatch = augmentation.unet_warp_channels(xpatch, delta=delta)
  #     ypatch = augmentation.unet_warp_channels(ypatch, delta=delta)
  if random.random()<0.5:
      randangle = (random.random()-0.5)*60 # even dist between ± 30
      xpatch  = rotate(xpatch, randangle, reshape=False, mode='reflect')
      ypatch  = rotate(ypatch, randangle, reshape=False, mode='reflect')
  if train:
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
bgen = unet.batch_generator_patches_aug(xs_train, ys_train, 
                                  # steps_per_epoch=100, 
                                  batch_size=batchsize,
                                  augment_and_norm=random_augmentation_xy,
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

net.save_weights(mypath / 'w002_2.h5')

# json.dump(history.history, open(mypath / 'history.json', 'w'))
print(history.history, file=open(mypath / 'history.txt','w'))
for k,v in history.history.items():
  plt.plot(v, label=k)
plt.legend()
plt.savefig(mypath / 'traj.png')


## plot results!

# plt.plot(history.history)

## Predictions!

x = perm(img,"tzcyx","tzyxc")
a,b,c,d,e = x.shape
# x = x[[9],:b//4] # downsize for local testing
# a,b,c,d,e = x.shape
x = x.reshape((a*b,c,d,e))
x = x/x.mean((1,2), keepdims=True)
x.shape
pimg = net.predict(x)
pimg = pimg.reshape((a,b,c,d,n_classes))
np.save(mypath / 'pimg_w002', pimg)

## max division channel across z

np.save(mypath / 'max_z_divchan', pimg[:,...,4].max(1))

## find divisions

x = pimg[:,:,:,:,:]
a,b,c,d,e = x.shape
x = x.reshape((a,b,c,d,e))
div = x[:,::6,...,4].sum((2,3))

dc = np.sort(div.flat)[-20]
tz = np.argwhere(div > dc)[:7]

lst = list(range(x.shape[0]))
x2 = x[:,::6]
x2 = np.array([x2[timewindow(lst, n[0], 7), n[1]] for n in tz])
a,b,c,d,e = x2.shape
x2.shape
x2 = perm(x2,'12yxc','1y2xc')
a,b,c,d,e = x2.shape
x2 = x2.reshape((a*b,c*d,e))
np.save(mypath / 'find_divisions', x2[::4,::4,[4,1,2]])

## compute instance segmentation statistics

def f(x):
  x1 = x[...,1] # nuclei
  x2 = x[...,2] # borders
  mask = (x1 > 0.5) & (x2 < 0.05)
  # res = watershed(1-x1,label(x1>0.9)[0], mask = mask)
  res = label(mask)[0]
  return res
lab = np.array([f(x) for x in pimg])

nhls = seglib.labs2nhls(lab, img[:,:,1], simple=True)

plt.figure()
cm = sns.cubehelix_palette(len(nhls))
for i,nhl in enumerate(nhls):
  areas = [n['area'] for n in nhl]
  areas = sorted(areas)
  plt.plot(areas, c=cm[i])
plt.savefig(mypath / 'sizes.png')








