from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.layers import Input, Conv2D, Concatenate, Lambda
from keras.models import Model
from keras.utils import Sequence
from keras.optimizers import Adam
from deeptools.blocks import unet_block
from tqdm import tqdm

from stardist.utils import stardist_from_labels, add_boundary_label, onehot_encoding, edt_softmask





def plot_history(hist,*keys,n_rows=2,figsize=(18,7)):
    import matplotlib.pyplot as plt
    n_rows = len(keys)
    n_cols = max([len(k) for k in keys])

    fig,ax = plt.subplots(n_rows,n_cols, figsize=figsize)
    if n_rows == 1:
        ax = np.expand_dims(ax,0)
    # if n_cols == 1:
    #     ax = np.expand_dims(ax,-1)

    for j in range(n_rows):
        for i,k in enumerate(keys[j]):
            ax[j,i].plot(hist.epoch,hist.history[k],label=k);
            if 'val_'+k in hist.history:
                ax[j,i].plot(hist.epoch,hist.history['val_'+k],label='val_'+k);
            ax[j,i].legend()

def masked_loss_laplace(mask, eps=1e-3, relative=False):
    C = np.log(2.0)
    if relative:
        def nll_laplace(d_true, d_pred):
            mu    = d_pred[...,:-1]
            sigma = d_pred[...,-1:] + eps
            return K.mean(mask * (K.abs(((mu-d_true)/(d_true+1e-3))/sigma) + K.log(sigma) + C), axis=-1)
    else:
        def nll_laplace(d_true, d_pred):
            mu    = d_pred[...,:-1]
            sigma = d_pred[...,-1:] + eps
            return K.mean(mask * (K.abs((mu-d_true)/sigma) + K.log(sigma) + C), axis=-1)
    return nll_laplace

def masked_loss(mask, penalty, relative=False):
    assert not relative, "dont use at the moment"
    if relative:
        def _loss(d_true, d_pred):
            return K.mean(mask * penalty((d_pred - d_true)/(d_true+1e-3)), axis=-1)
    else:
        def _loss(d_true, d_pred):
            return K.mean(mask * penalty(d_pred - d_true), axis=-1)
    return _loss

def masked_loss_mae(mask, relative=False):
    return masked_loss(mask, K.abs, relative)

def masked_loss_mse(mask, relative=False):
    return masked_loss(mask, K.square, relative)

def starpoly_area(radii):
    dphi = (2*np.pi) / tf.to_float(tf.shape(radii)[-1])
    # equivalent to: radii_rolled = np.roll(radii,1)
    radii_rolled = tf.concat((radii[...,-1:],radii[...,:-1]),axis=-1)
    return 0.5*tf.sin(dphi)*tf.reduce_sum(radii*radii_rolled,axis=-1,keep_dims=True)


def define_baseline(n_classes, input_shape=(None,None,1), **unet_kwargs):
    inp_X = Input(input_shape,name='X')
    unet = unet_block(**unet_kwargs)(inp_X)
    # unet = Conv2D(128,(3,3),name='features',padding='same',activation='relu')(unet)
    oup  = Conv2D(n_classes,(1,1),name='Y',padding='same',activation='softmax')(unet)
    return Model(inp_X,oup)

def compile_baseline(model, lr):
    model.compile(Adam(lr=lr), loss='categorical_crossentropy', metrics=['accuracy'])

def predict_baseline(model, X, **predict_kwargs):
    if X.shape[-1] != 1:
        X = np.expand_dims(X,-1)
    Y = model.predict(X,**predict_kwargs)
    return Y


def define_stardist(n_rays=32, input_shape=(None,None,1), **unet_kwargs):
    inp_X    = Input(input_shape,name='X')
    inp_mask = Input(input_shape,name='dist_mask')

    unet = unet_block(**unet_kwargs)(inp_X)
    unet = Conv2D(128,(3,3),name='features',padding='same',activation='relu')(unet)

    oup_dmap  = Conv2D(1,     (1,1),name='dmap',padding='same',activation='sigmoid')(unet)
    oup_dist  = Conv2D(n_rays,(1,1),name='dist',padding='same',activation='linear')(unet) # relu activation would be better, but sometimes causes problems (radii value stuck at 0)
    return Model([inp_X,inp_mask],[oup_dmap,oup_dist])

def compile_stardist(model, lr, dist_loss='mae'):
    assert dist_loss in ('mae','mse')
    _dist_loss = masked_loss_mae if dist_loss=='mae' else masked_loss_mse
    inp_mask = model.input[1] # second input layer is mask for dist loss
    model.compile(Adam(lr=lr), loss=['binary_crossentropy',_dist_loss(inp_mask)])

def predict_stardist(model, X, **predict_kwargs):
    if X.shape[-1] != 1:
        X = np.expand_dims(X,-1)
    dummy = np.empty(X.shape[:-1]+(1,), dtype=np.float32)
    dmap, dist = model.predict([X,dummy],**predict_kwargs)
    dmap = dmap[...,0]
    return dmap, dist


class StardistData(Sequence):

    def __init__(self, X, Y_lbl, Y_lbl_cleared, Y_dmap, Y_dmap_cleared, batch_size, n_rays, crop=None):
        self.X, self.Y_lbl, self.Y_lbl_cleared, self.Y_dmap, self.Y_dmap_cleared = X, Y_lbl, Y_lbl_cleared, Y_dmap, Y_dmap_cleared
        self.batch_size = batch_size
        self.n_rays = n_rays
        self.perm = np.random.permutation(len(self.X))
        self.crop = crop

        if crop is not None:
            s = slice(crop,-crop), slice(crop,-crop)
            self.Y_dmap         = np.empty_like(Y_dmap[(slice(None),)+s])
            self.Y_dmap_cleared = np.empty_like(Y_dmap[(slice(None),)+s])
            for i,lbl in tqdm(enumerate(Y_lbl),        total=len(Y_lbl),desc='StardistData dmap         '):
                self.Y_dmap[i] = edt_softmask(lbl[s])
            for i,lbl in tqdm(enumerate(Y_lbl_cleared),total=len(Y_lbl),desc='StardistData dmap_cleared '):
                self.Y_dmap_cleared[i] = edt_softmask(lbl[s])

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X))

    def __getitem__(self, i):
        idx = slice(i*self.batch_size,(i+1)*self.batch_size)
        idx = self.perm[idx]
        X, Y_lbl, Y_lbl_cleared, Y_dmap, Y_dmap_cleared = self.X[idx], self.Y_lbl[idx], self.Y_lbl_cleared[idx], self.Y_dmap[idx], self.Y_dmap_cleared[idx]
        Y_dist = np.stack([stardist_from_labels(lbl,self.n_rays) for lbl in Y_lbl_cleared])

        X = np.expand_dims(X,-1)
        Y_dmap_cleared = np.expand_dims(Y_dmap_cleared,-1)
        Y_dmap = np.expand_dims(Y_dmap,-1)

        if self.crop is not None:
            s = slice(None), slice(self.crop,-self.crop), slice(self.crop,-self.crop), slice(None)
            X,Y_dist = X[s],Y_dist[s]

        # print(idx)
        # print(list(map(np.shape,[X, Y_lbl, Y_lbl_cleared, Y_dmap, Y_dmap_cleared, Y_dist])))
        # assert Y_dist.dtype == np.float32
        # assert Y_dist.shape[:-1] == X.shape[:-1]
        # assert Y_dist.shape[-1] == self.n_rays
        # print(i)


        return [X,Y_dmap_cleared], [Y_dmap,Y_dist]

class BaselineData(Sequence):

    def __init__(self, X, Y_lbl, batch_size, n_classes):
        assert n_classes in (2,3)
        self.X, self.Y_lbl = X, Y_lbl
        self.batch_size = batch_size
        self.n_classes = n_classes
        self.perm = np.random.permutation(len(self.X))

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X))

    def __getitem__(self, i):
        idx = slice(i*self.batch_size,(i+1)*self.batch_size)
        idx = self.perm[idx]
        X, Y_lbl = self.X[idx], self.Y_lbl[idx]

        if self.n_classes == 2:
            Y = np.stack([onehot_encoding(lbl > 0,                 self.n_classes) for lbl in Y_lbl])
        else:
            Y = np.stack([onehot_encoding(add_boundary_label(lbl), self.n_classes) for lbl in Y_lbl])

        X = np.expand_dims(X,-1)
        return X, Y

class StardistRefinementData(Sequence):

    def __init__(self, X, Y_lbl, Y_lbl_stardist, batch_size, labels_per_image=1, fg_weight=1, same_labels_per_image=False):
        self.X, self.Y_lbl, self.Y_lbl_stardist = X, Y_lbl, Y_lbl_stardist
        self.batch_size = batch_size
        self.perm = np.random.permutation(len(self.X))
        self.labels_per_image = labels_per_image
        self.fg_weight = fg_weight
        self.same_labels_per_image = same_labels_per_image

    def __len__(self):
        return int(np.ceil(len(self.X) / float(self.batch_size)))

    def on_epoch_end(self):
        self.perm = np.random.permutation(len(self.X))

    def __getitem__(self, i):
        idx = slice(i*self.batch_size,(i+1)*self.batch_size)
        idx = self.perm[idx]
        X, Y_lbl, Y_lbl_stardist = self.X[idx], self.Y_lbl[idx], self.Y_lbl_stardist[idx]

        _X,_Y_true,_Y_pred,_Y_pred_other = [],[],[],[]
        for x, y_true, y_pred in zip(X,Y_lbl,Y_lbl_stardist):
            labels = tuple( set(np.unique(y_true)) - set((0,)) )
            n_labels = len(labels)
            if self.same_labels_per_image:
                rng = np.random.RandomState(n_labels)
            else:
                rng = np.random
            ind = rng.choice(labels, size=self.labels_per_image, replace=self.labels_per_image>n_labels)
            for j in ind:
                _X.append(x)
                _Y_true.append(y_true==j)
                _Y_pred.append(y_pred==j)
                _Y_pred_other.append((y_pred!=0) & (y_pred!=j))

        _X, _Y_true, _Y_pred, _Y_pred_other =       np.stack(_X),          np.stack(_Y_true),          np.stack(_Y_pred),          np.stack(_Y_pred_other)
        _X, _Y_true, _Y_pred, _Y_pred_other = np.expand_dims(_X,-1), np.expand_dims(_Y_true,-1), np.expand_dims(_Y_pred,-1), np.expand_dims(_Y_pred_other,-1)

        if self.fg_weight != 1:
            _Y_true = _Y_true.astype(np.float32)

        return [_X, _Y_pred, _Y_pred_other], _Y_true * self.fg_weight

