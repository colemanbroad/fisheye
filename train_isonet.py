from __future__ import print_function, unicode_literals, absolute_import, division
from ipython_remote_defaults import *
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
# %config InlineBackend.figure_format = 'retina'

import csbdeep
from csbdeep.utils import axes_dict
from csbdeep.train import load_data
from csbdeep.models import Config, IsotropicCARE
from csbdeep.tf import limit_gpu_memory
from csbdeep.plot_utils import plot_some

sys.stdout = open('training_iso_stdout.txt', 'w')
sys.stderr = open('training_iso_stderr.txt', 'w')

(X,Y), (X_val,Y_val), data_axes = load_data('my_training_data.npz', validation_split=0.1)
ax = axes_dict(data_axes)

n_train, n_val = len(X), len(X_val)
image_size = tuple(X.shape[i] for i in ((ax['Z'],ax['Y'],ax['X']) if (ax['Z'] is not None) else (ax['Y'],ax['X'])))
n_dim = len(image_size)
n_channel_in, n_channel_out = X.shape[ax['C']], Y.shape[ax['C']]

print('number of training images:\t', n_train)
print('number of validation images:\t', n_val)
print('image size (%dD):\t\t'%n_dim, image_size)
print('Channels in / out:\t\t', n_channel_in, '/', n_channel_out)

plt.figure(figsize=(10,4))
plot_some(X_val[:5],Y_val[:5])
plt.suptitle('5 example validation patches (top row: source, bottom row: target)')
plt.savefig('c.png')

config = Config(data_axes, n_channel_in, n_channel_out, train_steps_per_epoch=25)
print(config)
vars(config)

model = IsotropicCARE(config, 'my_model')

history = model.train(X,Y, validation_data=(X_val,Y_val))

from csbdeep.plot_utils import plot_history
print(sorted(list(history.history.keys())))
plt.figure(figsize=(16,5))
plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);

model.load_weights() # load best weights according to validation loss

plt.figure(figsize=(12,7))
_P = model.keras_model.predict(X_val[:5])
if config.probabilistic:
    _P = _P[...,:(_P.shape[-1]//2)]
plot_some(X_val[:5],Y_val[:5],_P,pmax=99.5)
plt.suptitle('5 example validation patches\n'       +
             'top row: input (source),  '           +
             'middle row: target (ground truth),  ' +
             'bottom row: predicted from source')
plt.tight_layout()
plt.savefig('d.png')

model.export_TF()