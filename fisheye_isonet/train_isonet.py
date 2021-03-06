from __future__ import print_function, unicode_literals, absolute_import, division
from segtools.defaults.ipython_remote import *
import numpy as np

import csbdeep
from csbdeep.utils import axes_dict
from csbdeep.io import load_training_data
from csbdeep.models.config import Config
from csbdeep.models.care_isotropic import IsotropicCARE
from csbdeep.utils.tf import limit_gpu_memory
from csbdeep.utils.plot_utils import plot_some, plot_history


def original():
  mypath = Path('isonet_psf_1')
  mypath.mkdir(exist_ok=True)

  # sys.stdout = open(mypath / 'train_stdout.txt', 'w')
  # sys.stderr = open(mypath / 'train_stderr.txt', 'w')

  (X,Y), (X_val,Y_val), data_axes = load_training_data(mypath / 'my_training_data.npz', validation_split=0.1)
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
  plt.savefig(mypath / 'train_1.png')

  config = Config(data_axes, n_channel_in, n_channel_out, train_epochs=200)
  print(config)
  vars(config)

  model = IsotropicCARE(config, str(mypath / 'my_model'))

  history = model.train(X,Y, validation_data=(X_val,Y_val))

  print(sorted(list(history.history.keys())))
  plt.figure(figsize=(16,5))
  plot_history(history,['loss','val_loss'],['mse','val_mse','mae','val_mae']);
  plt.savefig(mypath / 'train_history.png')

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
  plt.savefig(mypath / 'train_2.png')

  model.export_TF()

if __name__ == '__main__':
  original()