from segtools.defaults.ipython import *
from segtools.defaults.training import *
import keras
from segtools import label_tools
from segtools import graphmatch
from segtools import render

import matplotlib
import matplotlib.pyplot as plt
plt.switch_backend('Agg')

# import lib

import gputools
import ipdb
import pandas as pd

from contextlib import redirect_stdout
import train_seg_lib as ts
from sklearn.metrics import confusion_matrix

cat = np.concatenate
import scipy.ndimage.morphology as morph
from sklearn.mixture import GaussianMixture

from pykdtree.kdtree import KDTree as pyKDTree

from skimage.morphology import disk
from skimage.filters import threshold_otsu, rank

import segmentation

## testing segmentations against GT

## load challenge GT data

spimdir     = "/net/fileserver-nfs/stornext/snfs2/projects/myersspimdata/"
greencarpet = "/net/fileserver-nfs/stornext/snfs4/projects/green-carpet/"

seg1 = "/net/fileserver-nfs/stornext/snfs2/projects/myersspimdata/Coleman/Celegans/ISBI/Fluo-N3DH-CE_challenge/01/SEG"
seg2 = "/net/fileserver-nfs/stornext/snfs2/projects/myersspimdata/Coleman/Celegans/ISBI/Fluo-N3DH-CE_challenge/02/SEG"

datasets = dict()
base_train = 'Fluo-N3DH-CE/'
base_chall = 'Fluo-N3DH-CE_challenge/'
datasets['t1'] = {'base':'Fluo-N3DH-CE/',          'dind':1,'dset':'train1','n':250,'z':192,'sh':(192,256,354),'osh':(35, 512, 708),'path':Path('Fluo-N3DH-CE/01/')}
datasets['t2'] = {'base':'Fluo-N3DH-CE/',          'dind':2,'dset':'train2','n':250,'z':170,'sh':(170,256,356),'osh':(31, 512, 712),'path':Path('Fluo-N3DH-CE/02/')}
datasets['c1'] = {'base':'Fluo-N3DH-CE_challenge/','dind':1,'dset':'chall1','n':190,'z':170,'sh':(170,256,356),'osh':(31, 512, 712),'path':Path('Fluo-N3DH-CE_challenge/01/')}
datasets['c2'] = {'base':'Fluo-N3DH-CE_challenge/','dind':2,'dset':'chall2','n':190,'z':170,'sh':(170,256,356),'osh':(31, 512, 712),'path':Path('Fluo-N3DH-CE_challenge/02/')}

from gputools import pad_to_shape


## load data

def labnames2imgs_cens(labnames,n):
  data = []
  for ln in labnames:
    t,z = (int(ln[-11:-8]), int(ln[-7:-4]))
    z2  = floor((z+0.5)*5.5)
    img = imread(imgnames(n)[t])
    cen = imread(cennames(n)[t])

    lab = imread(ln)
    lab = zoom(lab, (0.5,0.5), order=0)
    lab = label(lab)[0]
    
    img = gputools.scale(img, (5.5,0.5,0.5))
    img = img[...,np.newaxis]

    ## upscale centerpoints. make them just a single pixel.
    pts = np.array([n['centroid'] for n in nhl_tools.hyp2nhl(cen)]).astype(np.int)
    pts = np.floor((pts+0.5)*[5.5,0.5,0.5]).astype(np.int) 
    cen = np.zeros(img.shape[:-1])
    cen[tuple(pts.T)] = 1
    data.append({'img':img, 'cen':cen, 'lab':lab, 't':t, 'z':z, 'z2':z2, 'labname':ln})
  return data

def labnames(n):
  return sorted(glob('Fluo-N3DH-CE/{:02d}_GT/SEG/man_seg_*'.format(n)))

def imgnames(n,base='Fluo-N3DH-CE'):
  return sorted(glob(str(Path(base) / '{:02d}/t???.tif'.format(n))))

def challnames(n,base='Fluo-N3DH-CE_challenge'):
  return sorted(glob(str(Path(base) / '{:02d}/t???.tif'.format(n))))

def cennames(n,base='Fluo-N3DH-CE'):
  return sorted(glob(str(Path(base) / '{:02d}_GT/TRA/man_track???.tif'.format(n))))

def imgcenpairs(data):
  times, ns, bases = data['times'], data['ns'], data['bases']
  return [(imgnames(n,b)[t],cennames(n,b)[t]) for t,n,b in zip(times,ns,bases)]


## utils

def invertdict(lod, f=lambda x: np.array(x)):
  d2 = dict()
  for k in lod[0].keys():
    d2[k] = f([x[k] for x in lod])
  return d2

def revertdict(dol):
  res = []
  for i in range(len(list(dol.values())[0])):
    print(i)
    res.append({k:v[i] for k,v in dol.items()})
  return res

def updatekeys(d1,d2,keys):
  for k in keys:
    d1[k] = d2[k]
  return d1

def fromlocals(_locals,keys):
  d = dict()
  for k in keys:
    d[k] = _locals[k]
  return d


## normalization

def norm_szyxc(img,axs=(1,2,3)):
  mi,ma = img.min(axs,keepdims=True), img.max(axs,keepdims=True)
  img = (img-mi)/(ma-mi)
  return img

def norm_szyxc_per(img,axs=(1,2,3),pc=[2,99.9],return_pc=False):
  mi,ma = np.percentile(img,pc,axis=axs,keepdims=True)
  img = (img-mi)/(ma-mi)
  img = img.clip(0,1)
  if return_pc:
    return img,mi,ma
  else:
    return img
