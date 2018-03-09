import numpy as np
import sys
sys.path.insert(0, '/Users/colemanbroaddus/Desktop/Projects/nucleipix/')

import skimage.io as io
import matplotlib.pyplot as plt
plt.ion()
from scipy.ndimage import zoom, label
from scipy.ndimage.filters import convolve1d
from scipy.signal import gaussian
from scipy.ndimage.morphology import binary_dilation

import pandas
from skimage.morphology import watershed
from segtools import voronoi

import gputools
import spimagine

from segtools import cell_view_lib as view
from segtools import lib

from keras.utils import np_utils

img6  = io.imread('data_train/img006.tif')
img6  = img6.transpose((0,1,3,4,2))
img6p = np.load('data_train/img006_Probabilities.npy')
ys_predict = np.load('/Volumes/projects/project-broaddus/fisheye/data_predict/ys_predict.npy')
ys_predict_1 = np.load('/Volumes/projects/project-broaddus/fisheye/data_predict/ys_predict_1.npy')
ys_predict_0_dense = np.load('training/t002/ys_predict_0_dense.npy')
ys_predict_1_dense = np.load('/Volumes/projects/project-broaddus/fisheye/data_predict/ys_predict_1_dense.npy')
lab6_sparse = np.load('data_train/labels006.npy')
lab6_sparse = lab6_sparse[0,...,0]
mask = lab6_sparse != 0
lab6_dense = np.load('lab6_dense.npy')

def seg1():
  x  = img6p[1,...,1]
  hx = gaussian(21, 3.0)
  x  = gputools.convolve_sep3(x, hx, hx, hx)
  x  = lib.normalize_percentile_to01(x, 0, 100)
  nhl,hyp  = lib.two_var_thresh(x, c1=0.4, c2=0.9)
  return nhl,hyp
nhl,hyp = seg1()

def seg2():
  # potential = 1 - img6p[1,...,1] + img6p[1,...,0]
  seeds = np.zeros_like(potential, dtype=np.int)
  seeds[[*points.T]] = np.arange(points.shape[0]) + 1
  hyp = watershed(potential, seeds, mask=potential<0.5)
  nhl = lib.hyp2nhl(hyp)
  return hyp, nhl
hyp,nhl = seg2()

def show():
  x0 = img6[1,...,0]
  x1 = img6[1,...,1]
  x2 = img6p[1,...,0]*x1.max()
  iss = view.ImshowStack(np.stack([x0,x1,x2], axis=-1), w=0, colorchan=True)
  return iss
iss = show()

def showseg():
  boundaryimg = voronoi.lab2binary_neibs3d(hyp2)
  imgcopy = img6[1,...,1].copy()
  imgcopy[boundaryimg<5] = imgcopy.max()*1.5
  iss = view.ImshowStack(imgcopy, w=0)
  return iss
iss = showseg()

def try_many_thresh_seg():
  t = np.linspace(0.95, .999, 30)
  cpal = sns.cubehelix_palette(30)
  f1 = plt.figure(1)
  f2 = plt.figure(2)
  x = img6[0,...,1]
  def compute(i):
    th = t[i]
    seg = label(x>th)[0]
    n = seg.max()
    f1.gca().plot(th, n, 'o', c=cpal[i])
    ids,cts = np.unique(seg, return_counts=True)
    f2.gca().plot(np.log2(sorted(cts)), c=cpal[i])
  for i in range(len(t)):
    compute(i)
try_many_thresh_seg()

def select_objects(hyp):
  labels, sizes = np.unique(hyp, return_counts=True)  
  inds = np.argsort(sizes)
  sizes = sizes[inds]
  sizes = np.log2(sorted(sizes))
  labels = labels[inds]  
  fig, ax = plt.subplots()
  col = ax.scatter(np.arange(len(sizes)), sizes, s=20) #, c=cpal[i])
  selector = view.SelectFromCollection(ax, col)
  input('Press Enter to accept selected points')
  print("Selected points:")
  print(selector.ind, sizes[selector.ind], labels[selector.ind])
  return labels[selector.ind]

  # selector.disconnect()
  # mask = lib.mask_labels(labels[selector.ind], hyp)
  # w.glWidget.dataModel[0][...] = img6[1,...,1].astype('float')
  # w.glWidget.dataModel[0][mask] *= 5.0
  # w.glWidget.dataPosChanged(0)
  # select_objects(hyp)
inds = select_objects(lab6_nuc_labeled)

for i in range(lab6_nuc_labeled.shape[0]):
  img = lab6_nuc_labeled[i]
  ind,cts = np.unique(img, return_counts=True)
  memid = ind[cts.argmax()]
  img[img==memid] = 1
  

def curate():
  pp = 140
  img_p = lib.pad_img(img6[1,...,1].astype('float'), pad=pp//2)
  hyp_p = lib.pad_img(hyp, pad=pp//2)
  nhl_p = lib.pad_nhl(nhl, pad=pp//2)
  ss = (slice(0,pp),slice(0,pp),slice(0,pp))
  # w = spimagine.volshow([img_p[ss], hyp_p[ss]], interpolation='nearest')
  anno = lib.curate_nhl(w, nhl_p[-5:], img_p, hyp_p, pp=pp)
  return w, anno
w, anno = curate()

def highlight():
  img = img6[1,...,1].copy().astype('float')
  img[hyp==nhl[-1]['label']] *= 3.5
  lib.update_spim(w, 1, img)
highlight()

def replace_old_centers(pts):
  pts = np.array(pts)
  current_map = hyp[[*points.T]]
  replace = hyp[[*pts.T]]
  good_indices = set(np.arange(points.shape[0]) + 1) - set(replace)
  points2 = np.concatenate([points[np.array(list(good_indices)) - 1], pts], axis=0)
  return points2
points2 = points.copy()
points = replace_old_centers(newcents[-2:])

def split_up_cell(pts):
  pts = np.array(pts)
  current_map = hyp[[*points.T]]
  replace = hyp[[*pts.T]]
  mask = lib.mask_labels(replace, hyp)
  seeds = np.zeros_like(potential, dtype=np.int)
  seeds[[*pts.T]] = np.arange(pts.shape[0]) + 1
  hyp2 = watershed(potential, seeds, mask=mask)
  hyp2 += hyp.max()
  hyp[mask] = hyp2[mask]
split_up_cell(newcents)

iss = view.ImshowStack([img6[1,...,0].copy(), hyp, ys_predict_1[...,1]])

def onclick(event):
    xi, yi = int(event.xdata + 0.5), int(event.ydata + 0.5)
    zi = iss.idx[1]
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' % (event.button, event.x, event.y, event.xdata, event.ydata))
    print(zi, yi, xi)
    print(event.key)
    # self.centerpoints.append([zi,yi,xi])
    w.glWidget.dataModel[0][...] = img6[1,...,1].astype('float')
    w.glWidget.dataModel[0][hyp==hyp[zi,yi,xi]] *= 1.8
    w.glWidget.dataPosChanged(0)
iss.fig.canvas.mpl_connect('button_press_event', onclick)


#### Fix the ground truth by removing the large blocks from the BG

def labnab1():
  lab6_nuc = io.imread('/Users/colemanbroaddus/Desktop/img006_borderlabels.tif')
  lab6_nuc = lab6_nuc[0,:11,...,2]
  lab6_nuc[lab6_nuc==9470] = 1
  #lab6_nuc = [label(img)]
  lab_full = np.array([label(1 - img)[0] for img in lab6_nuc])
  # view.ImshowStack(lab_full)
  for i in range(1, lab_full.shape[0]):
    lab_full[i] += lab_full[i-1].max() + 1
  for i in range(lab_full.shape[0]):
    lab = lab_full[i]
    ng = voronoi.label_neighbors(lab)
    perim = {}
    for (a,b) in ng.keys():
      perim[a] = ng[(a,b)] + perim.get(a,0)
      perim[b] = ng[(a,b)] + perim.get(b,0)
    perim = sorted(perim.items(), key=lambda x:x[1], reverse=True)
    memlabel = perim[0][0]
    bglabel = perim[1][0]
    lab[lab==memlabel] = 0
    lab[lab==bglabel] = 1
  return lab_full

labfull = labnab1()
remove_labels_at_pts(newcents, labfull)
inds = sizedist(labfull)
mask = lib.mask_labels(inds, labfull)
labfull[mask] = 1 # set small pieces to background

np.save("lab6_dense.npy", labfull[:10])

def gmmseg():
  pimg = img6p[1,...,1]
  anno = [2,2,2,3,3]
  hyp2 = lib.run_gmm(nhl[-5:], pimg, hyp, anno)
  return hyp2
hyp2 = gmmseg()


# json.dump(anno, open('small25_big25.json', 'w'))

