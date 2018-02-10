import numpy as np
import sys
sys.path.insert(0, '/Users/colemanbroaddus/Desktop/Projects/nucleipix/')

import skimage.io as io
import matplotlib.pyplot as plt
plt.ion()
from scipy.ndimage import zoom, label
from scipy.signal import gaussian
from scipy.ndimage.morphology import binary_dilation

import gputools
import spimagine

from segtools import cell_view_lib as view
from segtools import lib

# from tifffile import TiffFile, FileHandle
# fh = FileHandle('/Volumes/myersspimdata/Mauricio/for_coleman/test_lap2b_H2BGFP_H2BRFP/20_12_17_multiview H2B_RFP&BFP_Lap2bGFP_fish6_Multiview_RIF_Subset.czi')
# fh = FileHandle('fish6_ch2.tif')

img6  = io.imread('img006.tif')
img6  = img6.transpose((0,1,3,4,2))
img6p = np.load('img006_Probabilities.npy')
# img6p = img6p.transpose((0,1,4,2,3))
# img6all = np.concatenate([img6,img6p], axis=2)
lab6 = np.load('labels006.npy').transpose((0,1,2,3,4))

# img6 = np.stack([img6[...,0], img6[...,1], img6[...,1]], axis=-1)

def seg1():
  x  = img6p[1,...,1]
  hx = gaussian(21, 3.0)
  x  = gputools.convolve_sep3(x, hx, hx, hx)
  x  = lib.normalize_percentile_to01(x, 0, 100)
  nhl,hyp  = lib.two_var_thresh(x, c1=0.4, c2=0.9)
  return nhl,hyp
nhl,hyp = seg1()

import pandas

def mkpoints():
  pointanno = pandas.read_csv("~/Desktop/point_anno.csv", header=0, sep=' ')
  points = pointanno[['X', 'Y', 'Slice']]
  points.X *= 10
  points.X = np.ceil(points.X)
  points.Y *= 10
  points.Y = np.ceil(points.Y)
  points = points.as_matrix().astype(np.int)
  points = points[:,[2,1,0]]
  points[:,0] -= 1 # fix stupid 1-based indexing
points = mkpoints()

from skimage.morphology import watershed

def seg2():
  potential = 1 - img6p[1,...,1] + img6p[1,...,0]
  seeds = np.zeros_like(potential, dtype=np.int)
  seeds[[*points2.T]] = np.arange(points2.shape[0]) + 1
  hx = gaussian(21, 2.0)
  potential  = gputools.convolve_sep3(potential, hx, hx, hx)
  potential  = lib.normalize_percentile_to01(potential, 0, 100)*2
  hyp = watershed(potential, seeds, mask=potential<0.5)
  nhl = lib.hyp2nhl(hyp)  
  return hyp, nhl
hyp,nhl = seg2()

hyp[tuple([24, 57, 365])]
hyp[tuple([17, 24, 379])]
hyp[tuple([8, 63, 376])]

from segtools import voronoi

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

def show_nhl_borders():
  # ss = lib.nuc2slices(nhl[-1], 0)
  img = img6[1,...,0].copy()
  hyp2 = lib.remove_nucs_hyp(nhl[:-2], hyp)
  # iss = view.ImshowStack(hyp2)
  # return iss
  # iss = view.ImshowStack([img, hyp2])
  # mask = lib.mask_nhl(nhl[-1], hyp)
  # mask = hyp==nhl[-1]['label']
  borders = voronoi.lab2binary_neibs3d(hyp2)
  img[borders<6] = img.max() + hyp2[borders<6]/hyp2.max()*img.max()
  iss = view.ImshowStack([img, borders])
  # iss = view.imshowme(img.mean(0))
  return iss
iss = show_nhl_borders()

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
points = points2.copy()
points2 = replace_old_centers(newcents[-2:])

iss = view.ImshowStack([img6[1,...,0].copy(), hyp, img6p[1,...,1]])




def onclick(event):
    xi, yi = int(event.xdata + 0.5), int(event.ydata + 0.5)
    zi = iss.idx[1]
    print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
          (event.button, event.x, event.y, event.xdata, event.ydata))
    print(zi, yi, xi)
    print(event.key)
    # self.centerpoints.append([zi,yi,xi])
    w.glWidget.dataModel[0][...] = img6[1,...,1].astype('float')
    w.glWidget.dataModel[0][hyp==hyp[zi,yi,xi]] *= 2.5
    w.glWidget.dataPosChanged(0)
iss.fig.canvas.mpl_connect('button_press_event', onclick)

newcents = []
def onclick_centerpoints(event):
  xi, yi = int(event.xdata + 0.5), int(event.ydata + 0.5)
  zi = iss.idx[1]
  print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
      (event.button, event.x, event.y, event.xdata, event.ydata))
  print(zi, yi, xi)
  if event.key=='C':
    print('added! ', event.key)
    newcents.append([zi,yi,xi])
iss.fig.canvas.mpl_connect('button_press_event', onclick_centerpoints)


# pts = [[34, 284, 67],
#        [228, 18, 17],
#        [205, 377, 38],]
# pts = np.array(pts)
# pts = pts[:,[2,1,0]]
# points = np.concatenate([points, pts], axis=0)

def gmmseg():
  pimg = img6p[1,...,1]
  anno = [2,2,2,3,3]
  hyp2 = lib.run_gmm(nhl[-5:], pimg, hyp, anno)
  return hyp2
hyp2 = gmmseg()


# json.dump(anno, open('small25_big25.json', 'w'))

