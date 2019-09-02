## estimate number of cell centerpoints
## TODO: introduce non-max suppression?

from segtools.defaults.training import *
from segtools.defaults.ipython import *

def pimg_2_cell_centers(pimg, a=0.2, b=1.5, n=100):
  x = np.linspace(a,b,n)
  y = [label(pimg>i)[1] for i in x]
  y = np.array(y)
  # plt.figure()
  # plt.plot(x, y)
  y1 = y[1:]-y[:-1]
  i = np.argmax(y)
  yneg = np.where(y1<0,y1,0)
  n_fused = -yneg[:i].sum()
  estimated_number_of_cells = n_fused + y[i]
  optthresh = x[i]
  lab = label(pimg>optthresh)[0]
  nhl = nhl_tools.hyp2nhl(lab)
  centroids = np.array([n['centroid'] for n in nhl])

  res = dict()
  res['thresh'] = optthresh
  res['n_cells'] = y[i]
  res['lab'] = lab
  res['nhl'] = nhl
  res['centroids'] = centroids
  return res

def all_centroids(pimgs):
  centroids = [pimg_2_cell_centers(pi)['centroids'] for pi in pimgs]
  return centroids

