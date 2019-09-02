from segtools.defaults.ipython_local import *
from skimage.feature import peak_local_max
from skimage.measure import regionprops
# pimg = imread('/Volumes/CSBDeep/ISBI/training_data/center_01_adaptive_median/pimg/pimg/pred001.tif')

def thcc(th):
  return np.array([r.centroid for r in regionprops(label(pimg>th)[0], pimg)])

def plm(th):
  return peak_local_max(pimg.astype(np.float32), min_distance=6, threshold_abs=th)

def plot():
  n_plm  = [plm(th).shape[0] for th in np.linspace(0,1,50)]
  n_thcc = [thcc(th).shape[0] for th in np.linspace(0,1,50)]

  plt.figure()
  plt.plot(np.linspace(0,1,50), n_plm,label='plm')
  plt.plot(np.linspace(0,1,50)[1:], n_thcc[1:],label='thcc')
  plt.semilogy()
  plt.legend()


def plot_line_profile_and_centerpoint_detection_in_peaks(img,pimg,yval):
  assert img.ndim == 2
  f2 = plt.figure()
  # f1 = plt.figure()
  f1 = imshowme(pimg)
  x = np.arange(img.shape[1])
  y = np.ones(img.shape[1])*yval

  f1.gca().plot(x,y,'k--')
  f2.gca().plot(img[yval])
  f2.gca().plot(pimg[yval])
  plm  = peak_local_max(pimg.astype(np.float32),min_distance=4, threshold_abs=0.3)
  thcc = np.array([r.centroid for r in regionprops(label(pimg>0.3)[0], pimg)])
  f1.gca().plot(plm[:,1],plm[:,0],'ro')
  f1.gca().plot(thcc[:,1],thcc[:,0],'bo')
