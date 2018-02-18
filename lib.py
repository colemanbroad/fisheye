import numpy as np

def autocorr(x):
    result = np.correlate(x, x, mode='full')
    print(result.shape, result.size)
    return result[result.size//2:]

def autocorr_multi(img):
  xs = np.arange(0, 400, 40, np.int)
  ys = np.arange(0, 400, 40, np.int)
  zs = np.arange(0, 70, 7, np.int)
  ts = np.arange(0, 11, 1, np.int)  
  x  = img[0,0,0,:,i]

def mkpoints():
  pointanno = pandas.read_csv("~/Desktop/Projects/fisheye/anno/point_anno.csv", header=0, sep=' ')
  points = pointanno[['X', 'Y', 'Slice']]
  points.X *= 10
  points.X = np.ceil(points.X)
  points.Y *= 10
  points.Y = np.ceil(points.Y)
  points = points.as_matrix().astype(np.int)
  points = points[:,[2,1,0]]
  points[:,0] -= 1 # fix stupid 1-based indexing
  return points