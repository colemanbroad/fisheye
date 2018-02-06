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