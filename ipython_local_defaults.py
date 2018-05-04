from ipython_defaults import *

## visual stuff relying on anaconda
import matplotlib
# matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
plt.ion()
plt.switch_backend('qt5agg')
import seaborn as sns

## my local code

from segtools import cell_view_lib as view
from segtools import color
from segtools import lib as seglib
from segtools import spima
from segtools import segtools_simple as ss
from segtools import plotting
from segtools import voronoi
sys.path.insert(0,'/Users/broaddus/Desktop/Projects/')
from stackview.stackview import Stack #, StackQt
from planaria_tracking import lib as tracklib

## martin's stuff
import gputools

## martin's visual stuff
import spimagine


def qopene():
  res = run(['rsync efal:qsave.npy .'], shell=True)
  print(res)
  return np.load('qsave.npy')

def qopen():
  # import subprocess
  res = run(['rsync broaddus@falcon:qsave.npy .'], shell=True)
  print(res)
  return np.load('qsave.npy')

def updateall(w,lab):
  for i in range(lab.shape[0]):
    spima.update_spim(w,i,lab[i])

def update_selection(w, img, hyp, r, nhl):
  img2 = img.copy()
  mask = seglib.mask_nhl(nhl, hyp)
  img2[mask] = img2[mask]*r
  spima.update_spim(w, 0, img2)

def update_stack(iss, img, hyp, r, nhl):
  img2 = img.copy()
  mask = seglib.mask_nhl(nhl, hyp)
  img2[mask] = img2[mask]*r
  iss.stack = img2

newcents = []
def onclick_centerpoints(event):
  xi, yi = int(event.xdata + 0.5), int(event.ydata + 0.5)
  zi = iss.idx[0]
  print('button=%d, x=%d, y=%d, xdata=%f, ydata=%f' %
      (event.button, event.x, event.y, event.xdata, event.ydata))
  print(zi, yi, xi)
  if event.key=='C':
    print('added! ', event.key)
    newcents.append([zi,yi,xi])
# cid = iss.fig.canvas.mpl_connect('button_press_event', onclick_centerpoints)

def shownew(img,**kwargs):
  plt.figure()
  return plt.imshow(img, **kwargs)


