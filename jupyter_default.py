# import IPython
import matplotlib.pyplot as plt
from math import ceil

def set_cell_width(percent=100):
    assert 10 < percent <= 100
    from IPython.core.display import display, HTML
    display(HTML("<style>.container { width:%d%% !important; }</style>" % percent))

def perm(arr,p1,p2):
  "permutation mapping p1 to p2 for use in numpy.transpose. elems must be unique."
  assert len(p1)==len(p2)
  perm = list(range(len(p1)))
  for i,p in enumerate(p2):
    perm[i] = p1.index(p)
  return arr.transpose(perm)

def imshow(img, fs=(20,20), cb=True, **kwargs):
    plt.figure(figsize=fs, **kwargs)
    ax = plt.imshow(img, **kwargs)
    if cb:
        plt.colorbar()
    return ax

def timewindow(lst, t, l):
    "window of fixed length l, into list lst. try to center around t."
    if t < l//2: t=l//2
    if t >= len(lst) - l//2: t=len(lst) - ceil(l/2)
    return lst[t-l//2:t+ceil(l/2)]