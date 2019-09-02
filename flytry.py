from ipython_local_defaults import *
from collections import defaultdict

datadir = Path('/Users/broaddus/Downloads/Fluo-N3DL-DRO/')
imgdir = datadir / '01/'
labdir = datadir / '01_GT'
img = imread('/Users/broaddus/Downloads/Fluo-N3DL-DRO/01/t000.tif')
lab = imread('/Users/broaddus/Downloads/Fluo-N3DL-DRO/01_GT/SEG/man_seg_000_075.tif')

imgs = []
for f0 in glob('/Users/broaddus/Downloads/Fluo-N3DL-DRO/01/*.tif'):
  img = imread(f0)
  imgs.append(img)
imgs = np.array(imgs)

labs = []
for f0 in glob('/Users/broaddus/Downloads/Fluo-N3DL-DRO/01_GT/SEG/*'):
  lab = imread(f0)
  labs.append(lab)
  print(lab.max())
labs = np.array(labs)

tras = []
for f0 in glob('/Users/broaddus/Downloads/Fluo-N3DL-DRO//01_GT/TRA/*.tif'):
  tra = imread(f0)
  tras.append(tra)
  print(len(tras))
tras = np.array(tras)

def find_averages(pts, coords):
  d = defaultdict(lambda : (np.array([0,0,0]),0))
  for i in range(len(pts)):
    a,b = d[pts[i]]
    d[pts[i]] = (a+coords[i], b+1)
  for k,v in d.items():
    d[k] = v[0] / v[1]
  return d

coord_list = []
for i in range(tras.shape[0]):
  tra = tras[i]
  x = np.argwhere(tra > 0)
  pts = tra[list(x.T)]
  coord_list.append(find_averages(pts, x))

coords_dict = dict()
for k,v in coord_list[0].items():
  coords_dict[k] = []
  for i in range(len(coord_list)):
    d = coord_list[i]
    coords_dict[k].append(d[k])
  coords_dict[k] = np.array(coords_dict[k])


from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for k,v in coords_dict.items():
  ax.plot(*v.T)





thoughts = """
There are only ten or so cells in each image!
Almost nothing.
And only XY annotations.
But the full neural lineage has centerpoint annotations.
But we can't use this for training.
We don't even know which pixels are bg class!
The trajectories of each cell in the neural lineage can be found projected onto XY and
colored from tan to purple in `not001`.

## Wed Jul 18 15:58:35 2018

It's easy to see the trajectories when plotted in this way in 3D. See `res068`.
Notice how sparse the cells are. If we could eliminate the remaining cells in the image these would be
easy to track using centerpoints alone!




"""

