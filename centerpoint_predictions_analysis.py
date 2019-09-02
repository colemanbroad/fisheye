pimg = qopene() ## cell centerpoint prediction from u-net

centroids = dc.detect2(pimg, rawdata)
centroids_gt = lib.mkpoints()


from scipy.optimize import linear_sum_assignment
# cost = np.matmul(centroids, centroids_gt)
cost = np.zeros((len(centroids), len(centroids_gt)))
for i,c in enumerate(centroids):
  for j,d in enumerate(centroids_gt):
    cost[i,j] = np.linalg.norm(c-d)
res = linear_sum_assignment(cost)

## timimg
sh = (100,3)
x = np.random.rand(*sh)*100
y = x + np.random.rand(*sh)*5

def munkres_pts(x,y):
  "40s for 1k 3D pts"
  cost = np.zeros((len(x), len(y)))
  for i,c in enumerate(x):
    for j,d in enumerate(y):
      cost[i,j] = np.linalg.norm(c-d)
  res = linear_sum_assignment(cost)
  return res


## Easy!

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
# ax.scatter(*centroids_gt.T)
# ax.scatter(*centroids2.T)
ax.scatter(*centroids.T, c=np.array(det['remaining']))

plt.figure()
plt.scatter(centroids_gt[:,0],centroids_gt[:,1])
plt.scatter(centroids[:,0],2*centroids[:,1])

plt.figure()
plt.scatter(centroids_gt[:,1],centroids_gt[:,2])
plt.scatter(2*centroids[:,1],2*centroids[:,2])

## scroll through stack with blue blobs marking cell centers

img2 = img[1,:,::2,::2,[0,1,1]].astype(np.float)
img2 = perm(img2,'1234','2341')
img2[...,2] = pimg
n99 = np.percentile(img2, 95, axis=(0,1,2), keepdims=True)
img2 = img2 / n99
iss = Stack(img2)

## seeded watershed segmentation from learned centerpoints with raw intensity potential

seeds = zoom(pimg, (1,2,2))

potential = img[0,...,1].copy().astype(np.float)
potential = potential / potential.mean()
kern = np.ones((5,5,5),np.float32)/5**3
potential = convolve(potential, kern)

pts = label(seeds>optthresh)[0]
seg = watershed(-potential, , mask=potential>1.0)

def compute_seg_on_slices(hyp, rawdata):
  "requires hyp with time dimension. TZYXC."
  print("COMPARE SEGMENTATION AGAINST LABELED SLICES")
  inds_labeled_slices = rawdata['inds_labeled_slices']
  lab = rawdata['lab']
  img = rawdata['img']
  inds = inds_labeled_slices[:,:-4] # use full
  gt_slices  = np.array([lab2instance(x) for x in lab[inds[0], inds[1]]])
  pre_slices = hyp[inds[0], inds[1]]
  seg_scores = np.array([ss.seg(x,y) for x,y in zip(gt_slices, pre_slices)])
  return seg_scores

## evaluate the results on GT slices
seedpts = label(seeds>optthresh)[0]

for p in np.linspace(0.8,1.5,10):
  seg = watershed(-potential, seedpts, mask=potential>p)
  res = ts.compute_seg_on_slices(seg[np.newaxis], rawdata)
  print(p, res.mean(), res.std())

## now show the borders on top of the original image

seg = watershed(-potential, seedpts, mask=potential>1.0)
bnds = label_tools.find_boundaries(seg)
img2[...,2] = bnds[:,::2,::2]


history = """
Fri Jul 13 18:06:28 2018
We can use these cell centerpoint predictions to help with cell segmentation.
Using them as seeds inside a seeded watershed is already helpful.
Let's re-optimize the basic intensity-based-potential seeded watershed segmentation with these seedpoints.
Mon Jul 16 10:38:15 2018
note: our ground truth annotations are heavily biased towards the borders of the image! They mostly come
from the first few slices in the first timepoint, which means border effects are very important.
Let's retrain using training data that includes zero-padding.
The intensity-based-potential actually does a good job of identifying the boundaries between cells.
But first, we'll just try to fix some of the problems with a small hyperopt search over params and denoising.

The avg brightness of nuclei varies with space. Nuclei near the tissue surface close to the laser
are brighter and their size will biased by an intensity-based-watershed.
We have to have a *mask* for the watershed. This means the watershed's outer surface for cells is just
a binary threshold. NOT FANCY. This is a poor estimate for intensity based images.

SEG score with 5x5x5 flat kernel blur on img with seedthresh=0.9 and potential_thresh = 1.2 x mean
= 0.4975

SEG score after updating unet_dist2center and doing simple intensity watershed is 0.543!
This is clearly our best yet. It also finds 102 cells in the initial image and 108 cells in the 2nd 
(training) image. These seed points come from a model which was progressively optimized down 
to gaussians with sig=10 and a 4x1 z downsampling. (nearly isotropic). w 10px boundary.
And after a simple brute force 1d optimization over the mask threshold we improve the seg
score all the way up to 0.57691! huzzah. (when mask thresh == 1.0).

TODO:
- [ ] try intensity seg after a simple global brightness adjuster.


"""