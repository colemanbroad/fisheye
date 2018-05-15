# %load_ext autoreload
# %autoreload 2
from ipython_remote_defaults import *
import lib as ll
from segtools import lib as seglib
from segtools import segtools_simple as ss
from scipy.ndimage.filters import convolve

mypath = Path("intensity_seg/d003/")
mypath.mkdir(exist_ok=True)

if not run_from_ipython():
  myfile = Path(__file__)
  print(mypath / myfile.name)
  if not RERUN:
    shutil.copy(myfile, mypath / 'seg_intensity.py')
  sys.stdout = open(mypath / 'stdout.txt', 'w')
  sys.stderr = open(mypath / 'stderr.txt', 'w')

## define channels
ch_mem_img = 0
ch_nuc_img = 1
rgbimg = [ch_mem_img, ch_nuc_img, ch_nuc_img]

## define classes
n_classes  = 4
ch_div = 3
ch_nuc = 1
ch_mem = 0
ch_bg  = 2
ch_ignore = 1
rgbmem = [ch_mem, ch_nuc, ch_bg]
rgbdiv = [ch_div, ch_nuc, ch_bg]
classweights = [1/n_classes,]*n_classes

def condense_labels(lab):
  lab[lab==0]   = ch_bg
  lab[lab==255] = ch_mem
  lab[lab==168] = ch_nuc
  lab[lab==85]  = ch_div
  lab[lab==198] = ch_ignore
  return lab

## load data
img = imread('data/img006.tif')
lab = imread('data/labels_lut.tif')
lab = lab[:,:,0]
lab = condense_labels(lab)
mask_train_slices = lab.min((2,3)) < 2
inds_train_slices = np.indices(mask_train_slices.shape)[:,mask_train_slices]

inds = inds_train_slices[:,:-4] # just do time zero 

def lab2instance(x):
  x[x!=1] = 0
  x = label(x)[0]
  return x
y_gt_instance = np.array([lab2instance(x) for x in lab[inds[0], inds[1]]])

def convolve_zyx(pimg, axes="tzyxc"):
  assert pimg.ndim == 5
  pimg = perm(pimg, axes, "tzyxc")
  weights = np.full((3, 3, 3), 1.0/27)
  pimg = [[convolve(pimg[t,...,c], weights=weights) for c in range(pimg.shape[-1])] for t in range(pimg.shape[0])]
  pimg = np.array(pimg)
  pimg = pimg.transpose((0,2,3,4,1))
  pimg = [[convolve(pimg[t,...,c], weights=weights) for c in range(pimg.shape[-1])] for t in range(pimg.shape[0])]
  pimg = np.array(pimg)
  pimg = pimg.transpose((0,2,3,4,1))
  return pimg

## prepare the image for watershed
img_instseg = perm(img[[0]], "tzcyx", "tzyxc")
img_instseg = img_instseg / img_instseg.mean((1,2,3), keepdims=True)
img_instseg = convolve_zyx(img_instseg)

def instance_seg_raw_stack(x, lower_thresh=2, upper_thresh=2.8):
  assert x.ndim == 4
  ## axes = ZYXC
  x1 = x[...,ch_nuc_img] # nuclei
  x2 = x[...,ch_mem_img]
  poten = 1 - x1 #- x2
  poten = np.clip(poten, 0, 1)
  mask = (x1 > lower_thresh) #& (x2 < 0.15)
  seed = (x1 > upper_thresh) #& (x2 < 0.1)
  res  = watershed(poten, label(seed)[0], mask=mask)
  return res

def optimize_watershed(img_instseg):
	res_list = []
	time_list = []
	n = 3
	l1_list = np.linspace(.9, 1.1, n)
	l2_list = np.linspace(2.1, 2.4, n)

	for l1,l2 in itertools.product(l1_list, l2_list):
		t0 = time()
		hyp = np.array([instance_seg_raw_stack(x, lower_thresh=l1, upper_thresh=l2) for x in img_instseg])
		hyp = hyp.astype(np.uint16)
		y_pred_instance = hyp[inds[0], inds[1]]
		res = np.array([ss.seg(x,y) for x,y in zip(y_gt_instance, y_pred_instance)])
		t1 = time()
		res_list.append(res)
		time_list.append(t1-t0)

	res_list = np.array(res_list)
	res_list = res_list.reshape((n,n,-1))
	resmean = res_list.mean(-1)

	plt.figure()
	plt.imshow(resmean)
	plt.xticks(np.arange(n), np.around(l2_list, 2), rotation=45)
	plt.yticks(np.arange(n), np.around(l1_list, 2))
	plt.xlabel('seed threshold')
	plt.ylabel('base threshold')
	plt.colorbar()
	plt.savefig(mypath / 'mean.png')

	json.dump({'l1_list':l1_list,'l2_list':l2_list}, (mypath / 'params.json').open(mode='w'), cls=NumpyEncoder)

	print("Avg Time Taken: ", np.mean(time_list))

	idx = np.argwhere(resmean==resmean.max())
	idx = idx[0] # take first, there should only be one optimum
	l1opt, l2opt, val = l1_list[idx[0]], l2_list[idx[1]], resmean[idx[0], idx[1]]
	res = {'l1opt':l1opt, 'l2opt':l2opt, 'seg':val}
	json.dump(res, open(mypath / 'SEG.json', 'w'))

	print("number of optima:", len(idx))
	print("Optimal Params are:", l1opt, l2opt)
	print("Optimal SEG value is:", val)
	hyp = np.array([instance_seg_raw_stack(x, lower_thresh=l1opt, upper_thresh=l2opt) for x in img_instseg])
	hyp = hyp.astype(np.uint16)	
	return hyp, res

hyp, res = optimize_watershed(img_instseg)

