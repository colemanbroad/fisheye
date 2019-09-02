from ipython_local_defaults import *
from csbdeep.utils import normalize

def grid_artifacts():
	"""
	Striped, evenly spaced artifacts in ISONET come from downscaling + upscaling?
	"""
	N = 128
	a = np.random.rand(N)
	da = a-zoom(a[::5],5)
	plt.plot(da)
	mask = np.abs(da) < 2e-2
	plt.scatter(np.arange(N)[mask],da[mask])

img = np.load('data/img006_noconv.npy')
img2 = imread('data/img006.tif')
res = np.load('restored.npy')

def test_normalization():
	"""
	why is the normalization wrong after isonet prediction?
	"""
	input0 = img[3]
	mi,mx = np.percentile(input0, [1, 99.8])
	input0_normed = (input0-mi)/(mx-mi)
	
	prob = """
	When we run predict_isonet.py with
	`normalizer = PercentileNormalizer(1,99.8, do_after=True)`
	res has min and max of... (-39936.927734375, 105806094.24609375)
	"""
	
	resmin, resmax = res.min(), res.max()
	print(resmin, resmax)
	res_false = np.load('restored_doafterFalse.npy')
	resmin, resmax = res_false.min(), res_false.max()
	print(resmin, resmax)

	res_nonorm = np.load('restored_nonorm.npy')
	resmin, resmax = res_nonorm.min(), res_nonorm.max()
	res_nonorm = perm(res_nonorm, 'zcyx', 'zyxc')
	res_nonorm = normalize(res_nonorm, axis=(0,1,2), clip=True)
	print(resmin, resmax)

	plt.figure()
	plt.plot(np.percentile(res_nonorm,np.linspace(0,100,1000)))

	res_nonorm = perm(res_nonorm, "zcyx", "zyxc")

	res_dev = np.load('restored_dev.npy')
	resmin, resmax = res_dev.min(), res_dev.max()
	res_dev = perm(res_dev, 'zcyx', 'zyxc')
	res_dev = normalize(res_dev, axis=(0,1,2), clip=True)
	print(resmin, resmax)
	plt.figure()
	plt.plot(np.percentile(res_dev,np.linspace(0,100,1000)))

	res_dev_0 = np.load('restored_dev_0.npy')
	resmin, resmax = res_dev_0.min(), res_dev_0.max()
	res_dev_0 = perm(res_dev_0, 'zcyx', 'zyxc')
	res_dev_0 = normalize(res_dev_0, axis=(0,1,2), clip=True)
	print(resmin, resmax)
	
	plt.figure()
	plt.plot(np.percentile(res_dev_0/res_dev_0.mean((0,1,2)),np.linspace(0,100,1000), axis=(0,1,2)))
	plt.plot(np.percentile(img[0]/img[0].mean((0,2,3)),np.linspace(0,100,1000), axis=(0,2,3)))
	



	prob = """
	We need to know if res_nonorm matches up the the original labeling.
	"""

	lab = imread('data/labels_lut.tif').astype(np.float32)
	# lab = lab[:,:,0]
	lab = lab[3]
	lab = perm(lab, 'zcyx', 'zyxc')
	x = res_nonorm[::5]
	lab[...,[1,2]] = x




test_normalization()
