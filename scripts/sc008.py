from ipython_local_defaults import *
import lib as ll

img = imread('data/img006.tif')
pimg = np.load('img61.npy')

print(img.shape)
img = img.transpose((0,3,1,2))

hx = gaussian(9, 1.0)
def f(p):
	p = gputools.convolve_sep3(p, hx, hx, hx)
	p = p/p.max()
	return p
pimg_blur = np.array([f(p) for p in pimg])


## just one blob
lab = label(pimg_blur>0.8)[0]
## reasonable start
lab = np.array([label(x>0.8)[0] for x in pimg_blur])
## some gain over the above. 1min 30sec. if we blur first.
lab = np.array([watershed(1-x,label(x>0.9)[0], mask = x>0.5) for x in pimg_blur])

## now let's try tracking cells using the 2nd definition of lab
sys.path.insert(0, '../')
from planaria_tracking import lib as tracklib
lab2 = lab[2:].copy()
nhls = tracklib.labs2nhls(lab2, img[2:,:,1], simple=False)
plotting.plot_nhls(nhls, x=lambda n:n['moments_img'][0,0,0]/n['area'])
tr = tracklib.nhls2tracking(nhls)
cm2 = tracklib.lineagelabelmap(tr.tb, tr.tv)
lab_res_rgb = tracklib.recolor_every_frame(lab2, tr.cm)

plt.figure()
# xs = np.array([np.log2(n['area']) for n in nhl])
xs = np.array([n['coords'][0] for n in nhl])
# ys = np.array([n['moments_img'][0,0,0]/n['area'] for n in nhl])
ys = np.array([n['coords'][1] for n in nhl])
col = plt.scatter(xs,ys)
selector = view.SelectFromCollection(plt.gca(), col)


## try to plot each nucleus in a grid

imgpad = np.pad(img3, 10, 'constant')
patches = plotting.nhl2crops(imgpad,nhl[anno[:,0]=='1'],axis=None, pad=10)
fig, axs = plt.subplots(6, 5, sharex=True, sharey=True)
for i,p in enumerate(patches[:30]):
	ax = axs.flat[i]
	d = 0
	a = p.shape[d]
	ss = [slice(None)]*3
	tw = timewindow(np.arange(a), a//2, a//4)
	ss[d] = tw
	ax.imshow(p[ss].max(d), aspect='equal')
	ax.axis('off')

## we have masks for every object


res = qopen()
plt.imshow(res)

## lasso selector




lasso = LassoSelector(iss.fig.gca(), onselect=onselect)
vertices = []
def onselect(verts):
    vertices = verts
    # path = Path(verts)
    # self.ind = np.nonzero([path.contains_point(xy) for xy in self.xys])[0]
    # self.fc[:, -1] = self.alpha_other
    # self.fc[self.ind, -1] = 1
    # self.collection.set_facecolors(self.fc)
    # self.canvas.draw_idle()
