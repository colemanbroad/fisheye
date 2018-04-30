from ipython_local_defaults import *
import lib as ll

img = imread('data/img006.tif')
pimg = np.load('training/t009/img6pred_5class_aug.npy')

print(img.shape)

# hx = gaussian(9, 1.0)
# def f(p):
# 	p = gputools.convolve_sep3(p, hx, hx, hx)
# 	p = p/p.max()
# 	return p
# pimg_blur = np.array([f(p) for p in pimg])



## just one blob
lab = label(pimg_blur>0.8)[0]
## reasonable start
lab = np.array([label(x>0.5)[0] for x in pimg_blur])
## some gain over the above. 1min 30sec. if we blur first.
def f(x):
	x1 = x[...,1] # nuclei
	x2 = x[...,2] # borders
	mask = (x1 > 0.5) #& (x2 < 0.1)
	res = watershed(1-x1,label(x1>0.9)[0], mask = mask)
	return res
lab = np.array([f(x) for x in pimg[2:4]])

## focus on trackable timepoints

img2 = img[2:,:,1].copy()
lab2 = lab[2:].copy()

## now let's try tracking cells using the 2nd definition of lab

nhls = tracklib.labs2nhls(lab2, img2, simple=False)
tr = tracklib.nhls2tracking(nhls)
cm2 = tracklib.lineagelabelmap(tr.tb, tr.tv)
lab_res = tracklib.recolor_every_frame(lab2, cm2)
lab_res_rgb = tracklib.recolor_every_frame(lab2, tr.cm)
plotting.plot_nhls(nhls, x=lambda n:n['moments_img'][0,0,0]/n['area'])

## analysis of nhl

nhl = np.array(nhls[0])

w = spimagine.volshow(img2, stackUnits=[1,1,4])

## plot with selector

plt.figure()
nhl = nhl2
xs = np.array([n['area'] for n in nhl])
xs = np.array([np.log2(n['area']) for n in nhl])
# xs = np.array([n['coords'][0] for n in nhl])
# ys = np.array([n['moments_img'][0,0,0]/n['area'] for n in nhl])
# ys = np.array([n['coords'][1] for n in nhl])
ys = np.array([n['surf'] for n in nhl])
ys = np.array([np.log2(n['surf']) for n in nhl])
col2 = plt.scatter(xs,ys)
selector1 = view.SelectFromCollection(ax, col)

## pairplot all the features for a single nhl

dat = seglib.nhl2dataframe(nhl, vecs=False, moments=False)
sns.pairplot(dat[['area','surf','dims0','dims1','dims2']]) #,'min_intensity','max_intensity']])

## figure out neighbor stats for each object

neibs = voronoi.label_neighbors(lab2[0])
tot = neibs.sum(0) + neibs.sum(1)

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
