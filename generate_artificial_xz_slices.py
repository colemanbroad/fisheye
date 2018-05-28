from ipython_local_defaults import *

img = np.load('data/img006_noconv.npy')
# tzcyx

inds = np.arange(img_z.shape[1])
np.random.shuffle(inds)

ss = [slice(None)]*5
ss[0] = [0,4,8]
zi = inds[:10]

from scipy.ndimage import convolve

def k_random_inds(n,k):
	"n choose k. returns an ndarray of indices."
	inds = np.arange(n)
	np.random.shuffle(inds)
	return inds[:k]

img2 = img[[0,4,8]]
img_z = img2[:,k_random_inds(71,8)]
img_z = perm(img_z, "tzcyx", "tczyx")
img_y = img2[:,:,:,k_random_inds(400,8)]
img_y = perm(img_y, "tzcyx", "tcyxz")
img_x = img2[:,:,:,:,k_random_inds(400,8)]
img_x = perm(img_x, "tzcyx", "tcxyz")

def buildkern():
	kern = np.exp(- (np.arange(10)**2 / 2))
	kern /= kern.sum()
	return kern
def downsamp(img, axis=0, kern=buildkern()):
  assert img.ndim==2
  kern_shape = [1,1]
  kern_shape[axis] = len(kern)
  kern = kern.reshape(kern_shape)
  img = convolve(img, kern)  
  ss = list(np.s_[:,:])
  ds = 71*5
  ss[axis] = slice(None,ds,5)
  img = img[ss]
  # img = imresize(img, newshape, interp='nearest')
  return img
def makecat():
	img_z_down_x = ll.broadcast_nonscalar_func(lambda x: downsamp(x,0), img_z, (3,4))
	img_z_down_y = ll.broadcast_nonscalar_func(lambda x: downsamp(x,1), img_z, (3,4))
	img_z_down_x = np.swapaxes(img_z_down_x, -1, -2)
	cat = np.concatenate([img_z_down_x, img_x, img_z_down_y, img_y], axis=-1)
	return cat
iss = Stack(makecat())


learn = """
np.piecewise
np.frompyfunc
np.vectorize

np.apply_along_axis
np.apply_over_axes

np.fromfunctiopn

np.s_
np.index_exp
np.ndindex
"""






_end = '_end_'

def make_trie(*words):
    root = dict()
    for word in words:
        current_dict = root
        for letter in word:
            current_dict = current_dict.setdefault(letter, {})
        current_dict[_end] = _end
    return root


import string
import random
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
   return ''.join(random.choice(chars) for _ in range(size))
