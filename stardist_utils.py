from __future__ import print_function, unicode_literals, absolute_import, division

import numpy as np
from gputools import OCLProgram, OCLArray, OCLImage
from scipy.ndimage.morphology import distance_transform_edt
import argparse
from skimage.segmentation import find_boundaries


kernel = """
__constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

inline float2 pol2cart(const float rho, const float phi) {
    const float x = rho * cos(phi);
    const float y = rho * sin(phi);
    return (float2)(x,y);
}

__kernel void star_dist(__global float* dst, read_only image2d_t src) {

    const int i = get_global_id(0), j = get_global_id(1);
    const int Nx = get_global_size(0), Ny = get_global_size(1);
    const float M_PI = 3.141592;

    const float2 origin = (float2)(i,j);
    const int value = read_imageui(src,sampler,origin).x;

    if (value == 0) {
        // background pixel -> nothing to do, write all zeros
        for (int k = 0; k < N_RAYS; k++) {
            dst[k + i*N_RAYS + j*N_RAYS*Nx] = 0;
        }
    } else {
        float st_rays = (2*M_PI) / N_RAYS; // step size for ray angles
        // for all rays
        for (int k = 0; k < N_RAYS; k++) {
            const float phi = k*st_rays; // current ray angle phi
            const float2 dir = pol2cart(1,phi); // small vector in direction of ray
            float2 offset = 0; // offset vector to be added to origin
            // find radius that leaves current object
            while (1) {
                offset += dir;
                const int offset_value = read_imageui(src,sampler,origin+offset).x;
                if (offset_value != value) {
                    const float dist = sqrt(offset.x*offset.x + offset.y*offset.y);
                    dst[k + i*N_RAYS + j*N_RAYS*Nx] = dist;
                    break;
                }
            }
        }
    }

}
"""

def stardist_from_labels(a, n_rays=32):
    """ assumes a to be a label image with integer values that encode object ids. id 0 denotes background. """
    out_shape = a.shape+(n_rays,)
    src = OCLImage.from_array(a.astype(np.uint16, copy=False))
    dst = OCLArray.empty(out_shape, dtype=np.float32)

    # program = OCLProgram("/home/uschmidt/research/dsb2018/notebooks/kernel.cl", build_options=["-D", "N_RAYS=%d" % n_rays])
    # program = OCLProgram("kernel.cl", build_options=["-D", "N_RAYS=%d" % n_rays])
    program = OCLProgram(src_str=kernel, build_options=["-D", "N_RAYS=%d" % n_rays])
    program.run_kernel('star_dist', src.shape, None, dst.data, src)
    return dst.get()


def ray_angles(n_rays=32):
    return np.linspace(0,2*np.pi,n_rays,endpoint=False)


def radii2coord(rhos):
    """ convert from polar to cartesian coordinates for a single image (3-D array) or multiple images (4-D array) """

    is_single_image = rhos.ndim == 3
    if is_single_image:
        rhos = np.expand_dims(rhos,0)
    assert rhos.ndim == 4

    n_images,h,w,n_rays = rhos.shape
    coord = np.empty((n_images,h,w,2,n_rays),dtype=rhos.dtype)

    start = np.meshgrid(np.arange(h),np.arange(w), indexing='ij')
    for i in range(2):
        start[i] = start[i].reshape(1,h,w,1)
        # start[i] = np.tile(start[i],(n_images,1,1,n_rays))
        start[i] = np.broadcast_to(start[i],(n_images,h,w,n_rays))
        coord[...,i,:] = start[i]

    phis = ray_angles(n_rays).reshape(1,1,1,n_rays)

    coord[...,0,:] += rhos * np.sin(phis) # row coordinate
    coord[...,1,:] += rhos * np.cos(phis) # col coordinate

    return coord[0] if is_single_image else coord


def radii2area(radii):
    #radii = dist[...,:-1]
    dphi = 2*np.pi/radii.shape[-1]
    return 0.5*np.sin(dphi)*np.sum(radii*np.roll(radii,1,axis=-1),axis=-1,keepdims=0)


def get_bbox(polys):
    """returns min,max of a single star-shaped polygon (or an array of polygons)

    polys.shape = (2, n_rays)

    or
    polys.shape = (n_polygons, 2, n_rays)

    """
    if not polys.ndim in (2,3):
        raise ValueError("polys should be given as 2 or 3 dimensional input (polys.shape = %s)"%str(polys.shape))
    if not polys.shape[-2] ==2:
        raise ValueError("the shape of polys should be (2, n_rays) or (n_polys,2,n_rays) but is of form %s "%str(polys.shape))

    return np.stack([np.amin(polys, axis = -2),np.amax(polys, axis = -2)], axis =0)




def get_slices(s_img,s_patch=256,allowed_remainder=0.25):
    assert s_img >= s_patch
    assert s_patch % 2 == 0
    n_slices = s_img // s_patch
    if (s_img%s_patch)/s_patch > allowed_remainder:
        n_slices += 1
    #n_slices = int(np.round(s_img/s_patch))
    s = s_patch // 2
    if n_slices == 1:
        # patch only fits once, take central position
        c = [s_img//2]
    else:
        # equi-spaced points
        c = np.linspace(s,s_img-s,n_slices,dtype=np.int)
    return tuple(slice(p-s,p+s) for p in c)


def edt_softmask(_lbl):
    dmap = np.zeros(_lbl.shape,np.float32)
    for l in (set(np.unique(_lbl)) - set([0])):
        mask = _lbl==l
        foo = distance_transform_edt(mask)
        foo[mask] /= np.max(foo[mask])
        dmap[mask] = foo[mask]
        #dmap += foo
    return dmap


# https://stackoverflow.com/a/43357954
def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def add_boundary_label(lbl):
    """ lbl is an integer label image (not binarized) """
    b = find_boundaries(lbl,mode='outer')
    res = (lbl>0).astype(np.uint8)
    res[b] = 2
    return res

def onehot_encoding(lbl,n_classes=None):
    """ n_classes will be determined by max lbl value if its value is None """
    from keras.utils import to_categorical

    onehot = to_categorical(lbl,num_classes=n_classes)
    n_classes = onehot.shape[-1]
    return onehot.reshape(lbl.shape+(n_classes,))
