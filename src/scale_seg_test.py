import numpy as np
from gputools import scale, pad_to_shape
from gputools.transforms.scale import _scale_shape

def crop_stack(data, sc = (5.5,.5,.5)):
    newshape = _scale_shape(_scale_shape(data.shape,sc),tuple(1./s for s in sc))
    return pad_to_shape(data, newshape)

def pad_stack(data, target_shape):
    return pad_to_shape(data, target_shape)


data_target = np.zeros((31, 512, 712), np.uint16)

sc = (5.5, .5, .5)

sc_inv = tuple(1./s for s in sc)

data_crop = crop_stack(data_target, sc)

data_crop_scaled = scale(data_crop,sc)

# do segmenting on data_crop_scaled, resulting in seg_crop_scaled
seg_crop_scaled = data_crop_scaled

seg_crop = scale(seg_crop_scaled, sc_inv)

seg_target = pad_stack(seg_crop, data_target.shape)

print(data_target.shape, seg_target.shape)

