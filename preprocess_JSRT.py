import os
import numpy as np
from skimage import io, exposure
"""
Data is preprocessed in the following ways:
    - resize to im_shape;
    - equalize histogram (skimage.exposure.equalize_hist);
   
"""

root = '/Volumes/auri\'s home-4'
def make_lungs():
    path = root + '/JSRT/All247images/'
    for i, filename in enumerate(os.listdir(path)):
        img = 1.0 - np.fromfile(path + filename, dtype='>u2').reshape((2048, 2048)) * 1. / 4096
        img = exposure.equalize_hist(img)
        io.imsave(root + '/JSRT/new/' + filename[:-4] + '.png', img)
        print ('Lung', i, filename)

def make_masks():
    path = root + '/JSRT/All247images/'
    for i, filename in enumerate(os.listdir(path)):
        filepath1_left = root + '/JSRT/JSRT_segmented/fold1/masks/left lung/' + filename[:-4] + '.gif'
        filepath1_right = root + '/JSRT/JSRT_segmented/fold1/masks/right lung/' + filename[:-4] + '.gif'
        filepath2_left = root + '/JSRT/JSRT_segmented/fold2/masks/left lung/' + filename[:-4] + '.gif'
        filepath2_right = root + '/JSRT/JSRT_segmented/fold2/masks/right lung/' + filename[:-4] + '.gif'

        if os.path.isfile(filepath1_left):
            left = io.imread(filepath1_left )
            right = io.imread(filepath1_right)
            io.imsave(root + '/JSRT/new/' + filename[:-4] + 'msk.png', np.clip(left + right, 0, 255))
            print ('Mask', i, filename)
        elif os.path.isfile(filepath2_left):
            left = io.imread(filepath2_left)
            right = io.imread(filepath2_right)
            io.imsave(root + '/JSRT/new/' + filename[:-4] + 'msk.png', np.clip(left + right, 0, 255))
            print ('Mask', i, filename)

make_lungs()
make_masks()
