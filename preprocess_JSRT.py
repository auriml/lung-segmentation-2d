import os
import numpy as np
from skimage import io, exposure
"""
Data is preprocessed in the following ways:
    - resize to im_shape;
    - equalize histogram (skimage.exposure.equalize_hist);
   
"""
currentroot = os.getcwd()
os.chdir("../")
root = os.getcwd()
os.chdir(currentroot)


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

def make_clav_heart_masks():
    # both clavicles and heart do not overlap  and can be included in a single mask
    path = root + '/JSRT/All247images/'
    for i, filename in enumerate(os.listdir(path)):
        filepath1_left = root + '/JSRT/JSRT_segmented/fold1/masks/left clavicle/' + filename[:-4] + '.gif'
        filepath1_right = root + '/JSRT/JSRT_segmented/fold1/masks/right clavicle/' + filename[:-4] + '.gif'
        filepath1_heart = root + '/JSRT/JSRT_segmented/fold1/masks/heart/' + filename[:-4] + '.gif'
        filepath2_left = root + '/JSRT/JSRT_segmented/fold2/masks/left clavicle/' + filename[:-4] + '.gif'
        filepath2_right = root + '/JSRT/JSRT_segmented/fold2/masks/right clavicle/' + filename[:-4] + '.gif'
        filepath2_heart = root + '/JSRT/JSRT_segmented/fold2/masks/heart/' + filename[:-4] + '.gif'

        if os.path.isfile(filepath1_left):
            left = io.imread(filepath1_left )
            right = io.imread(filepath1_right)
            heart = io.imread(filepath1_heart )
            io.imsave(root + '/JSRT/new/' + filename[:-4] + 'clav_heart_msk.png', np.clip(left + right + heart, 0, 255))
            print ('Mask', i, filename)
        elif os.path.isfile(filepath2_left):
            left = io.imread(filepath2_left)
            right = io.imread(filepath2_right)
            heart = io.imread(filepath2_heart )
            io.imsave(root + '/JSRT/new/' + filename[:-4] + 'clav_heart_msk.png', np.clip(left + right + heart, 0, 255))
            print ('Mask', i, filename)

#make_lungs()
#make_masks()
make_clav_heart_masks()
