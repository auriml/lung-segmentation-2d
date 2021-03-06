import numpy as np
from skimage import transform, io, img_as_float, exposure
import os

"""
Data was preprocessed in the following ways:
    - resize to im_shape;
    - equalize histogram (skimage.exposure.equalize_hist);
    - normalize by data set mean and std.
Resulting shape should be (n_samples, img_width, img_height, 1).

It may be more convenient to store preprocessed data for faster loading.

Dataframe should contain paths to images and masks as two columns (relative to `path`).
"""

def loadDataJSRT(df, path, im_shape, n_images = None, load_clav_heart_masks=True):
    """This function loads data preprocessed with `preprocess_JSRT.py`"""
    X, y = [], []
    images = df.iterrows() if n_images is None else df[:n_images].iterrows()
    for i, item in images:
        img = io.imread(path + item[0])
        img = transform.resize(img, im_shape)
        img = np.expand_dims(img, -1)
        mask = io.imread(path + item[1])
        mask = transform.resize(mask, im_shape)
        mask = np.expand_dims(mask, -1)
        if load_clav_heart_masks:
            clav_heart_mask = io.imread((path + item[1]).replace('msk.png', 'clav_heart_msk.png') )
            clav_heart_mask = transform.resize(clav_heart_mask, im_shape)
            clav_heart_mask = np.expand_dims(clav_heart_mask, -1)
            mask = np.concatenate([mask, clav_heart_mask], axis = 2)
        X.append(img)
        y.append(mask)
    X = np.array(X)
    y = np.array(y)
    X -= X.mean()
    X /= X.std()

    print ('### Data loaded')
    print ('\t{}'.format(path))
    print ('\t{}\t{}'.format(X.shape, y.shape))
    print ('\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max()))
    print ('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))
    return X, y


def loadDataMontgomery(df, path, im_shape):
    """Function for loading Montgomery dataset"""
    X, y = [], []
    for i, item in df.iterrows():
        img = img_as_float(io.imread(path + item[0]))
        gt = io.imread(path + item[1])
        l, r = np.where(img.sum(0) > 1)[0][[0, -1]]
        t, b = np.where(img.sum(1) > 1)[0][[0, -1]]
        img = img[t:b, l:r]
        mask = gt[t:b, l:r]
        img = transform.resize(img, im_shape)
        img = exposure.equalize_hist(img)
        img = np.expand_dims(img, -1)
        mask = transform.resize(mask, im_shape)
        mask = np.expand_dims(mask, -1)
        X.append(img)
        y.append(mask)
    X = np.array(X)
    y = np.array(y)
    X -= X.mean()
    X /= X.std()

    print ('### Data loaded')
    print ('\t{}'.format(path))
    print ('\t{}\t{}'.format(X.shape, y.shape))
    print ('\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max()))
    print ('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))
    return X, y


def loadDataGeneral(df, path, im_shape, n_images = None):
    """Function for loading arbitrary data in standard formats"""
    X, y = [], []
    images = df.iterrows() if n_images is None else df[:n_images].iterrows()
    for i, item in images:
        img = img_as_float(io.imread(path + item[0]))
        if os.path.isfile(path + item[1]):
            mask = io.imread(path + item[1])
            mask = transform.resize(mask, im_shape)
            mask = np.expand_dims(mask, -1)
            y.append(mask)
        img = transform.resize(img, im_shape)
        img = exposure.equalize_hist(img)
        img = np.expand_dims(img, -1)


        X.append(img)

    X = np.array(X)
    y = np.array(y)
    X -= X.mean()
    X /= X.std()

    print ('### Dataset loaded')
    print ('\t{}'.format(path))
    print ('\t{}\t{}'.format(X.shape, y.shape))
    #print ('\tX:{:.1f}-{:.1f}\ty:{:.1f}-{:.1f}\n'.format(X.min(), X.max(), y.min(), y.max()))
    print ('\tX.mean = {}, X.std = {}'.format(X.mean(), X.std()))
    if not y: y = None
    return X, y

