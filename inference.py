from load_data import loadDataJSRT, loadDataMontgomery, loadDataGeneral

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, color, io, exposure
import os

currentroot = os.getcwd()
os.chdir("../")
root = os.getcwd()
os.chdir(currentroot)


def IoU(y_true, y_pred):
    """Returns Intersection over Union score for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    union = np.logical_or(y_true_f, y_pred_f).sum()
    return (intersection + 1) * 1. / (union + 1)

def Dice(y_true, y_pred):
    """Returns Dice Similarity Coefficient for ground truth and predicted masks."""
    assert y_true.dtype == bool and y_pred.dtype == bool
    y_true_f = y_true.flatten()
    y_pred_f = y_pred.flatten()
    intersection = np.logical_and(y_true_f, y_pred_f).sum()
    return (2. * intersection + 1.) / (y_true.sum() + y_pred.sum() + 1.)

def masked_withGT(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    dilation = morphology.dilation(gt, morphology.disk(3))
    boundary = np.subtract(dilation, gt, dtype=np.float)
    #boundary = morphology.dilation(gt, morphology.disk(3)) - gt
    color_mask[mask == 1] = [0, 0, 1]
    color_mask[boundary == 1] = [1, 0, 0]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked

def masked(img,  mask, alpha=1):
    """Returns image with  predicted lung field
    filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    color_mask[mask == 1] = [0, 0, 1]
    img_color = np.dstack((img, img, img))

    img_hsv = color.rgb2hsv(img_color)
    color_mask_hsv = color.rgb2hsv(color_mask)

    img_hsv[..., 0] = color_mask_hsv[..., 0]
    img_hsv[..., 1] = color_mask_hsv[..., 1] * alpha

    img_masked = color.hsv2rgb(img_hsv)
    return img_masked


def remove_small_regions(img, size):
    """Morphologically removes small (less than size) connected regions of 0s or 1s."""
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)
    return img


def test_benchmark_JSRT(model_name = 'UNet_trained_model.hdf5', im_shape = (256, 256)):
    #Load JSRT dataset
    path = root + '/JSRT/new/'
    y = [s for s in os.listdir(path) if not s.endswith('msk.png')]
    df = pd.DataFrame({'filename':y})
    df['mask filename'] = df.apply(lambda row: str(row.filename).replace('.png' , 'msk.png'), axis=1)

    # Load test data
    X, y = loadDataJSRT(df, path, im_shape)

    n_test = X.shape[0]
    inp_shape = X[0].shape

    # Load model
    model = load_model(model_name)

    # For inference standard keras ImageGenerator is used.
    test_gen = ImageDataGenerator(rescale=1.)

    ious = np.zeros(n_test)
    dices = np.zeros(n_test)

    i = 0

    for xx, yy in test_gen.flow(X, y, batch_size=1, shuffle=False):
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
        pred = model.predict(xx)[..., 0].reshape(inp_shape[:2])
        mask = yy[..., 0].reshape(inp_shape[:2])

        #Binarize masks
        gt = mask > 0.5
        pr = pred > 0.5

        #Remove regions smaller than 2% of the image
        pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))

        io.imsave(root + '/lung-segmentation-2d/results/{}'.format(model_name + df.iloc[i][0]), masked_withGT(img, gt, pr, 1))

        ious[i] = IoU(gt, pr)
        dices[i] = Dice(gt, pr)
        #print (df.iloc[i][0], ious[i], dices[i])

        i += 1
        if i == n_test:
            break


    print (model_name + ' Mean IoU:', ious.mean())
    print (model_name + ' Mean Dice:', dices.mean())


def segment_SanJuan_dataset(model_name = 'UNet_trained_model.hdf5', im_shape = (256, 256), n_images = None):
    #Load SanJuan dataset
    path = root + '/Rx-thorax-automatic-captioning/image_dir_processed/'
    y = [s for s in os.listdir(path) if not s.endswith('msk.png')]
    df = pd.DataFrame({'filename':y})
    df['mask filename'] = df.apply(lambda row: str(row.filename).replace('.png' , 'msk.png'), axis=1)

    # Load test data
    X, y = loadDataGeneral(df, path, im_shape, n_images = n_images)

    n_test = X.shape[0]
    inp_shape = X[0].shape

    # Load model
    model = load_model(model_name)

    # For inference standard keras ImageGenerator is used.
    test_gen = ImageDataGenerator(rescale=1.)

    ious = np.zeros(n_test)
    dices = np.zeros(n_test)

    i = 0

    for xx in X:
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
        xx = np.expand_dims(xx, 0)

        pred = model.predict(xx)[..., 0].reshape(inp_shape[:2])
        pr = pred > 0.5

        # Remove regions smaller than 2% of the image
        pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))
        io.imsave(root + '/lung-segmentation-2d/results/{}'.format(model_name + df.iloc[i][0]), masked(img,  pr, 1))
        i += 1




if __name__ == '__main__':

    #test_benchmark_JSRT(model_name='UNet_trained_model.hdf5', im_shape = (256, 256))
    test_benchmark_JSRT(model_name='gan_generator_model.hdf5' , im_shape= (400,400))
    #segment_SanJuan_dataset(model_name='UNet_trained_model.hdf5' , im_shape= (256,256) , n_images = 30)
    segment_SanJuan_dataset(model_name='gan_generator_model.hdf5' , im_shape= (400,400) , n_images = 30)