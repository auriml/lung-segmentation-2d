from load_data import loadDataJSRT, loadDataMontgomery, loadDataGeneral

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from skimage import morphology, color, io, exposure
import os

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

def masked(img, gt, mask, alpha=1):
    """Returns image with GT lung field outlined with red, predicted lung field
    filled with blue."""
    rows, cols = img.shape
    color_mask = np.zeros((rows, cols, 3))
    boundary = morphology.dilation(gt, morphology.disk(3)) - gt
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

if __name__ == '__main__':

    # Path to csv-file. File should contain X-ray filenames as first column,
    # mask filenames as second column.
    #csv_path = '/path/to/JSRT/idx.csv'
    # Path to the folder with images. Images will be read from path + path_from_csv
    #path = csv_path[:csv_path.rfind('/')] + '/'

    #df = pd.read_csv(csv_path)

    root = '/Volumes/auri\'s home-5'
    #path = root + '/JSRT/new/'
    path = root + '/Rx-thorax-automatic-captioning/image_dir_processed/'
    y = [s for s in os.listdir(path) if not s.endswith('msk.png')]
    df = pd.DataFrame({'filename':y})
    df['mask filename'] = df.apply(lambda row: str(row.filename).replace('.png' , 'msk.png'), axis=1)

    # Load test data
    im_shape = (256, 256)
    #X, y = loadDataJSRT(df, path, im_shape)
    X, y = loadDataGeneral(df, path, im_shape)

    n_test = X.shape[0]
    inp_shape = X[0].shape

    # Load model
    model_name = 'trained_model.hdf5'
    UNet = load_model(model_name)

    # For inference standard keras ImageGenerator is used.
    test_gen = ImageDataGenerator(rescale=1.)

    ious = np.zeros(n_test)
    dices = np.zeros(n_test)

    i = 0

    for xx in X:
        img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
        xx = np.expand_dims(xx, 0)

        pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])
        pr = pred > 0.5

        # Remove regions smaller than 2% of the image
        pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))
        io.imsave('/Volumes/auri\'s home-5/lung-segmentation-2d/results/{}'.format(df.iloc[i][0]), masked(img,  pr, 1))
        i += 1


    # for xx, yy in test_gen.flow(X, y, batch_size=1):
    #     img = exposure.rescale_intensity(np.squeeze(xx), out_range=(0,1))
    #     pred = UNet.predict(xx)[..., 0].reshape(inp_shape[:2])
    #     mask = yy[..., 0].reshape(inp_shape[:2])
    #
    #     # Binarize masks
    #     gt = mask > 0.5
    #     pr = pred > 0.5
    #
    #     # Remove regions smaller than 2% of the image
    #     pr = remove_small_regions(pr, 0.02 * np.prod(im_shape))
    #
    #     io.imsave('/Volumes/auri\'s home-4/lung-segmentation-2d/results/{}'.format(df.iloc[i][0]), masked(img, gt, pr, 1))
    #
    #     ious[i] = IoU(gt, pr)
    #     dices[i] = Dice(gt, pr)
    #     print (df.iloc[i][0], ious[i], dices[i])
    #
    #     i += 1
    #     if i == n_test:
    #         break
    #
    # print ('Mean IoU:', ious.mean())
    # print ('Mean Dice:', dices.mean())

