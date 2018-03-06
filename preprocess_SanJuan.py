import os
import numpy as np
from skimage import io, exposure
import dicom
"""
Data is preprocessed in the following ways:
    - equalize histogram (skimage.exposure.equalize_hist);
   
"""

root = '/Volumes/auri\'s home-5/'
def make_lungs():
    path = root + 'Rx-thorax-automatic-captioning'
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName,filename))
    ConstPixelDims = None
    ConstPixelSpacing = None
    for i, filename in enumerate(lstFilesDCM):
        e = np.fromfile(filename, dtype='>u2')
        RefDs = dicom.read_file(filename)
        # Load dimensions based on the number of rows, columns
        ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns))
        # Load spacing values (in mm)
        if hasattr(RefDs, 'PixelSpacing'): ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]))
        x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
        y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
        # The array is sized based on 'ConstPixelDims'
        ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
        # store the raw image data
        ArrayDicom[:, :] = RefDs.pixel_array
        img = 1.0 - ArrayDicom * 1. / 4096
        img = exposure.equalize_hist(img)
        io.imsave(path + '/image_dir_processed/inverted_' + filename[-10:-4] + '.png', img)
        img =  ArrayDicom * 1. / 4096
        img = exposure.equalize_hist(img)
        io.imsave(path + '/image_dir_processed/' + filename[-10:-4] + '.png', img)
        print ('Lung', i, filename)


make_lungs()

