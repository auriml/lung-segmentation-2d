import os
import sys
import numpy as np
from skimage import io, exposure
import pydicom
import re
"""
Data is preprocessed in the following ways:
    - equalize histogram (skimage.exposure.equalize_hist);
   
"""

currentroot = os.getcwd()
os.chdir("../")
root = os.getcwd()
os.chdir(currentroot)

def make_lungs(replace = False):
    path = root + '/SJ'
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(path):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                study_id = re.search('image_dir/(.+?)/',dirName).group(1)
                if replace or (not replace and not os.path.exists(path + '/image_dir_processed/' + study_id + '_' + filename[-10:-4] + '.png')) :
                    lstFilesDCM.append(os.path.join(dirName,filename))

    ConstPixelDims = None
    ConstPixelSpacing = None
    for i, filename in enumerate(lstFilesDCM):
        try:
            study_id = re.search('image_dir/(.+?)/',filename).group(1)
            e = np.fromfile(filename, dtype='>u2')
            RefDs = pydicom.read_file(filename)
            #print(RefDs)
            # Load dimensions based on the number of rows, columns
            ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns))
            # Load spacing values (in mm)
            if hasattr(RefDs, 'PixelSpacing'):
                ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]))
                x = np.arange(0.0, (ConstPixelDims[0]+1)*ConstPixelSpacing[0], ConstPixelSpacing[0])
                y = np.arange(0.0, (ConstPixelDims[1]+1)*ConstPixelSpacing[1], ConstPixelSpacing[1])
            # The array is sized based on 'ConstPixelDims'
            ArrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
            # store the raw image data
            ArrayDicom[:, :] = RefDs.pixel_array
            #img = 1.0 - ArrayDicom * 1. / 4096
            #img = exposure.equalize_hist(img)
            #io.imsave(path + '/image_dir_processed/inverted_' + filename[-10:-4] + '.png', img)
            img =  ArrayDicom * 1. / 4096
            img = exposure.equalize_hist(img)
            io.imsave(path + '/image_dir_processed/' + study_id + '_' + filename[-10:-4] + '.png', img)
            print ('Lung', i, filename)
        except:
            print ("Unexpected error:", sys.exc_info()[0])
            pass



make_lungs()


