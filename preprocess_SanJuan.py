import os
import sys
import numpy as np
from skimage import io, exposure
import pydicom
import re
import time
"""
Data is preprocessed in the following ways:
    - equalize histogram (skimage.exposure.equalize_hist);
   
"""

currentroot = os.getcwd()
os.chdir("../")
root = os.getcwd()
os.chdir(currentroot)

def make_lungs(replace = False, imagePaths = None):
    path = root + '/SJ'
    lstFilesDCM = []
    if imagePaths is not None:
        lstFilesDCM = imagePaths
    else:
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
            start = time.clock()
            min = ArrayDicom.min()
            max = ArrayDicom.max()
            print ("ArrayDicom Mean: " + str(ArrayDicom.mean()))
            print ("Window Center: " + str(RefDs.WindowCenter))
            print ("ArrayDicom Min: " + str(min))
            print ("ArrayDicom Max: " + str(max))
            print ("Window Width: " + str(RefDs.WindowWidth))

            cp = np.copy(ArrayDicom)
            nmin = RefDs.WindowCenter - 0.5 - (RefDs.WindowWidth -1)/2
            nmax = RefDs.WindowCenter - 0.5 + (RefDs.WindowWidth -1)/2
            cp[ArrayDicom <= nmin] = min
            cp[ArrayDicom > nmax ] = max
            temp = ((ArrayDicom - (RefDs.WindowCenter - 0.5))/ (RefDs.WindowWidth -1) + 0.5) * (max-min) + min
            cp[ (ArrayDicom <= nmax) & (ArrayDicom > nmin)] = temp[(ArrayDicom <= nmax) & (ArrayDicom > nmin)]


            img =  cp * 1. / 4096
            print("Time elapsed")
            print(time.clock() - start)
            #img = exposure.equalize_hist(img)
            io.imsave(path + '/image_dir_processed/' + study_id + '_' + filename[-10:-4] + '.png', img)
            print ('Lung', i, filename)
        except:
            print ("Unexpected error:", sys.exc_info()[0])
            pass

images = ['/image_dir_processed/259099525557219735264115148468152712554_m5ff9v.png',
          '/image_dir_processed/299164937313584841767678964232362685010_sx5mth.png',
          '/image_dir_processed/315752159734031831877330441630077004881-2_a83wu8.png']

imagePath = [root + '/SJ/image_dir' + '/259099525557219735264115148468152712554/scans/1314-unknown/resources/DICOM/files/1.2.840.113619.2.182.10808617219229.1432721139.241245-1314-3963-1m5ff9v.dcm',
             ]
make_lungs()


