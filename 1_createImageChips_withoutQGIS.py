"""
(1/3)
Supporting script for the medium post titled:
'Is CNN equally shiny on mid-resolution satellite data?'
available at https://medium.com/p/9e24e68f0c08

Author: Pratyush Tripathy
Date: 21 April, 2020

Following package versions were used:
numpy - 1.17.2
sklearn - 0.22.1
pyrsgis - 0.3.1
tensorflow - 2.0.0
"""

import os, math, random, glob, time
random.seed(2)
import numpy as np
from pyrsgis import raster

#####################################################################
##### PART - A: CREATING AND STORING IMAGE CHIPS AS NUMPY ARRAYS ####
#####################################################################

# Change the working directory
output_directory = r"E:\TDS_CNN"
os.chdir(output_directory)

# Read the feature and label rasters
feature_raster_file = r"Jiaxing_2015_Landsat.tif"
label_raster_file = r"Jiaxing_2015_builtup.tif"

ds, feature_raster = raster.read(feature_raster_file)
ds, label_raster = raster.read(label_raster_file)

if (feature_raster.shape[-1] != label_raster.shape[-1]) or\
   (feature_raster.shape[-2] != label_raster.shape[-2]):
    print('Shape of the input rasters do not match. Ending program.')
else:
    print("Rasters' shape matched.")

    # Generate image chips in the back-end
    def CNNdataGenerator(mxBands, labelBand, kSize):
        mxBands = mxBands / 255.0
        nBands, rows, cols = mxBands.shape
        
        margin = math.floor(kSize/2)
        mxBands = np.pad(mxBands, margin, mode='constant')[margin:-margin, :, :]
        labelBand = np.pad(labelBand, margin, mode='constant')

        features_arr = np.empty((rows*cols, kSize, kSize, nBands))
        labels_arr = np.empty((rows*cols, ))

        n = 0
        for row in range(margin, rows+margin):
            for col in range(margin, cols+margin):
                if (row % 500 == 0) and (col % 500 == 0):
                    print('Row: %d/%d, Col: %d/%d' % (row, rows, col, cols))
                                         
                feat = mxBands[:, row-margin:row+margin+1, col-margin:col+margin+1]
                label = labelBand[row, col]

                b1, b2, b3, b4, b5 = feat
                feat = np.dstack((b1, b2, b3, b4, b5))

                features_arr[n, :, :, :] = feat
                labels_arr[n] = label
                n += 1
                
        return(features_arr, labels_arr)

    features, labels = CNNdataGenerator(feature_raster, label_raster, kSize=3)
       
    print('Input features shape:', features.shape)
    print('Input labels shape:', labels.shape)
    print('Values in input features, min: %d & max: %d' % (features.min(), features.max()))
    print('Values in input labels, min: %d & max: %d' % (labels.min(), labels.max()))

    # Save the arrays as .npy files
    np.save('CNN_3by3_features.npy', features)
    np.save('CNN_3by3_labels.npy', labels)
    print('Arrays saved at location %s' % (os.getcwd()))
