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
##### PART - A: READING AND STORING IMAGE CHIPS AS NUMPY ARRAYS #####
#####################################################################

# Change the working directory
imageDirectory = r"E:\CNN_Builtup\TDS_CNN\11by11\ImageChips"
os.chdir(imageDirectory)

# Get the number of files in the directory
nSamples = len(glob.glob('*.tif'))

# Get basic information about the image chips
ds, tempArr = raster.read(os.listdir(imageDirectory)[0])
nBands, rows, cols = ds.RasterCount, ds.RasterXSize, ds.RasterYSize

# Create empty arrays to store data later
features = np.empty((nSamples, nBands, rows, cols))
labels = np.empty((nSamples, ))

# Loop through the files, read and stack
for n, file in enumerate(glob.glob('*.tif')):
    if n==nSamples:
        break
    if n % 5000 == 0:
        print('Sample read: %d of %d' % (n, nSamples))
    ds, tempArr = raster.read(file)
    # Get filename without extension, split by underscore and get the label
    tempLabel = os.path.splitext(file)[0].split('_')[-1]

    features[n, :, :, :] = tempArr
    labels[n] = tempLabel
    
    
print('Input features shape:', features.shape)
print('Input labels shape:', labels.shape)
print('Values in input features, min: %d & max: %d' % (features.min(), features.max()))

os.chdir('..\\..')

# Save the arrays as .npy files
np.save('CNN_11by11_features.npy', features)
np.save('CNN_11by11_labels.npy', labels)
print('Arrays saved at location %s' % (os.getcwd()))
