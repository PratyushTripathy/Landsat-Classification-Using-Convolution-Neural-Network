"""
(1/3)
Supporting script for the medium post titled:
'Is CNN equally shiny on mid-resolution satellite data?'
available at https://medium.com/p/9e24e68f0c08

Author: Pratyush Tripathy
Date: 29 May, 2021

Following package versions were used:
numpy - 1.17.2
sklearn - 0.22.1
pyrsgis - 0.3.9
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
    # create feature chips using pyrsgis
      
    features = imageChipsFromFile(feature_raster_file, x_size=7, y_size=7)
    """
    Since I added this code chunk later, I wanted to make least 
    possible changes in the remaining sections. The below line changes
    the index of the channels. This will be undone at a later stage.
    """
    features = np.rollaxis(features, 3, 1)

    # read the label file and reshape it
    labels = label_raster.flatten()
        
    print('Input features shape:', features.shape)
    print('Input labels shape:', labels.shape)
    print('Values in input features, min: %d & max: %d' % (features.min(), features.max()))
    print('Values in input labels, min: %d & max: %d' % (labels.min(), labels.max()))

    # Save the arrays as .npy files
    np.save('CNN_3by3_features.npy', features)
    np.save('CNN_3by3_labels.npy', labels)
    print('Arrays saved at location %s' % (os.getcwd()))
