"""
(3/3)
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

import os, math
random.seed(2)
import numpy as np
import tensorflow as tf
from pyrsgis import raster

######################################################################
##### PART - C: LOADING THE SAVED MODEL AND PREDICTING NEW IMAGE #####
######################################################################

# Change the working directory
os.chdir(r'E:\CNN_Builtup\TDS_CNN')

# Load the saved model
model = tf.keras.models.load_model('trained_models/200409_CNN_Builtup_11by11_PScore0.928_RScore0.936_FScore0.932.h5')

# Load a new multispectral image
ds, featuresHyderabad = raster.read('l5_Hyderabad2011_raw.tif')
outFile = 'l5_Hyderabad_2011_Builtup_CNN_predicted_11by11.tif'

# Generate image chips in the back-end
def CNNdataGenerator(mxBands, kSize):
    mxBands = mxBands / 255.0
    nBands, rows, cols = mxBands.shape
    margin = math.floor(kSize/2)
    mxBands = np.pad(mxBands, margin, mode='constant')[margin:-margin, :, :]

    features = np.empty((rows*cols, kSize, kSize, nBands))

    n = 0
    for row in range(margin, rows+margin):
        for col in range(margin, cols+margin):
            feat = mxBands[:, row-margin:row+margin+1, col-margin:col+margin+1]

            b1, b2, b3, b4, b5, b6 = feat
            feat = np.dstack((b1, b2, b3, b4, b5, b6))

            features[n, :, :, :] = feat
            n += 1
            
    return(features)

# Call the function to generate features tensor
new_features = CNNdataGenerator(featuresHyderabad, kSize=11)
print('Shape of the new features', new_features.shape)

# Predict new data and export the probability raster
newPredicted = model.predict(new_features)
newPredicted = newPredicted[:,1]

prediction = np.reshape(newPredicted, (ds.RasterYSize, ds.RasterXSize))
raster.export(prediction, ds, filename=outFile, dtype='float')
