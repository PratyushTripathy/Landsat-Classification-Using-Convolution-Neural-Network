import os, math, random, glob, time
random.seed(2)
import numpy as np
import tensorflow as tf
from pyrsgis import raster
from sklearn.utils import resample

from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from tensorflow.keras.callbacks import ModelCheckpoint
from pyrsgis import raster
from pyrsgis.convert import changeDimension
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from matplotlib import pyplot as plt

import logging
logging.getLogger('tensorflow').disabled = True


#####################################################################
##### PART - A: READING AND STORING IMAGE CHIPS AS NUMPY ARRAYS #####
#####################################################################
"""
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


#########################################################################
##### PART - B: READING NUMPY ARRAYS. TRAINING AND SAVING THE MODEL #####
#########################################################################

os.chdir(r"E:\CNN_Builtup\TDS_CNN")

# Load arrays from .npy files
features = np.load('CNN_11by11_features.npy')
labels = np.load('CNN_11by11_labels.npy')

# Separate and balance the classes
built_features = features[labels==1]
built_labels = labels[labels==1]

unbuilt_features = features[labels==0]
unbuilt_labels = labels[labels==0]

print('Number of records in each class:')
print('Built: %d, Unbuilt: %d' % (built_labels.shape[0], unbuilt_labels.shape[0]))

# Downsample the majority class
unbuilt_features = resample(unbuilt_features,
                            replace = False, # sample without replacement
                            n_samples = built_features.shape[0], # match minority n
                            random_state = 2)

unbuilt_labels = resample(unbuilt_labels,
                          replace = False, # sample without replacement
                          n_samples = built_features.shape[0], # match minority n
                          random_state = 2)

print('Number of records in balanced classes:')
print('Built: %d, Unbuilt: %d' % (built_labels.shape[0], unbuilt_labels.shape[0]))

# Combine the balanced features
features = np.concatenate((built_features, unbuilt_features), axis=0)
labels = np.concatenate((built_labels, unbuilt_labels), axis=0)

# Normalise the features
features = features / 255.0
print('New values in input features, min: %d & max: %d' % (features.min(), features.max()))

# Define the function to split features and labels
def train_test_split(features, labels, trainProp=0.6):
    dataSize = features.shape[0]
    sliceIndex = int(dataSize*trainProp)
    randIndex = np.arange(dataSize)
    random.shuffle(randIndex)
    train_x = features[[randIndex[:sliceIndex]], :, :, :][0]
    test_x = features[[randIndex[sliceIndex:]], :, :, :][0]
    train_y = labels[randIndex[:sliceIndex]]
    test_y = labels[randIndex[sliceIndex:]]
    return(train_x, train_y, test_x, test_y)
  
# Call the function to split the data
train_x, train_y, test_x, test_y = train_test_split(features, labels)

# Transpose the features to channel last format
train_x = tf.transpose(train_x, [0, 2, 3, 1])
test_x = tf.transpose(test_x, [0, 2, 3, 1])
print('Reshaped features:', train_x.shape, test_x.shape)
_, rowSize, colSize, nBands = train_x.shape

# Create a model
model = keras.Sequential()
model.add(Conv2D(32, kernel_size=1, padding='valid', activation='relu', input_shape=(rowSize, colSize, nBands)))
model.add(Dropout(0.25))
model.add(Conv2D(48, kernel_size=1, padding='valid', activation='relu'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

# Run the model
model.compile(loss='sparse_categorical_crossentropy', optimizer= 'rmsprop',metrics=['accuracy'])
history = model.fit(train_x, train_y, epochs=10)

# Predict for test data 
yTestPredicted = model.predict(test_x)
yTestPredicted = yTestPredicted[:,1]

# Calculate and display the error metrics
yTestPredicted = (yTestPredicted>0.5).astype(int)
cMatrix = confusion_matrix(test_y, yTestPredicted)
pScore = precision_score(test_y, yTestPredicted)
rScore = recall_score(test_y, yTestPredicted)
fScore = f1_score(test_y, yTestPredicted)

print("Confusion matrix:\n", cMatrix)

print("\nP-Score: %.3f, R-Score: %.3f, F-Score: %.3f" % (pScore, rScore, fScore))

# Save the model to use later
model.save('trained_models/200409_CNN_Builtup_11by11_PScore%.3f_RScore%.3f_FScore%.3f.h5' % (pScore, rScore, fScore)) 

"""

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

plt.imshow(prediction)
plt.show()

