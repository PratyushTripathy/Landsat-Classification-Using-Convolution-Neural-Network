"""
(2/3)
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
import tensorflow as tf
from sklearn.utils import resample

from tensorflow import keras
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

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
if not os.path.exists(os.path.join(os.getcwd(), 'trained_model')):
    os.mkdir(os.path.join(os.getcwd(), 'trained_model'))
    
model.save('trained_models/200409_CNN_Builtup_11by11_PScore%.3f_RScore%.3f_FScore%.3f.h5' % (pScore, rScore, fScore)) 
