#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Image classification with convolutional neural network for 
differentiating dicot and monocot root cross sections

# CURRENTLY FOR TESTING ONLY #
"""
__author__ = 'Michael Gruenstaeudl <m.gruenstaeudl@fu-berlin.de>'
__info__ = 'Image classification with machine learning for differentiating'\ 
           'dicot and monocot root cross sections'
__version__ = '2022.01.26.1700'

##################################################
# IMPORT OPERATIONS #
#####################

import numpy as np
from glob import glob
from cv2 import imread #, cvtColor, COLOR_BGR2RGB
from tensorflow.keras import backend, layers, models, preprocessing
from matplotlib import pyplot

##################################################
# SETTING VARIABLES #
#####################

# Root XS images
img_width, img_height = 150, 150
traindata_dir = '/home/michael_science/Desktop/TEMP/Dicot_monocot_rootXS/train_small/'
testdata_dir = '/home/michael_science/Desktop/TEMP/Dicot_monocot_rootXS/test_small/'

# Model configuration
batch_size = 10
n_epochs = 25
n_classes = 10
validation_split = 0.2

# Set dimensions of images
if backend.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

##################################################

# Load the model
#model = model.load_model(my_file_path, compile = True)

##################################################
# SETTING UP MODEL #
####################

# Define a sequential model (appropriate for plain stack of layers 
# where each layer has exactly one input and one output tensor.
model = models.Sequential()

# Two convolution layers, each followed by max-pooling layer
model.add(layers.Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))  # Max Pooling for spatial hierarchy
model.add(layers.Dropout(0.25))  # Invoke dropout against overfitting

# One convolution layer with increased number of filters, 
# followed by max-pooling layer
model.add(layers.Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.25))

# Flatten, then fully connected layer
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))

# Add layer that has a sigmoid activation
model.add(layers.Dropout(0.5))  # Invoke very high dropout number against overfitting
model.add(layers.Dense(1, activation='sigmoid'))

## ALTERNATIVE recommended to me:
#model.add(layers.Dense(256, activation='relu'))
#model.add(layers.Dense(no_classes, activation='softmax'))

# Compile the model
model.compile(loss='binary_crossentropy', 
              optimizer='rmsprop', 
              metrics=['accuracy'])

## ALTERNATIVE recommended to me:
#from tensorflow.keras.losses import sparse_categorical_crossentropy
#from tensorflow.keras.optimizers import Adam
#model.compile(loss=sparse_categorical_crossentropy,
#              optimizer=Adam(),
#              metrics=['accuracy'])

# Set up generators
traindata_gen = preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = traindata_gen.flow_from_directory(
        traindata_dir,
        target_size=(img_width, img_height),  # resize images to 150x150
        batch_size=batch_size,
        class_mode='binary')  # binary labels b/c 'loss=binary_crossentropy' above
validation_generator = traindata_gen.flow_from_directory(
        traindata_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='binary')

# Train the model
model.fit(train_generator,
          #batch_size=batch_size,
          #steps_per_epoch=2000 // batch_size,
          steps_per_epoch=200 // batch_size,
          epochs=n_epochs,
          validation_data=validation_generator,
          #validation_steps=800 // batch_size,
          validation_steps=80 // batch_size)

# Save model and weights after training
#model.save_model(model, 'Dicot_monocot_rootXS__model.mdl') ## Currently error: AttributeError: 'Sequential' object has no attribute 'save_model'
#model.save_weights('Dicot_monocot_rootXS__weights.h5')

##################################################
# LOADING TEST IMAGES #
#######################

# Read test images
test_images = []
for i in glob(testdata_dir+"*.jpg"):
    img = imread(i)
    #img = cvtColor(img, COLOR_BGR2RGB)
    img = img[0:img_width, 0:img_height]
    pyplot.imshow(img)
    test_images.append(img)
test_images = np.array(test_images)

# Reshape images, cast numbers to float32
test_images = test_images.reshape(test_images.shape[0], img_width, img_height, 3)
test_images = test_images.astype('float32')
#test_images = test_images / 255  # scale data

##################################################
# MAKING PREDICTIONS #
######################

# Evaluate test images
model.evaluate(test_images)

# Generate predictions for test images
predictions = model.predict(test_images)

# Generate arg maxes for predictions
#np.argmax(predictions, axis = 1)


##################################################
# EOF #
#######