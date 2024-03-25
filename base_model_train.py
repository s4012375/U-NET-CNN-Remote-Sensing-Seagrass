import numpy as np
import keras
from keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

## PARAMS
CLASSES = 9
INPUT_SHAPE = (64,64,3)

dataset = images = ['20191211','20191027','20190627','20190227', '20190123',
          '20181224','20181010','20180624','20171126','20171116',
          '20170717','20170525','20170125','20161226','20161129',
          '20161116','20160609','20160420','20160210','20151125']


# ENCODER - feature extraction
base_model = keras.applications.ResNet50(
    weights = None, # Generate new weights at random
    input_shape=INPUT_SHAPE, # Input shape with 3 bands
    include_top=False # Using a decoder instead,
    classes = CLASSES
)


## DECODER - classification
inputs = keras.Input(shape=INPUT_SHAPE)
# Convert features of shape `base_model.output_shape[1:]` to vectors
x = keras.layers.GlobalAveragePooling2D()(x)
# Converts from 2d to 1d
x = keras.layers.flatten()
# Also try 128 here, 256, etc.
keras.layers.Dense(64, activation=activations.relu))
# A Dense classifier with a 9 classes
outputs = keras.layers.Dense(9, activation=activation.softmax)(x)

model = keras.Model(inputs, outputs)


