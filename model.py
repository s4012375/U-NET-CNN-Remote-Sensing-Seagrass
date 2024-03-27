import keras
from keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

## PARAMS
CLASSES = 9
INPUT_SHAPE = (64,64,3)
EPOCHS = 100
BATCH_SIZE = 4

def getTrainableModel(base_weights, is_base_trainable):
    # ENCODER - feature extraction
    base_model = keras.applications.ResNet50(
        weights = base_weights, # Use existing weights
        input_shape=INPUT_SHAPE, # Input shape with 3 bands
        include_top=False # Using a custom decoder instead,
    )
    base_model.trainable = is_base_trainable
   
        
    ## ENCODER
    inputs = keras.Input(shape=INPUT_SHAPE)
    x = base_model(inputs, training=is_base_trainable)
    
    ## DECODER - classification
    # Convert upsamples to a higher level
    x = keras.layers.Conv2D(1,(3,3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D(size=(2,2))(x)
    x = keras.layers.Conv2D(1,(3,3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D(size=(2,2))(x)
    x = keras.layers.Conv2D(1,(3,3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D(size=(2,2))(x)
    x = keras.layers.Conv2D(1,(3,3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D(size=(2,2))(x)
    x = keras.layers.Conv2D(1,(3,3), activation='relu', padding='same')(x)
    x = keras.layers.UpSampling2D(size=(2,2))(x)
    # A Dense classifier with a 9 classes
    outputs = keras.layers.Softmax(axis=-1)(x)
    model = keras.Model(inputs, outputs)

    compile_model(model)
    model.summary(show_trainable=True)
    return model


def compile_model(model):
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

def base_model_train(train_inputs, target_outputs , model):
    model.fit(train_inputs, target_outputs, epochs=EPOCHS, batch_size=BATCH_SIZE)
    return model
