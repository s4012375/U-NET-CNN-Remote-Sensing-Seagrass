import keras
from keras import layers
import matplotlib.pyplot as plt

import tensorflow as tf
import get_dataset as ds

## PARAMS
img_size = (64, 64)
num_classes = 9
batch_size = 20
epochs = 100

def get_untrained_model():
    inputs = keras.Input(shape=img_size + (3,))

    ### [First half of the network: downsampling inputs] ###

    # Entry block
    x = layers.Conv2D(32, 3, strides=2, padding="same")(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    previous_block_activation = x  # Set aside residual

    # Blocks 1, 2, 3 are identical apart from the feature depth.
    for filters in [64, 128, 256]:
        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # Project residual
        residual = layers.Conv2D(filters, 1, strides=2, padding="same")(
            previous_block_activation
        )
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    ### [Second half of the network: upsampling inputs] ###

    for filters in [256, 128, 64, 32]:
        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.Conv2DTranspose(filters, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.UpSampling2D(2)(x)

        # Project residual
        residual = layers.UpSampling2D(2)(previous_block_activation)
        residual = layers.Conv2D(filters, 1, padding="same")(residual)
        x = layers.add([x, residual])  # Add back residual
        previous_block_activation = x  # Set aside next residual

    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    show_trainable=True
    return model

def get_trained_model(model_file):
    model = keras.saving.load_model(model_file)

    for layer in model.layers[:30]:
        layer.trainable = False
    model.summary(show_trainable=True)
    return model


def train_base_model(train_dataset, valid_dataset, model, model_name):
    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )
    # Saves the weights for use later
    callbacks = [
        keras.callbacks.ModelCheckpoint("model_%s.keras"%model_name, save_best_only=True)
    ]
    history = model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=valid_dataset,
        callbacks=callbacks,
        verbose=2,
    )
    print(model.get_weights())

def transfer_learn_model(train_dataset, valid_dataset, model, model_name, tile_name):
    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )
    # Saves the weights for use later for a specific tile
    callbacks = [
        keras.callbacks.ModelCheckpoint("model_%d_%s.keras"%(model_name, tile_name), save_best_only=True)
    ]
    model.fit(
        train_dataset,
        epochs=epochs,
        validation_data=valid_dataset,
        callbacks=callbacks,
        verbose=2,
    )

def validate_model(val_input_img_paths, val_target_img_paths, model):
    # Generate predictions for all images in the validation set
    val_dataset = ds.get_dataset(
        batch_size, img_size, val_input_img_paths, val_target_img_paths
    )
    val_preds = model.predict(val_dataset)
    return val_preds
