import keras
from keras import layers
import matplotlib.pyplot as plt

import tensorflow as tf
import get_dataset as ds

## PARAMS
img_size = (64, 64)
num_classes = 9
batch_size = 20
base_epochs = 100
transfer_epochs = 50


# Residual Convolutional Block
def rcu(x, filters):
    residual = x
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters*2, 3, padding="same")(x)
    
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.add([x, residual])
    return x

# Chained Residual Pooling
def crp(x, filters, pooling):
    x = layers.Activation("relu")(x)
    residual = x
    x = layers.MaxPooling2D((pooling, pooling), strides = 1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.add([x, residual])
    x = layers.MaxPooling2D((pooling, pooling), strides = 1, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 3, padding="same")(x)
    x = layers.add([x, residual])
    return x

def get_imagenet_resnet_model():
    ## [First half of the model is resnet encoder] ##
    resnet = keras.applications.ResNet50(
        include_top=False,
        input_shape=(64,64,3),
        weights = 'imagenet'
    )
    resnet.trainable = False
    inputs = keras.Input(shape=(64,64,3))
    x = resnet(inputs)
    
    ## [Middle of the model is bridging] ##
    # Bridge Block 1
    bridge_residual1 = resnet.get_layer('conv2_block1_out').output # out shape (16,16,256)
    bridge_residual1 = rcu(bridge_residual1, 256)
    bridge_residual1 = rcu(bridge_residual1, 256)
    bridge_residual1 = layers.BatchNormalization()(bridge_residual1)
    bridge_residual1 = layers.Conv2D(256,3,padding="same")(bridge_residual1)

    # Bridge Block 2
    bridge_residual2 = resnet.get_layer('conv3_block2_out').output # out shape (8,8,512)
    bridge_residual2 = rcu(bridge_residual2, 512)
    bridge_residual2 = rcu(bridge_residual2, 512)
    bridge_residual2 = layers.BatchNormalization()(bridge_residual2)
    bridge_residual2 = layers.Conv2D(512,3,padding="same")(bridge_residual2)
    
    # Bridge Block 3
    bridge_residual3 = resnet.get_layer('conv4_block3_out').output # out shape (4,4,1024)
    bridge_residual3 = rcu(bridge_residual3, 1024)
    bridge_residual3 = rcu(bridge_residual3, 1024)
    bridge_residual3 = layers.BatchNormalization()(bridge_residual3)
    bridge_residual3 = layers.Conv2D(1024,3,padding="same")(bridge_residual3)

    # Bridge Block 4
    
    bridge_residual4 = x # out shape (2,2,2048)
    bridge_residual4 = rcu(bridge_residual4, 2048)
    bridge_residual4 = rcu(bridge_residual4, 2048)
    bridge_residual4 = layers.BatchNormalization()(bridge_residual4)
    bridge_residual4 = layers.Conv2D(2048,3,padding="same")(bridge_residual4)
    previous_block_activation = bridge_residual4

    
    ### ResNet50 encoder ###
    previous_block_activation = x
    #x = previous_block_activation
    
    ### [Second half of the network: upsampling inputs] ###
    for (filters, residual) in [(1024, bridge_residual3), (512, bridge_residual2), (256, bridge_residual1)]:
    #for filters in [1024, 512, 256]:
        x = layers.Activation("relu")(previous_block_activation)
        x = crp(x, filters*2, 5)
        x = rcu(x, filters*2)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.UpSampling2D(2)(x)
        #x = layers.add([x, residual])

        previous_block_activation = x  # Set aside next residual
        
    # Final layer
    x = layers.Activation("relu")(previous_block_activation)
    x = crp(x, 256, 1)
    x = rcu(x, 256)
    x = layers.UpSampling2D(2)(x)
    x = rcu(x, 256)
    x = layers.UpSampling2D(2)(x)
    
    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    model.summary(show_trainable=True)
    return model

def get_trained_resnet_model(model_file):
     
 ## [First half of the model is resnet encoder] ##
    resnet = keras.applications.ResNet50(
        include_top=False,
        input_shape=(64,64,3),
        weights = 'imagenet'
    )
    resnet.load_weights(model_file)
    resnet.trainable = False
    
    inputs = keras.Input(shape=(64,64,3))
    x = resnet(inputs)
    
    ## [Middle of the model is bridging] ##
    # Bridge Block 1
    bridge_residual1 = resnet.get_layer('conv2_block1_out').output # out shape (16,16,256)
    bridge_residual1 = rcu(bridge_residual1, 256)
    bridge_residual1 = rcu(bridge_residual1, 256)
    bridge_residual1 = layers.BatchNormalization()(bridge_residual1)
    bridge_residual1 = layers.Conv2D(256,3,padding="same")(bridge_residual1)

    # Bridge Block 2
    bridge_residual2 = resnet.get_layer('conv3_block2_out').output # out shape (8,8,512)
    bridge_residual2 = rcu(bridge_residual2, 512)
    bridge_residual2 = rcu(bridge_residual2, 512)
    bridge_residual2 = layers.BatchNormalization()(bridge_residual2)
    bridge_residual2 = layers.Conv2D(512,3,padding="same")(bridge_residual2)
    
    # Bridge Block 3
    bridge_residual3 = resnet.get_layer('conv4_block3_out').output # out shape (4,4,1024)
    bridge_residual3 = rcu(bridge_residual3, 1024)
    bridge_residual3 = rcu(bridge_residual3, 1024)
    bridge_residual3 = layers.BatchNormalization()(bridge_residual3)
    bridge_residual3 = layers.Conv2D(1024,3,padding="same")(bridge_residual3)

    # Bridge Block 4
    
    bridge_residual4 = x # out shape (2,2,2048)
    bridge_residual4 = rcu(bridge_residual4, 2048)
    bridge_residual4 = rcu(bridge_residual4, 2048)
    bridge_residual4 = layers.BatchNormalization()(bridge_residual4)
    bridge_residual4 = layers.Conv2D(2048,3,padding="same")(bridge_residual4)
    previous_block_activation = bridge_residual4

    
    ### ResNet50 encoder ###
    previous_block_activation = x
    #x = previous_block_activation
    
    ### [Second half of the network: upsampling inputs] ###
    for (filters, residual) in [(1024, bridge_residual3), (512, bridge_residual2), (256, bridge_residual1)]:
    #for filters in [1024, 512, 256]:
        x = layers.Activation("relu")(previous_block_activation)
        x = crp(x, filters*2, 5)
        x = rcu(x, filters*2)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(filters, 3, padding="same")(x)
        x = layers.UpSampling2D(2)(x)
        #x = layers.add([x, residual])

        previous_block_activation = x  # Set aside next residual
        
    # Final layer
    x = layers.Activation("relu")(previous_block_activation)
    x = crp(x, 256, 1)
    x = rcu(x, 256)
    x = layers.UpSampling2D(2)(x)
    x = rcu(x, 256)
    x = layers.UpSampling2D(2)(x)
    
    # Add a per-pixel classification layer
    outputs = layers.Conv2D(num_classes, 3, activation="softmax", padding="same")(x)

    # Define the model
    model = keras.Model(inputs, outputs)
    
    # Sets all layers prior to the last 20 to trainable
    for l in model.layers[:-25]:
        l.trainable = False
    model.summary(show_trainable=True)
    return model

def get_full_trained_resnet_model(model_file):
    model = keras.models.load_model(model_file)
    
    model.summary(show_trainable=True)
    return model


def train_base_model(train_dataset, valid_dataset, base_model, model_name):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'full_resnet_checkpoint_model_%s.keras'%(model_name),
        monitor='val_accuracy',
        verbose=0,
        save_best_only=True,
    )
    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    base_model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )
    base_model.fit(
        train_dataset,
        epochs=base_epochs,
        validation_data=valid_dataset,
        callbacks=checkpoint,
        verbose=2
    )
    # Saves the weights for use later
    base_model.save("full_model_%s.keras"%model_name)
    return base_model

def transfer_learn_model(train_dataset, valid_dataset, tile_model, model_name, tile_name):
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        'full_resnet_checkpoint_model_%s_%s.keras'%(model_name, tile_name),
        monitor='accuracy',
        verbose=0,
        save_best_only=True,
    )
    # Configure the model for training.
    # We use the "sparse" version of categorical_crossentropy
    # because our target data is integers.
    tile_model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=['accuracy']
    )
    tile_model.fit(
        train_dataset,
        epochs=transfer_epochs,
        validation_data=valid_dataset,
        callbacks=checkpoint,
        verbose=2
    )
    tile_model.save("full_model_%s_%s.keras"%(model_name,tile_name))
    return tile_model

def validate_model(val_input_img_paths, val_target_img_paths, model):
    # Generate predictions for all images in the validation set
    val_dataset = ds.get_dataset(
        batch_size, img_size, val_input_img_paths, val_target_img_paths
    )
    val_preds = model.predict(val_dataset, verbose=0)
    return val_preds
