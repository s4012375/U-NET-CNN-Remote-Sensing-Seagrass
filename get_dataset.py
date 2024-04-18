import os
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
import random
import model

import keras

INPUT_IMG_DIR = ".\\TCI\\"
TARGET_DIR = ".\\ground-truth\\"
BATCH_SIZE = model.BATCH_SIZE
IMG_SIZE = (64,64)

# Gets all paths for a given tile
def get_paths(tiles):
    target_num = 0
    img_num = 0
    seperate_img_paths = {}
    seperate_target_paths = {}
    for tile in tiles:
        seperate_img_paths[tile] = sorted(
            [
                os.path.join("%s%s\\Patches"%(INPUT_IMG_DIR, tile), fname)
                for fname in os.listdir("%s%s\\Patches"%(INPUT_IMG_DIR, tile))
                if fname.endswith(".png")
            ]
        )
        img_num += len(seperate_img_paths[tile])
    for tile in tiles:
        seperate_target_paths[tile] = sorted(
            [
                os.path.join("%s%s\\Patches"%(TARGET_DIR, tile), fname)
                for fname in os.listdir("%s%s\\Patches"%(TARGET_DIR, tile))
                if fname.endswith(".png")
            ]
        )
        target_num += len(seperate_target_paths[tile])

    print("Number of training samples: ", img_num)
    print("Number of targets samples: ", target_num)

    return (seperate_img_paths, seperate_target_paths)

# Reads all files and constructs the TF dataset
def get_dataset(
    batch_size,
    img_size,
    input_img_paths,
    target_img_paths,
    max_dataset_len=None,
):
    """Returns a TF Dataset."""

    def load_img_masks(input_img_path, target_img_path):
        input_img = tf_io.read_file(input_img_path)
        input_img = tf_io.decode_png(input_img, channels=3)
        input_img = tf_image.convert_image_dtype(input_img, "float32")
        input_img = keras.applications.resnet.preprocess_input(input_img)
        

        target_img = tf_io.read_file(target_img_path)
        target_img = tf_io.decode_png(target_img, channels=1)
        target_img = tf_image.convert_image_dtype(target_img, "uint8")
        
        return (input_img, target_img)

    # For faster debugging, limit the size of data
    if max_dataset_len:
        input_img_paths = input_img_paths[:max_dataset_len]
        target_img_paths = target_img_paths[:max_dataset_len]

    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    dataset = dataset.map(
        load_img_masks,
        num_parallel_calls=tf_data.AUTOTUNE 
    )
    return dataset.batch(batch_size)

# Split the images according to a test_divide (number fo testing ings to use)
def train_test_split(input_img_paths, target_img_paths, test_divide):# Split our img paths into a training and a validation set
    # Maps used for testing model on each tile
    val_input_img_paths_by_tile = {}
    val_target_img_paths_by_tile = {}
    # Complete set of image/mask paths for training and validation
    all_train_input_img_paths = []
    all_train_target_img_paths = []
    all_val_input_img_paths = []
    all_val_target_img_paths = []
    # Gets all paths for the tile
    for tile in input_img_paths.keys():
        random.Random(1337).shuffle(input_img_paths[tile])
        random.Random(1337).shuffle(target_img_paths[tile])
        # Adds them to the tilewise paths training and testing
        tile_train_input_img_paths = input_img_paths[tile][:-test_divide] # Training samples
        tile_train_target_img_paths = target_img_paths[tile][:-test_divide] # Training targets
        tile_val_input_img_paths = input_img_paths[tile][-test_divide:] # Validation samples
        tile_val_target_img_paths = target_img_paths[tile][-test_divide:] # Validation targets
        # Adds them to the overall tile paths
        all_train_input_img_paths += tile_train_input_img_paths
        all_train_target_img_paths += tile_train_target_img_paths
        all_val_input_img_paths += tile_val_input_img_paths
        all_val_target_img_paths += tile_val_target_img_paths
        # Keeps a tile-wise record of the input and targets for validation 
        val_input_img_paths_by_tile[tile] = tile_val_input_img_paths
        val_target_img_paths_by_tile[tile] = tile_val_target_img_paths
    # Gets the full training and testing paths
    full_train_dataset = get_dataset(
            BATCH_SIZE,
            IMG_SIZE,
            all_train_input_img_paths,
            all_train_target_img_paths
    )
    full_valid_dataset = get_dataset(
        BATCH_SIZE, IMG_SIZE, all_val_input_img_paths, all_val_target_img_paths
    )
    print("Training with %d training inputs from tiles %s"%(len(all_train_input_img_paths), list(val_input_img_paths_by_tile.keys())))
    print("Training with %d target masks from tiles %s"%(len(all_train_target_img_paths), list(val_target_img_paths_by_tile.keys())))
    print("Validating and testing with %d inputs from tiles %s"%(len(all_val_input_img_paths), list(val_input_img_paths_by_tile.keys())))
    print("Validating and testing with %d target masks from tiles %s"%(len(all_val_target_img_paths), list(val_target_img_paths_by_tile.keys())))
    return (full_train_dataset, full_valid_dataset, val_input_img_paths_by_tile, val_target_img_paths_by_tile)