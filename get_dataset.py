import os
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
import random

input_dir = ".\\TCI\\"
target_dir = ".\\ground-truth\\"
batch_size = 4
img_size = (64,64)

def get_paths(tiles):
    #input_img_paths = []
    #target_img_paths = []
    target_num = 0
    img_num = 0
    seperate_img_paths = {}
    seperate_target_paths = {}
    for tile in tiles:
        seperate_img_paths[tile] = sorted(
            [
                os.path.join("%s%s\\Patches"%(input_dir, tile), fname)
                for fname in os.listdir("%s%s\\Patches"%(input_dir, tile))
                if fname.endswith(".png")
            ]
        )
        img_num += len(seperate_img_paths[tile])
    for tile in tiles:
        seperate_target_paths[tile] = sorted(
            [
                os.path.join("%s%s\\Patches"%(target_dir, tile), fname)
                for fname in os.listdir("%s%s\\Patches"%(target_dir, tile))
                if fname.endswith(".png")
            ]
        )
        target_num += len(seperate_target_paths[tile])

    print("Number of training samples: ", img_num)
    print("Number of targets samples: ", target_num)

    return (seperate_img_paths, seperate_target_paths)


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
        #input_img = tf_image.convert_image_dtype(input_img, "float32")

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
            batch_size,
            img_size,
            all_train_input_img_paths,
            all_train_target_img_paths
    )
    full_valid_dataset = get_dataset(
        batch_size, img_size, all_val_input_img_paths, all_val_target_img_paths
    )
    print("Training with %d training inputs from tiles %s"%(len(all_train_input_img_paths), list(val_input_img_paths_by_tile.keys())))
    print("Training with %d target masks from tiles %s"%(len(all_train_target_img_paths), list(val_target_img_paths_by_tile.keys())))
    print("Validating and testing with %d inputs from tiles %s"%(len(all_val_input_img_paths), list(val_input_img_paths_by_tile.keys())))
    print("Validating and testing with %d target masks from tiles %s"%(len(all_val_target_img_paths), list(val_target_img_paths_by_tile.keys())))
    return (full_train_dataset, full_valid_dataset, val_input_img_paths_by_tile, val_target_img_paths_by_tile)

def get_control_group_images(input_img_paths, target_img_paths):# Split our img paths into a training and a validation set
    # Gets all paths for the tile
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    print(input_img_paths)
    # Gets the full training and testing paths
    full_control_group_dataset = get_dataset(
            batch_size,
            img_size,
            input_img_paths,
            target_img_paths
    )
    print("Control group has full dataset with %d input samples"%(len(input_img_paths)))
    print("Control group has full masking dataset with %d target masks"%(len(target_img_paths)))
    return (full_control_group_dataset, target_img_paths)
