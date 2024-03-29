import os
from tensorflow import data as tf_data
from tensorflow import image as tf_image
from tensorflow import io as tf_io
import random

input_dir = ".\\TCI\\20170717\\Patches"
target_dir = ".\\ground-truth\\20170717\\Patches"
batch_size = 4
img_size = (64,64)
def get_paths():
    input_img_paths = sorted(
        [
            os.path.join(input_dir, fname)
            for fname in os.listdir(input_dir)
            if fname.endswith(".png")
        ]
    )
    target_img_paths = sorted(
        [
            os.path.join(target_dir, fname)
            for fname in os.listdir(target_dir)
            if fname.endswith(".png")
        ]
    )

    print("Number of training samples: ", len(input_img_paths))
    print("Number of targets samples: ", len(input_img_paths))

    return (input_img_paths, target_img_paths)


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


def train_test_split(input_img_paths, target_img_paths):# Split our img paths into a training and a validation set
    val_samples = 50
    random.Random(1337).shuffle(input_img_paths)
    random.Random(1337).shuffle(target_img_paths)
    train_input_img_paths = input_img_paths[:-val_samples] # Training samples
    train_target_img_paths = target_img_paths[:-val_samples] # Training targets
    val_input_img_paths = input_img_paths[-val_samples:] # Validation samples
    val_target_img_paths = target_img_paths[-val_samples:] # Validation targets

    # Instantiate dataset for each split
    # Limit input files in `max_dataset_len` for faster epoch training time.
    # Remove the `max_dataset_len` arg when running with full dataset.
    train_dataset = get_dataset(
        batch_size,
        img_size,
        train_input_img_paths,
        train_target_img_paths,
        max_dataset_len=1000,
    )
    valid_dataset = get_dataset(
        batch_size, img_size, val_input_img_paths, val_target_img_paths
    )
    return (train_dataset, valid_dataset, val_input_img_paths, val_target_img_paths)
