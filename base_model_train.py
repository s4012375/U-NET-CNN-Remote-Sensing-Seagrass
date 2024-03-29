import rasterio
import random
import math
import numpy as np
import model as cnn_model
import keras

dataset = images = ['20191211','20191027','20190627','20190227',
                    '20190123','20181224','20181010','20180624',
                    '20171126','20171116','20170717','20170525',
                    '20170125','20161226','20161129','20161116',
                    '20160609','20160420','20160210','20151125']
model_configs = {
    '1':[0.8, # 80% training, 20% testing for base model
        ['20170717'],  
        ['20160609', '20180624', '20181010', '20191027']],
    '2':[0.6, # 60% training, 40% testing for base model
        [ '20170717','20190627'], 
        ['20160609', '20180624', '20181010']],
    '3':[0.6, # 60% training, 40% testing for base model
        ['20160609', '20170717', '20180624', '20181010','20190627'],
        ['20191211','20191027','20190627','20190227', '20190123',
          '20181224','20181010','20180624','20171126','20171116',
          '20170717','20170525','20170125','20161226','20161129',
          '20161116','20160609','20160420','20160210','20151125']],
    '4':[0.6, # 60% training, 40% testing for base model
        ['20191211','20191027','20190123','20181224','20171126','20171116','20170125','20161226','20160609','20160420'], 
        ['20190627','20190227','20181010','20180624','20170717','20170525','20161129','20161116','20160210','20151125']]}

# 251 patches
area_patches = {'Fenham': (8, 11), 'Budle': (6,2), 'Beadnell': (3,19), 'Embleton': (2,13)}

# Testing dataset for control
control_testing = []

# Training and testing combined dataset for optimisation/cross tile classifications
cross_ref_training = {}
cross_ref_testing = {}

def get_dataset(
    batch_size,
    img_size,
    input_img_paths,
    target_img_paths,
    max_dataset_len=None,
):
    def load_img_masks(input_img_path, target_img_path):
        input_img = tf_io.read_file(input_img_path)
        input_img = tf_io.decode_png(input_img, channels=3)
        input_img = tf_image.resize(input_img, img_size)
        input_img = tf_image.convert_image_dtype(input_img, "float32")

        target_img = tf_io.read_file(target_img_path)
        target_img = tf_io.decode_png(target_img, channels=1)
        target_img = tf_image.resize(target_img, img_size, method="nearest")
        target_img = tf_image.convert_image_dtype(target_img, "uint8")

        # Ground truth labels are 1, 2, 3. Subtract one to make them 0, 1, 2:
        target_img -= 1
        return input_img, target_img

    # For faster debugging, limit the size of data
    if max_dataset_len:
        input_img_paths = input_img_paths[:max_dataset_len]
        target_img_paths = target_img_paths[:max_dataset_len]
    dataset = tf_data.Dataset.from_tensor_slices((input_img_paths, target_img_paths))
    dataset = dataset.map(load_img_masks, num_parallel_calls=tf_data.AUTOTUNE)
    return dataset.batch(batch_size)

def random_sample(all_TCI_patches, all_GT_patches, sample_size):
    training_TCI_patches = []
    testing_TCI_patches = []
    training_target_patches = []
    testing_target_patches = []
    # Create sample of a given size:`sample_size`
    samples = random.sample(range(0, 251), sample_size)
    for i in range(0, 251):
        processed_TCI = keras.applications.resnet.preprocess_input(all_TCI_patches[i])
        processed_GT = all_GT_patches[i]
        if i in samples:
            training_TCI_patches.append(processed_TCI)
            training_target_patches.append(processed_GT)
        else:
            testing_TCI_patches.append(processed_TCI)
            testing_target_patches.append(processed_GT)
    return (training_TCI_patches, training_target_patches, testing_TCI_patches, testing_target_patches)

def getModelDataset(model):
    # Training and testing compiled dataset for base model
    base_model_training_ds = []
    base_model_testing_ds = []
    base_model_training_target = []
    base_model_testing_target = []
    # Get dataset for Model 1
    for training_tile in model_configs[model][1]:
        tile_TCI_patches = []
        tile_GT_patches = []
        tile_training_patches = []
        tile_testing_patches = []
        # Reads the raster file for the ground-truth
        for patch_name, patch_size in area_patches.items():
            for x in range(0, patch_size[0]+1):
                for y in range(0, patch_size[1]+1):
                    with rasterio.open('.\\TCI\\%s\\Patches\\%s_image_%d_%d.tif'%(training_tile, patch_name, x, y), 'r') as ds:
                        tci_rgb = ds.read()
                        #tci_rgb = np.moveaxis(tci_rgb, 0, 2) # SEMANTIC SEGMENTATION WANTS IN ORIGINAL ORDER
                        tile_TCI_patches.append(tci_rgb)
                    with rasterio.open('.\\ground-truth\\%s\\Patches\\%s_image_%d_%d.tif'%(training_tile, patch_name, x, y), 'r') as ds:
                        ground_truth_patch = ds.read()
                        #ground_truth_patch = np.moveaxis(ground_truth_patch, 0, 2) # SEMANTIC SEGMENTATION WANTS IN ORIGINAL ORDER
                        tile_GT_patches.append(ground_truth_patch)
                        
        # Create sample of x% training and y% testing           
        tile_training_patches, tile_training_targets, tile_testing_patches, tile_testing_targets = random_sample(tile_TCI_patches, tile_GT_patches, math.ceil(model_configs[model][0]*251))
        base_model_training_ds = base_model_training_ds + tile_training_patches
        base_model_testing_ds = base_model_testing_ds + tile_testing_patches
        base_model_training_target = base_model_training_target + tile_training_targets
        base_model_testing_target = base_model_testing_target + tile_testing_targets
    print("Retrieved base model training and testing data...")
    return (np.array(base_model_training_ds), np.array(base_model_training_target), np.array(base_model_testing_ds), np.array(base_model_testing_target))

MODEL = '1'
training_TCI_ds, training_target_ds, testing_TCI_ds, testing_target_ds = getModelDataset(MODEL)
print("Generated training inputs with ",training_TCI_ds.shape[0], ' images ', training_TCI_ds.shape[1], 'x', training_TCI_ds.shape[2], 'px with ', training_TCI_ds.shape[3], ' bands/channels')
print("Generated training targets with ",training_target_ds.shape[0], ' images ', training_target_ds.shape[1], 'x', training_target_ds.shape[2], 'px with ', training_target_ds.shape[3], ' bands/channels')
print("Generated testing inputs with ",testing_TCI_ds.shape[0], ' images ', testing_TCI_ds.shape[1], 'x', testing_TCI_ds.shape[2], 'px with ', testing_TCI_ds.shape[3], ' bands/channels')
print("Generated testing targets with ",testing_target_ds.shape[0], ' images ', testing_target_ds.shape[1], 'x', testing_target_ds.shape[2], 'px with ', testing_target_ds.shape[3], ' bands/channels')

raw_model = cnn_model.getSemanticSegmentationModel(True)
base_model = cnn_model.base_model_train(training_TCI_ds, training_target_ds, raw_model)
results = base_model.predict(testing_TCI_ds, batch_size=4)
stats = base_model.evaluate(testing_TCI_ds, testing_target_ds, batch_size = 4)
i = 0
for result in results:
    tiff.imwrite('./RESULTS/model ' + MODEL + '/original_test_'+i+'.tif', result)
    print('Outputed results patch ', i)
area_i = area_i + 1


