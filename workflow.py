import keras
import numpy as np

from PIL import ImageOps
from keras.utils import load_img
from IPython.display import Image
import os

import get_dataset as ds
import model as u_net
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

patches_per_tile = 119
model_configs = {
    '1':[24, 
        ['20170717'],  # 80% training, 20% testing for base model
        ['20160609', '20180624', '20181010', '20191027']], # No-retraining and optimisation with individual 60% training for each tile and testing on 40%
    '2':[47, 
        ['20170717','20190627', '20181010'], # 60% training, 40% testing for base model
        ['20160609', '20180624']], # No-retraining and optimisation with individual 60% training for each tile and testing on 40%
    '3':[47, 
        ['20171126', '20171116', '20170525', '20170125', '20161226', '20161129', '20161116', '20160420', '20160210', '20151125'],# 60% training, 40% testing for base model
        ['20191211', '20191027', '20190227', '20190123', '20181224']], # No-retraining and optimisation with individual 60% training for each tile and testing on 40%
    '4':[47,
        ['20191211', '20191027', '20190123', '20181224', '20181010', '20180624', '20171126', '20171116', '20170125', '20161226', '20160609', '20160420'], # 60% training, 40% testing for base model 
        ['20190627', '20190227', '20170717', '20170525', '20160210', '20151125', '20161129', '20161116']],
    '5':[20,
        ['20191211', '20191027', '20190227', '20190123', '20181224', '20181010', '20180624', '20171126', '20171116', '20170125', '20161226', '20160609', '20160420','20151125', '20161129', '20161116'], # 60% training, 40% testing for base model 
        ['20190627', '20170717', '20170525', '20160210']]# No-retraining and optimisation with individual 60% training for each tile and testing on 40%
}

classes = ['Land', 'Sand', 'Seaweed','Salt marsh', 'Water', 'Seagrass']

MODEL = '5' # THIS IS THE MODEL TO BE RUN


# Displays the model success metrics and saves them to a file
def evaluate_tile(val_preds, val_target_img_paths, average_precision, tile, stage):
    os.system('rm -rf .\\RESULTS\\model %s\\%s\\%s\\*'%(MODEL, stage, tile)) # Empties the result directory so the results start from scratch
    average_f1=np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    count_f1 = np.array([0,0,0,0,0,0])
    average_precision=np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    count_precision = np.array([0,0,0,0,0,0])
    average_recall = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    count_recall = np.array([0,0,0,0,0,0])
    average_accuracy=0
    
    # For each image in the returned dataset evaluate its performance
    for j in range(0, len(val_preds)):
        current_path = val_target_img_paths[j]
        img_name = current_path.split('\\')[-1]
        predicted_mask = np.argmax(val_preds[j], axis=-1)
        predicted_mask = np.expand_dims(predicted_mask, axis=-1)
        predicted_array = predicted_mask.flatten()
        actual = np.moveaxis(np.asarray(load_img(current_path)), 2, 0)[0:1].flatten()
        # Calculates f1 stats for this tile
        f1 = f1_score(actual, predicted_array, labels=[0,1,2,3,4,5], average=None, zero_division=np.nan)
        for i in range(0, len(classes)):
            if (not np.isnan(f1[i])):
                average_f1[i] += f1[i]
                count_f1[i] += 1
        # Calculates precision stats for this tile
        precision = precision_score(actual, predicted_array, labels=[0,1,2,3,4,5], average=None, zero_division=np.nan)
        for i in range(0, len(classes)):    
            if (not np.isnan(precision[i])):
                average_precision[i] += precision[i]
                count_precision[i] += 1
        # Calculates recall stats for this tile
        recall = recall_score(actual, predicted_array, labels=[0,1,2,3,4,5], average=None, zero_division=np.nan)
        for i in range(0, len(classes)):
            if (not np.isnan(recall[i])):
                    average_recall[i] += recall[i]
                    count_recall[i] += 1
        average_accuracy += accuracy_score(actual, predicted_array)

        # Saves result to a file
        mask = np.argmax(val_preds[j], axis=-1)
        mask = np.expand_dims(mask, axis=-1)
        img = keras.utils.array_to_img(mask)
        img.save('.\\RESULTS\\model %s\\%s\\%s\\%s'%(MODEL, stage, tile, img_name))
    for i in range(0, len(classes)):
        if (count_f1[i] != 0):
            average_f1[i] = average_f1[i] / count_f1[i]
    for i in range(0, len(classes)):
        if (count_precision[i] != 0):
            average_precision[i] = average_precision[i] / count_precision[i]
    for i in range(0, len(classes)):
        if (count_recall[i] != 0):
            average_recall[i] = average_recall[i] / count_recall[i]
    average_accuracy = average_accuracy / len(val_preds)
    print('{} accuracy: {}'.format(tile, average_accuracy))
    # Writes tile-wise log of results
    with open('.\\RESULTS\\model %s\\%s\\%s\\evaluation.txt'%(MODEL,stage,tile), 'w') as f:
        f.write("              -- %s --- %s --- %s -- %s -- %s -- %s \n"%(classes[0], classes[1], classes[2], classes[3], classes[4], classes[5]))
        f.write("F1            %f  %f  %f  %f  %f  %f \n"%(average_f1[0], average_f1[1], average_f1[2], average_f1[3], average_f1[4], average_f1[5]))  # Writes F1 to log file
        f.write("Recall        %f  %f  %f  %f  %f  %f \n"%(average_recall[0], average_recall[1], average_recall[2], average_recall[3], average_recall[4], average_recall[5]))  # Writes F1 to log file
        f.write("Precision     %f  %f  %f  %f  %f  %f \n"%(average_precision[0], average_precision[1], average_precision[2], average_precision[3], average_precision[4], average_precision[5]))  # Writes F1 to log file
        f.write("Overall Accuracy %f \n"%(average_accuracy))  # Writes F1 to log file
        
    # When all patch values are nan for a class the average falsely defaults to zero which will impact model average
    # Returning an array which detects invalid zero values
    valid_f1s = [0,0,0,0,0,0]
    valid_recalls = [0,0,0,0,0,0]
    valid_precisions = [0,0,0,0,0,0]
    for i in range(0, len(classes)):
        if (count_f1[i] > 0):
              valid_f1s[i] = 1
    for i in range(0, len(classes)):
        if (count_recall[i] > 0):
              valid_recalls[i] = 1
    for i in range(0, len(classes)):
        if (count_precision[i] > 0):
              valid_precisions[i] = 1
    return average_f1, average_recall, average_precision, average_accuracy, valid_f1s, valid_recalls, valid_precisions

def evaluate_model(image_paths, target_paths, model,stage):
     ## VALIDATES FOR REMAINDER OF EACH TRAINING TILE AND SAVES RESULT
    average_f1=np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    count_all_f1 = np.array([0,0,0,0,0,0])
    average_precision=np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    count_all_precision = np.array([0,0,0,0,0,0])
    average_recall = np.array([0.0,0.0,0.0,0.0,0.0,0.0])
    count_all_recall = np.array([0,0,0,0,0,0])
    average_accuracy=0
    for tile in image_paths.keys():
        # Make predictions of segments and average_precision using the model on different subset of the tiles used in training
        val_preds = u_net.validate_model(image_paths[tile], target_paths[tile], model)
        # Evaluates the model
        f1, recall, precision, accuracy, cnt_f1, cnt_recall, cnt_precision = evaluate_tile(val_preds, target_paths[tile], MODEL, tile, stage)
        average_f1 += f1
        average_recall += recall
        average_precision += precision
        average_accuracy += accuracy
        count_all_f1 += cnt_f1 # Counts how many valid f1s there are across tiles
        count_all_recall += cnt_recall # Counts how many valid f1s there are across tiles
        count_all_precision += cnt_precision # Counts how many valid f1s there are across tiles
    
    for i in range(0, len(classes)):
        if (count_all_f1[i] != 0):
            average_f1[i] = average_f1[i] / count_all_f1[i]
        else:
            average_f1[i] = np.nan
    for i in range(0, len(classes)):
        if (count_all_precision[i] != 0):
            average_precision[i] = average_precision[i] / count_all_precision[i]
        else:
            average_precision[i] = np.nan
    for i in range(0, len(classes)):
        if (count_all_recall[i] != 0):
            average_recall[i] = average_recall[i] / count_all_recall[i]
        else:
            average_recall[i] = np.nan
    average_accuracy=average_accuracy / len(image_paths)
    print('Overall accuracy: {}'.format(average_accuracy))
    
    # Writes overall model log of results
    with open('.\\RESULTS\\model %s\\%s\evaluation.txt'%(MODEL, stage), 'w') as f:
        f.write("              -- %s --- %s --- %s -- %s -- %s -- %s \n"%(classes[0], classes[1], classes[2], classes[3], classes[4], classes[5]))
        f.write("F1            %f  %f  %f  %f  %f  %f \n"%(average_f1[0], average_f1[1], average_f1[2], average_f1[3], average_f1[4], average_f1[5]))  # Writes F1 to log file
        f.write("Recall        %f  %f  %f  %f  %f  %f \n"%(average_recall[0], average_recall[1], average_recall[2], average_recall[3], average_recall[4], average_recall[5]))  # Writes F1 to log file
        f.write("Precision     %f  %f  %f  %f  %f  %f \n"%(average_precision[0], average_precision[1], average_precision[2], average_precision[3], average_precision[4], average_precision[5]))  # Writes F1 to log file
        f.write("Overall Accuracy %f \n"%(average_accuracy))  # Writes F1 to log file

def run_model_workflow():

    ## GENRATES TRAINING/TESTING DATA AND GETS PATHS
    image_paths, target_paths = ds.get_paths(model_configs[MODEL][1])
    train_dataset, valid_dataset, val_input_img_paths_by_tile, val_target_img_paths_by_tile = ds.train_test_split(image_paths, target_paths, model_configs[MODEL][0])
    
    ## BUILD MODEL FROM SCRATCH
    base_model = u_net.get_imagenet_resnet_model()
    
    ## RUNS THE MODEL
    # Train the model doing validation at the end of each epoch.
    base_model = u_net.train_base_model(train_dataset, valid_dataset, base_model, MODEL)
    evaluate_model(image_paths, target_paths, base_model, 'Training')

    ##  --- USE IF BASE MODEL STOPPED PREMATURELY AND YOU WANT TO CONTINUE TRAINING ---
##    image_paths, target_paths = ds.get_paths(model_configs[MODEL][1])
##    train_dataset, valid_dataset, val_input_img_paths_by_tile, val_target_img_paths_by_tile = ds.train_test_split(image_paths, target_paths, model_configs[MODEL][0])# Use these two is continuing to train
##    base_model = u_net.train_base_model(train_dataset, valid_dataset, base_model, MODEL) # Use these two is continuing to train
    
    ##  --- TESTS OJ OWN TRAINING TILES IF THE MODEL WAS STOPPED EARLY ---
##    image_paths, target_paths = ds.get_paths(model_configs[MODEL][1])
##    base_model = u_net.get_full_trained_resnet_model('full_resnet_checkpoint_model_' + MODEL + '.keras')  
##    evaluate_model(image_paths, target_paths, base_model, 'Training')
##

    ## RUNS CONTROL GROUP
    base_model = u_net.get_full_trained_resnet_model('full_resnet_checkpoint_model_' + MODEL + '.keras')
    training_paths, target_paths = ds.get_paths(model_configs[MODEL][2])
    ## EVALUATES THE MODEL WITHOUT RETRAININGON NEW TILES
    evaluate_model(training_paths, target_paths, base_model, 'Control')

    ## TRAIN FOR EACH TILE SPECIFICALLY
    for tile in model_configs[MODEL][2]:
        tile_model = u_net.get_trained_resnet_model('full_resnet_checkpoint_model_' + MODEL + '.keras')
        train_dataset, valid_dataset, val_input_img_paths_for_tile, val_target_img_paths_for_tile = ds.train_test_split({tile: training_paths[tile]}, {tile: target_paths[tile]}, 47)
        tile_model = u_net.transfer_learn_model(train_dataset, valid_dataset, tile_model, MODEL, tile)
        evaluate_model(val_input_img_paths_for_tile, val_target_img_paths_for_tile, tile_model, 'Transfer_learned')
        tile_model = None # Clears the model at the end of running

run_model_workflow()
