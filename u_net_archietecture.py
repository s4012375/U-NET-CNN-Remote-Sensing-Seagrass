import keras
import numpy as np

from PIL import ImageOps
from keras.utils import load_img
from IPython.display import Image

import get_dataset as ds
import model as u_net
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

model_configs = {
    '1':[0.8, 
        ['20170717'],  # 80% training, 20% testing for base model
        ['20160609', '20180624', '20181010', '20191027']], # No-retraining and optimisation with individual 60% training for each tile and testing on 40%
    '2':[0.6, 
        ['20170717','20190627'], # 60% training, 40% testing for base model
        ['20160609', '20180624', '20181010']], # No-retraining and optimisation with individual 60% training for each tile and testing on 40%
    '3':[0.6, 
        ['20160609', '20170717', '20180624', '20181010','20190627'],# 60% training, 40% testing for base model
        ['20191211','20191027','20190627','20190227', '20190123',# No-retraining and optimisation with individual 60% training for each tile and testing on 40%
          '20181224','20181010','20180624','20171126','20171116',
          '20170717','20170525','20170125','20161226','20161129',
          '20161116','20160609','20160420','20160210','20151125']],
    '4':[0.6,
        ['20191211','20191027','20190123','20181224','20171126','20171116','20170125','20161226','20160609','20160420'],# 60% training, 40% testing for base model 
        ['20190627','20190227','20181010','20180624','20170717','20170525','20161129','20161116','20160210','20151125']]# No-retraining and optimisation with individual 60% training for each tile and testing on 40%
}


def display_mask(i):
    """Quick utility to display a model's prediction."""
    mask = np.argmax(val_preds[i], axis=-1)
    mask = np.expand_dims(mask, axis=-1)
    img = ImageOps.autocontrast(keras.utils.array_to_img(mask))
    img.show(title='Result %d'%i)

## GENRATES TRAINING/TESTING DATA AND GETS PATHS
training_paths, target_paths = ds.get_paths()
train_dataset, valid_dataset, val_input_img_paths, val_target_img_paths = ds.train_test_split(training_paths, target_paths)


## BUILD MODEL
model = u_net.get_untrained_model()

## RUNS THE MODEL
# Train the model, doing validation at the end of each epoch.
model = u_net.train_model(train_dataset, valid_dataset, model)

## VALIDATES FOR EVERYTHING
val_preds = u_net.validate_model(val_input_img_paths, val_target_img_paths, model)

from sklearn.metrics import precision_recall_fscore_support as score


average_f1=[]
average_precision=[]
average_recall=[]
average_accuracy=[]

for i in range(0, len(val_preds)):
    predicted = np.argmax(val_preds[i], axis=-1).flatten()
    actual = load_img(val_target_img_paths[i]).flatten()

    average_f1 += f1_score(actual, predicted, lables=[0,1,2,3,4,5,5,6,7,8], average=None)
    average_precision += precision_score(actual, predicted, lables=[0,1,2,3,4,5,5,6,7,8], average=None)
    average_recall += recall_score(actual, predicted, lables=[0,1,2,3,4,5,5,6,7,8], average=None)
    average_accuracy += accuracy_score(actual, predicted, lables=[0,1,2,3,4,5,5,6,7,8], average=None)
average_f1=average_f1 / len(val_preds)
average_precision=average_precision / len(val_preds)
average_recall=average_recall/ len(val_preds)
print('accuracy: {}'.format(average_accuracy))
print('precision: {}'.format(average_precision))
print('recall: {}'.format(average_recall))
print('fscore: {}'.format(average_fscore))


## DISPLAYS AND SAVES RESULTS
# Display results for 5 validation images 
for i in range(0,5):
    # Display input image
    img = load_img(val_input_img_paths[i])
    img.show(title='Input image %d'%i)
    

    # Display ground-truth target mask
    img = ImageOps.autocontrast(load_img(val_target_img_paths[i]))
    img.show(title='Ground-truth %d'%i)

    # Display mask predicted by our model
    display_mask(i)  # Note that the model only sees inputs at 150x150.
