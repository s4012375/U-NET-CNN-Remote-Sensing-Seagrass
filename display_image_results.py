import model as u_net
import patchify
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.utils import load_img

MODEL = 5
TILES = ['20170717', '20190627','20160210','20170525','20181010','20180624','20191027','20160609','20161129','20190227']
SIZE = {'Beadnell':(256,320,1), 'Budle': (256,192,1), 'Fenham': (576,512,1), 'Embleton': (192,320,1)}
input_dir = ".\\TCI\\"
target_dir = ".\\ground-truth\\"
shape = {'Beadnell':(4,5), 'Budle': (4,3), 'Fenham': (9,8), 'Embleton': (3,5)}

def getPaths(region, tile):
    seperate_img_paths = sorted(
        [
            os.path.join("%s%s\\Patches"%(input_dir, tile), fname)
            for fname in os.listdir("%s%s\\Patches"%(input_dir, tile))
            if fname.endswith(".png") and fname.startswith(region)
        ]
    )
    seperate_target_paths = sorted(
        [
            os.path.join("%s%s\\Patches"%(target_dir, tile), fname)
            for fname in os.listdir("%s%s\\Patches"%(target_dir,tile))
            if fname.endswith(".png") and fname.startswith(region)
        ]
    )
    return (seperate_img_paths, seperate_target_paths)

def print_full_region_images():
    # model = u_net.get_full_trained_resnet_model('full_resnet_checkpoint_model_%s.keras'%(MODEL)) # Gets the model in question (base model)
    for tile in TILES:
        model = u_net.get_full_trained_resnet_model('full_resnet_checkpoint_model_%s_%s.keras'%(MODEL, tile)) # Gets the model in question (transfer learned)
        for region, size in SIZE.items():
            tci_paths, target_paths = getPaths(region, tile) # Gets all tiles
            val_preds = u_net.validate_model(tci_paths, target_paths, model) # Classifies the entire image
            predicted_mask = np.argmax(val_preds, axis=-1)
            predicted_mask = np.expand_dims(predicted_mask, axis=-1)
            results = []
            for x in range(0, shape[region][0]):
                row = []
                for y in range(x*shape[region][1], shape[region][1]*(x+1)):
                    row.append([predicted_mask[y]])
                results.append(row)
            results = np.array(results)
            reconstructed_result = patchify.unpatchify(results, size)
            plt.figure()
            plt.imshow(reconstructed_result)
            plt.savefig('.\\RESULTS\\model %d\\%s_%s_TL.png'%(MODEL, tile, region))
            plt.close()

            ## TO GET OR SHOW GROUNDTRUTH
##            GT = []
##            for x in range(0, shape[region][0]):
##                row = []
##                for y in range(x*shape[region][1], shape[region][1]*(x+1)):
##                    row.append([np.moveaxis(np.moveaxis(np.asarray(load_img(target_paths[y])), 2, 0)[0:1], 0,2)])
##                GT.append(row)
##            GT = np.array(GT)
##            reconstructed_GT = patchify.unpatchify(GT, size)
##            print('Reconstructed GT for %s'%region)
##            plt.figure()
##            plt.imshow(reconstructed_GT)
##            plt.savefig('.\\RESULTS\\model %d\\%s_%s_GT.png'%(MODEL, tile, region))
##            plt.show()  # display it

print_full_region_images()
