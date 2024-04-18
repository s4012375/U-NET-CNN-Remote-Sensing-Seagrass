import model as u_net
import patchify
import matplotlib.pyplot as plt
import os
import numpy as np
from keras.utils import load_img

MODEL = 5
TILES = ['20191027', '20181010','20180624','20160609']
SIZE = {'Beadnell':(256,320,1), 'Budle': (256,192,1), 'Fenham': (576,512,1), 'Embleton': (192,320,1)}
SIZE_TCI = {'Beadnell':(256,320,3), 'Budle': (256,192,3), 'Fenham': (576,512,3), 'Embleton': (192,320,3)}
INPUT_IMG_DIR = ".\\TCI\\"
TARGET_IMG_DIR = ".\\ground-truth\\"
GT_SHAPE = {'Beadnell':(4,5), 'Budle': (4,3), 'Fenham': (9,8), 'Embleton': (3,5)}

def getPaths(region, tile):
    seperate_img_paths = sorted(
        [
            os.path.join("%s%s\\Patches"%(INPUT_IMG_DIR, tile), fname)
            for fname in os.listdir("%s%s\\Patches"%(INPUT_IMG_DIR, tile))
            if fname.endswith(".png") and fname.startswith(region)
        ]
    )
    seperate_target_paths = sorted(
        [
            os.path.join("%s%s\\Patches"%(TARGET_IMG_DIR, tile), fname)
            for fname in os.listdir("%s%s\\Patches"%(TARGET_IMG_DIR,tile))
            if fname.endswith(".png") and fname.startswith(region)
        ]
    )
    return (seperate_img_paths, seperate_target_paths)

def print_full_region_images():
    # Gets the model in question (base model) #1
    model = u_net.get_full_trained_resnet_model('full_resnet_checkpoint_model_%s_joint.keras'%(MODEL))
    for tile in TILES:
        # Gets the model in question (transfer learned) #2
        #model = u_net.get_full_trained_resnet_model('full_resnet_checkpoint_model_%s_%s.keras'%(MODEL, tile)) 
        
        for region, size in SIZE.items():
            tci_paths, target_paths = getPaths(region, tile) # Gets all patches
            
            # TO WRITE RESULT TO FILE
            val_preds = u_net.validate_model(tci_paths, target_paths, model) # Classifies the entire image
            predicted_mask = np.argmax(val_preds, axis=-1)
            predicted_mask = np.expand_dims(predicted_mask, axis=-1)
            results = []
            for x in range(0, GT_SHAPE[region][0]):
                row = []
                for y in range(x*GT_SHAPE[region][1], GT_SHAPE[region][1]*(x+1)):
                    row.append([predicted_mask[y]])
                results.append(row)
            results = np.array(results)
            reconstructed_result = patchify.unpatchify(results, size)
            plt.figure()
            plt.imshow(reconstructed_result)
            plt.savefig('.\\RESULTS\\model %d\\%s_%s_TL_2.png'%(MODEL, tile, region))
            plt.close()

            ## TO WRITE GROUNDTRUTH TO FILE
            GT = []
            for x in range(0, GT_SHAPE[region][0]):
                row = []
                for y in range(x*GT_SHAPE[region][1], GT_SHAPE[region][1]*(x+1)):
                    row.append([np.moveaxis(np.moveaxis(np.asarray(load_img(target_paths[y])), 2, 0)[0:1], 0,2)])
                GT.append(row)
            GT = np.array(GT)
            reconstructed_GT = patchify.unpatchify(GT, size)
            print('Reconstructed GT for %s'%region)
            plt.figure()
            plt.imshow(reconstructed_GT)
            plt.savefig('.\\RESULTS\\model %d\\%s_%s_GT.png'%(MODEL, tile, region))
            plt.close()
            
            ## TO WRITE TCI TO FILE
            TCI = []
            for x in range(0, GT_SHAPE[region][0]):
                row = []
                for y in range(x*GT_SHAPE[region][1], GT_SHAPE[region][1]*(x+1)):
                    row.append([np.asarray(load_img(tci_paths[y]))])
                TCI.append(row)
            TCI = np.array(TCI)
            reconstructed_TCI = patchify.unpatchify(TCI, SIZE_TCI[region])
            print('Reconstructed TCI for %s'%region)
            plt.figure()
            plt.imshow(reconstructed_TCI)
            plt.savefig('.\\RESULTS\\model %d\\%s_%s_TCI.png'%(MODEL, tile, region))
            plt.close()


def get_patch_result(patch_name, tile):
    model = u_net.get_full_trained_resnet_model('full_resnet_checkpoint_model_%s_joint.keras'%(MODEL)) # Gets the model in question (base model)
    val_preds = u_net.validate_model(['.\\TCI\\%s\\Patches\\%s.png'%(tile, patch_name)], ['.\\ground-truth\\%s\\Patches\\%s.png'%(tile, patch_name)], model) # Classifies the entire image
    predicted_mask = np.argmax(val_preds, axis=-1)
    predicted_mask = np.expand_dims(predicted_mask, axis=-1)
    plt.figure()
    plt.imshow(predicted_mask[0])
    plt.savefig('..\\model_%d_%s_%s_Patch_Result.png'%(MODEL, tile, patch_name))
    plt.close()
    
    plt.figure()
    plt.imshow(np.moveaxis(np.moveaxis(np.asarray(load_img('.\\ground-truth\\%s\\Patches\\%s.png'%(tile, patch_name))), 2, 0)[0:1], 0,2))
    plt.savefig('..\\model_%d_%s_%s_Patch_GT.png'%(MODEL, tile, patch_name))
    plt.close()


print_full_region_images()

# USE THIS TO PATCH AN INDIVIDUAL IMAGE
#get_patch_result('Fenham_image_2_7','20170717')
