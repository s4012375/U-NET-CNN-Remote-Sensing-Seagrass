# U-Net CNN Remote Sensing Seagrass
A repository for the code and images Dissertation at the University of Gloucestershire involving the mapping of seagrass meadows over several years using U-Net CNN. The aim of this dissertation is to investigate the impact of transfer learning on optimising a general model for a specific tile.

## Workflow
The following workflow processes imagery such that it generates a model trained on one or more tiles which acts as a base model. That model can be directly reused on other tile, as in the control scenario, or optimised using a tile specific training set. The aim is to investigate whether a tile optimised model performs better than a general model and the extent that the type of tiles the base model is trained on impacts the results.

## Contents
- `preprocess_acolite.py` Preprocessing imagery with Acolite (this is the only script which requires a conda environment for some libraries)
- `patch_imagery.py` Cuts TCI and ground-truth into png file patches
- `u-net_workflow.py` The workflow to run a model from training the base model, validating a control group, and applying transfer learning on the same tiles as the control group evaluating the results
- `model.py` The u-net architecture used
- `get-dataset.py` Reads the files used for training and testing
- `model_[x].keras` A u-net base model trained with model [x] configuration
- `model_[x]_[tile]` A u-net transfer learned model optimised for [tile] originating from base model [x]
- `ground-truth/` Folder which contains the ground-truth tif files and the Patches folder by tile
- `TCI/` Folder which contains the TCI input tif files and the Patches folder by tile
- `RESULTS` Folder containing the classification/semantic segmentation results by Model, Training/Control/Transfer Learning, Tile
- `Outpuf_Tiff` Folder which contains the pre-processed outputted tif bands from acolite - Move to `TCI` after combining to get TCI
- `datasets/` Some of the ground-truth files before and after combining, and a region of interest (RoI) for square polygons to clip the TCI and ground-truth against during `patch_imagery.py`, and the imgery to pre-process in acolite
- `display_imge_results.py` Saves regions, results and individual patches to file for a given model

## Usage
- Gather the satellite imagery to run through acolite
- Install a conda environment with the requirements as specified in `acolite main`
- Run `preprocess_acolite.py` to pre-process the imagery (apply custom limits and file paths as required)
- Use other software, e.g. QGIS, SNAP, to create TCI from bands 2,3,4
- Use other software to make ground-truth raster (with classes labelled 1 to `x`) and create own RoI shapefile as required
- Place ground-truth, TCIs, and RoI shapefile in folders
- Run `patch_imagery`
- Run `u-net_workflow` (set model_configs to custom images and testing image numbers if neccessary and MODEL to which ever model configuration you wish to run)
- To recreate an TCI region, groundtruth region, result region or find an individual patch use `display_imge_results.py` with your own configurations (what you want to save)

## NOTE
All shapefiles, rasters and other geospatial files are using EPSG: 27700. If you use another projection, ensure that all geospatial files are consistent.
