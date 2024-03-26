import rasterio
from rasterio import mask 
import numpy as np
import patchify
import tifffile as tiff
import fiona

images = ['20191211','20191027','20190627','20190227', '20190123',
          '20181224','20181010','20180624','20171126','20171116',
          '20170717','20170525','20170125','20161226','20161129',
          '20161116','20160609','20160420','20160210','20151125']
#size_gnd_tr = [(6144,8576,1),(5248,2368,1),(2816,13952,1),(2176,9600,1)]
size_gnd_tr = [(576,768,1),(448,192,1),(256,1280,1),(192,896,1)]
area_name = ['Fenham', 'Budle', 'Beadnell', 'Embleton']

# Loops through each image that needs patchifying
for image_name in images[0:20]:
    area_i = 0
    # Reads the raster file for the ground-truth
    with rasterio.open('.\\ground-truth\\' + image_name + '\\' + image_name + '.tif', 'r') as ds:
        # Reads all Region of Interest geometries in the file
        with fiona.open(".\\datasets\\shapefiles\\RoI\\RoI.shp", "r") as shapefile:
            # Loops through each polygon region
            geoms = [feature["geometry"] for feature in shapefile]
            for geom in geoms:
                # Extracts the RoI from the image
                area = rasterio.mask.mask(ds, [geom], crop=True)
                img_arr = np.moveaxis(area[0], 0, 2)
                patches = patchify.patchify(img_arr, (64,64,1), step=64)
                # Save each patch as a separate GeoTIFF file
                for x in range(patches.shape[0]):
                    for y in range(patches.shape[1]):
                        for z in range(patches.shape[2]):
                            single_patch = patches[x, y, z, :, :, :]
                            tiff.imwrite('./ground-truth/' + image_name + '/Patches/' + area_name[area_i] + f'_image_{x}_{y}.tif', single_patch)
                print('Writen patches for ' + area_name[area_i])
                area_i = area_i + 1

shape_tci = [(576,768,1),(448,192,1),(256,1280,1),(192,896,1)]
# Loops through each image that needs patchifying
for image_name in images[0:20]:
    area_i = 0
    # Reads the raster file for the ground-truth
    with rasterio.open('.\\TCI\\' + image_name + '\\' + image_name + '.tif', 'r') as ds:
        # Reads all Region of Interest geometries in the file          
        with fiona.open(".\\datasets\\shapefiles\\RoI\\RoI.shp", "r") as shapefile:
            # Loops through each polygon region
            geoms = [feature["geometry"] for feature in shapefile]
            for geom in geoms:
                # Extracts the RoI from the image
                area = rasterio.mask.mask(ds, [geom], crop=True)
                img_arr = np.moveaxis(area[0], 0, 2)
                patches = patchify.patchify(img_arr, (64,64,1), step=64)
                # Save each patch as a separate GeoTIFF file
                for x in range(patches.shape[0]):
                    for y in range(patches.shape[1]):
                        for z in range(patches.shape[2]):
                            single_patch = patches[x, y, z, :, :, :]
                            tiff.imwrite('./TCI/' + image_name + '/Patches/' + area_name[area_i] + f'_image_{x}_{y}.tif', single_patch)
                print('Writen patches for ' + area_name[area_i])
                area_i = area_i + 1

## To Reconstruct
                #reconstructed_image = patchify.unpatchify(patches, shape_tci[area_i])
                #tiff.imwrite('./TCI/' + image_name + '/ORIGINAL/'+ area_name[area_i] + '_original.tif', reconstructed_image)
                #print('Patched and reconstructed ' + area_name[area_i])
                #area_i = area_i + 1
                

