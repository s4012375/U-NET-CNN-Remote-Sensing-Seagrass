import rasterio
from rasterio import mask 
import numpy as np
import patchify
import tifffile as tiff
import fiona
from PIL import Image

IMAGES = ['20191211','20191027','20190627','20190227', '20190123',
          '20181224','20181010','20180624','20171126','20171116',
          '20170717','20170525','20170125','20161226','20161129',
          '20161116','20160609','20160420','20160210','20151125']
GT_SIZE = [(576,768,1),(448,192,1),(256,1280,1),(192,896,1)]
AREA_NAME = ['Fenham', 'Budle', 'Beadnell', 'Embleton']

# Loops through each image that needs patchifying
for image_name in IMAGES[0:20]:
    area_i = 0
    # Reads the raster file for the ground-truth
    with rasterio.open('.\\ground-truth\\' + image_name + '\\' + image_name + '_GT.tif', 'r', nodata=1.0) as ds:
        # Reads all Region of Interest geometries in the file
        with fiona.open(".\\datasets\\shapefiles\\RoI\\RoI.shp", "r") as shapefile:
            # Loops through each polygon region
            geoms = [feature["geometry"] for feature in shapefile]
            for geom in geoms:
                # Extracts the RoI from the image
                area = rasterio.mask.mask(ds, [geom], crop=True, nodata=1.0)
                img_arr = np.moveaxis(area[0], 0, 2)
                img_arr -= 1 # starts classes at 0
                patches = patchify.patchify(img_arr, (64,64,1), step=64)

                # Save the patches
                for x in range(patches.shape[0]):
                    for y in range(patches.shape[1]):
                       for z in range(patches.shape[2]):
                           single_patch = patches[x, y, z, :, :, :]
                           # Save each patch as a separate GeoTIFF file
                           tiff.imwrite('./ground-truth/' + image_name + '/Patches/' + AREA_NAME[area_i] + f'_image_{x}_{y}.tif', single_patch)
                           
                           img = Image.open('./ground-truth/' + image_name + '/Patches/' + AREA_NAME[area_i] + f'_image_{x}_{y}.tif')
                           img = img.convert('RGB')
                           img.save('./ground-truth/' + image_name + '/Patches/' + AREA_NAME[area_i] + f'_image_{x}_{y}.png', 'PNG')

                print('Written ground-truth patches for ' + AREA_NAME[area_i])
                area_i = area_i + 1

TCI_SIZE = [(576,768,3),(448,192,3),(256,1280,3),(192,896,3)]
# Loops through each image that needs patchifying
for image_name in IMAGES[0:20]:
    area_i = 0
    # Reads the raster file for the ground-truth
    with rasterio.open('.\\TCI\\' + image_name + '\\' + image_name + '.tif', 'r', nodata = 0.0) as ds:
        # Reads all Region of Interest geometries in the file          
        with fiona.open(".\\datasets\\shapefiles\\RoI\\RoI.shp", "r") as shapefile:
            # Loops through each polygon region
            geoms = [feature["geometry"] for feature in shapefile]
            for geom in geoms:
                # Extracts the RoI from the image
                area = rasterio.mask.mask(ds, [geom], crop=True, nodata=0.0)
                img_arr = np.moveaxis(area[0], 0, 2)
                img_arr = img_arr * (255 / np.max(img_arr))
                patches = patchify.patchify(img_arr, (64,64,3), step=64)
                
                ## Save each patch as a separate GeoTIFF file
                for x in range(patches.shape[0]):
                    for y in range(patches.shape[1]):
                        for z in range(patches.shape[2]):
                            single_patch = patches[x, y, z, :, :, :]
                            tiff.imwrite('./TCI/' + image_name + '/Patches/' + AREA_NAME[area_i] + f'_image_{x}_{y}.tif', single_patch)
                            
                            im=Image.fromarray(single_patch.astype(np.uint8))
                            im.save('./TCI/%s/Patches/%s_image_%d_%d.png'%(image_name,AREA_NAME[area_i],x,y))
                            
                print('Written TCI patches for ' + AREA_NAME[area_i])
                area_i = area_i + 1
                

