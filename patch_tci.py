import rasterio
import numpy as np
import patchify
import tifffile as tiff

images = ['20191211','20191027','20190627','20190227', '20190123',
          '20181224','20181010','20180624','20171126','20171116',
          '20170717','20170525','20170125','20161226','20161129',
          '20161116','20160609','20160420','20160210','20151125']
roi = {
"type": "FeatureCollection",
"name": "RoI",
"features": [
{ "type": "Feature", "properties": { "id": 1 }, "geometry": { "type": "Polygon", "coordinates": [ [ [ 405017.315, 637433.868 ], [ 405017.315, 643311.712 ], [ 413283.449, 643311.712 ], [ 413552.31373723509023, 637433.868 ], [ 405017.315, 637433.868 ], [ 405017.315, 637433.868 ] ] ] } },
{ "type": "Feature", "properties": { "id": 2 }, "geometry": { "type": "Polygon", "coordinates": [ [ [ 413723.946, 636624.809 ], [ 416020.248270017327741, 636624.809 ], [ 415997.735504780139308, 631559.436617574538104 ], [ 413723.946, 631559.436617574538104 ], [ 413723.946, 636624.809 ] ] ] } },
{ "type": "Feature", "properties": { "id": 3 }, "geometry": { "type": "Polygon", "coordinates": [ [ [ 416080.814, 628656.478 ], [ 429459.468, 628656.478 ], [ 429459.468, 625920.158 ], [ 416080.814, 625920.158 ], [ 416080.814, 628656.478 ] ] ] } },
{ "type": "Feature", "properties": { "id": 4 }, "geometry": { "type": "Polygon", "coordinates": [ [ [ 418877.08, 624316.286 ], [ 428057.342, 624316.286 ], [ 428057.342, 622195.562249805196188 ], [ 418877.08, 622204.331 ], [ 418877.08, 624316.286 ] ] ] } }
]
}

for image_name in images[0:1]:
    with rasterio.open('.\\ground-truth\\' + image_name + '\\' + image_name + '.tif', 'r') as ds:
        areas = rasterio.mask.mask(ds, roi, crop=True)
        for area in areas:
            img_arr = area.read()  # read all raster values
            print(img_arr.shape)
            img_arr = np.moveaxis(img_arr, 0, 2)
            print(img_arr[2])
            patches = patchify.patchify(img_arr, (64,64,1), step=64)
            # Save each patch as a separate GeoTIFF file
            for x in range(patches.shape[0]):
                for y in range(patches.shape[1]):
                    for z in range(patches.shape[2]):
                        single_patch = patches[x, y, z, :, :, :]
                        tiff.imwrite('./ground-truth/' + image_name + '/Patches/' +  f'Image_{x}_{y}.tif', single_patch)
            reconstructed_image = unpatchify(patches, img_arr.shape)
            tiff.imwrite('./ground-truth/' + image_name + '/ORIGINAL/' + 'original.tif', reconstructed_image)
