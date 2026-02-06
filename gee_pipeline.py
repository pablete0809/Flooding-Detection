import ee
import geemap
import os
import shutil
import numpy as np
import rasterio

# Initialize Earth Engine
try:
    ee.Initialize()
except Exception as e:
    print("Authenticating Earth Engine...")
    ee.Authenticate()
    ee.Initialize()

def get_sentinel2_data(roi, start_date, end_date, cloud_threshold=60):
    """
    Fetches and processes Sentinel-2 data.
    """
    s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', cloud_threshold))

    def mask_s2_clouds(image):
        scl = image.select('SCL')
        # Keep clear (4), water (6)
        # 3: Cloud shadows, 8: Cloud medium probability, 9: Cloud high probability, 10: Thin cirrus
        cloud_mask = scl.neq(3).And(scl.neq(8)).And(scl.neq(9)).And(scl.neq(10))
        return image.updateMask(cloud_mask)

    def add_mndwi(image):
        # MNDWI = (Green - SWIR) / (Green + SWIR) = (B3 - B11) / (B3 + B11)
        mndwi = image.normalizedDifference(['B3', 'B11']).rename('S2_MNDWI')
        # Keep NDWI too just in case
        ndwi = image.normalizedDifference(['B3', 'B8']).rename('S2_NDWI')
        return image.addBands([mndwi, ndwi])

    s2_processed = s2.map(mask_s2_clouds) \
                     .map(add_mndwi) \
                     .select(['B2', 'B3', 'B4', 'B8', 'B11', 'S2_NDWI', 'S2_MNDWI'], 
                             ['S2_B2', 'S2_B3', 'S2_B4', 'S2_B8', 'S2_B11', 'S2_NDWI', 'S2_MNDWI'])

    return s2_processed

def get_sentinel1_data(roi, start_date, end_date):
    """
    Fetches and processes Sentinel-1 data.
    """
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VH')) \
        .select(['VV', 'VH'])

    def add_sar_features(image):
        vv = image.select('VV')
        vh = image.select('VH')
        ratio = vv.subtract(vh).rename('S1_VV_VH_ratio') # in dB, subtraction is division
        return image.addBands([vv.rename('S1_VV'), vh.rename('S1_VH'), ratio]) \
                    .select(['S1_VV', 'S1_VH', 'S1_VV_VH_ratio'])

    s1_processed = s1.map(add_sar_features)
    return s1_processed

def fuse_datasets(s2_collection, s1_collection, roi, start_date, end_date):
    """
    Creates daily composites fusing S2 and S1 data.
    """
    days = ee.List.sequence(
        ee.Date(start_date).millis(),
        ee.Date(end_date).millis(),
        24 * 60 * 60 * 1000
    )

    def process_day(day_millis):
        date = ee.Date(day_millis)
        
        # S2 Daily limits
        s2_daily = s2_collection.filterDate(date, date.advance(1, 'day'))
        
        # S1 Daily limits
        s1_daily = s1_collection.filterDate(date, date.advance(1, 'day'))
        
        # Check if we have both
        s2_img = ee.Algorithms.If(s2_daily.size().gt(0), s2_daily.median(), null_img())
        s1_img = ee.Algorithms.If(s1_daily.size().gt(0), s1_daily.median(), null_img()) # Use median or specific orbit logic
        
        # For this pipeline, we might want strict intersection (days with BOTH)
        # Or filling. User asked for Dataset creation. 
        # Prudent approach: Return merged image if both exist.
        
        return ee.Algorithms.If(
            s2_daily.size().eq(0).max(s1_daily.size().eq(0)),
            ee.Image(), # Return empty image for filtering later (prevent null in list)
            ee.Image(s2_daily.median()).addBands(ee.Image(s1_daily.median()))
                .set('system:time_start', date.millis())
                .set('date', date.format('YYYY-MM-dd'))
                .clip(roi)
        )

    # Helper for empty image if needed, but returning null works better with filter
    def null_img(): return ee.Image()

    daily_fused = ee.ImageCollection.fromImages(
        days.map(process_day)
    ).filter(ee.Filter.notNull(['system:time_start']))
    
    return daily_fused

def add_weak_labels(image, threshold=0.0):
    """
    Adds flood labels based on MNDWI threshold.
    MNDWI > 0.0 is often used for water, but can be tuned (e.g. -0.2 to 0.2).
    """
    # Use MNDWI (Index 0 is water/land separation usually around 0)
    flood = image.select('S2_MNDWI').gt(threshold).rename('LABEL_flood_raw')
    return image.addBands(flood)

def download_tile(image, region, filename, scale=10):
    """
    Downloads a single image to local disk.
    """
    print(f"Downloading {filename}...")
    geemap.ee_export_image(
        image, 
        filename=filename, 
        scale=scale, 
        region=region, 
        file_per_band=False
    )
    print("Done.")

def download_patches(image, roi, output_dir, scale=10, overwrite=False):
    """
    Downloads the image as patches (tiles) and splits them into S1, S2, and Label folders.
    Aligns with SEN12FLOOD dataset structure.
    
    Args:
        overwrite (bool): If True, redownloads all tiles. If False, skips existing tiles (Resume mode).
    """

    # 1. Create directory structure
    s1_dir = os.path.join(output_dir, 'S1')
    s2_dir = os.path.join(output_dir, 'S2')
    label_dir = os.path.join(output_dir, 'labels')
    temp_dir = os.path.join(output_dir, 'temp_tiles')
    
    for d in [s1_dir, s2_dir, label_dir, temp_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            
    print(f"Downloading tiles to {temp_dir}...")
    
    # 2. Download tiles using geemap
    # Note: geemap.download_ee_image_tiles usually expects a fusion table or specific geometry logic
    # But we can also use export logic. For simplicity in local dev, we might use
    # geemap.fishnet to create a grid, then iterate download.
    # However, 'geemap.download_ee_image' with clipping can work if the image is huge.
    # Let's use the straightforward export which handles auto-tiling if dimensions > maxPixels?
    # No, we want explicit small patches.
    
    # Let's use fishnet to get the grid
    # Let's use fishnet to get the grid
    grid = geemap.fishnet(roi, rows=4, cols=4)
    
    # helper to process feature collection to list of geometries
    count = grid.size().getInfo()
    print(f"Grid Size: {count} tiles")
    
    # We need to iterate client-side. Convert grid to list.
    grid_list = grid.toList(count)
    
    for i in range(count):
        feature = ee.Feature(grid_list.get(i))
        region = feature.geometry()
        
        # Define filename
        # Use simple index or coordinates naming
        fname = os.path.join(temp_dir, f"tile_{i}.tif")
        
        if not overwrite and os.path.exists(fname):
            print(f"Tile {i} exists. Skipping (Resume mode).")
            continue
        
        print(f"Downloading tile {i+1}/{count}...")
        
        try:
            # Use geemap.download_ee_image for single tile
            # This uses geedim under the hood but we wrap it to catch errors
            geemap.download_ee_image(
                image,
                fname,
                region=region,
                scale=scale,
                crs='EPSG:4326',
                overwrite=overwrite
            )
        except Exception as e:
            print(f"ERROR downloading tile {i}: {e}")
            print("Skipping this tile.")
            continue
    
    print("Processing tiles into separate source folders...")
    
    # 3. Post-process: Split bands
    # Expected Band Order from pipeline: 
    # ['S2_B2', 'S2_B3', 'S2_B4', 'S2_B8', 'S2_B11', 'S2_NDWI', 'S2_MNDWI', 'S1_VV', 'S1_VH', 'S1_VV_VH_ratio', 'LABEL_flood_raw']
    # Indices (0-based):
    # S2: 0, 1, 2, 3, 4, 5, 6
    # S1: 7, 8, 9
    # Label: 10
    
    tiles = [f for f in os.listdir(temp_dir) if f.endswith('.tif')]
    
    for filename in tiles:
        src_path = os.path.join(temp_dir, filename)
        
        with rasterio.open(src_path) as src:
            # Read all data
            data = src.read()
            profile = src.profile.copy()
            
            # --- Save S2 ---
            # S2 bands are 0 to 6 (7 bands)
            s2_data = data[0:7, :, :] 
            profile.update(count=7)
            with rasterio.open(os.path.join(s2_dir, filename), 'w', **profile) as dst:
                dst.write(s2_data)
                
            # --- Save S1 ---
            # S1 bands are 7 to 9 (3 bands)
            s1_data = data[7:10, :, :]
            profile.update(count=3)
            with rasterio.open(os.path.join(s1_dir, filename), 'w', **profile) as dst:
                dst.write(s1_data)
                
            # --- Save Label ---
            # Label band is 10 (1 band)
            label_data = data[10:11, :, :]
            profile.update(count=1, dtype=rasterio.uint8, nodata=None) # Labels usually uint8, remove float nodata
            with rasterio.open(os.path.join(label_dir, filename), 'w', **profile) as dst:
                dst.write(label_data.astype(rasterio.uint8))
                
    # Cleanup
    # shutil.rmtree(temp_dir) # Optional: remove temp dir
    print(f"Dataset generated in {output_dir}")

