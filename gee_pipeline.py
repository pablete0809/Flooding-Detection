import ee
import geemap
import os
import shutil
import numpy as np
import rasterio


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
        # Data appears to be already in dB based on diagnostic (-40 to +20 range)
        # So we do NOT convert again.
        vv = image.select('VV').rename('S1_VV')
        vh = image.select('VH').rename('S1_VH')
        
        # Ratio in dB = VV - VH
        ratio = vv.subtract(vh).rename('S1_VV_VH_ratio')
        
        return image.addBands([vv, vh, ratio], overwrite=True) \
                    .select(['S1_VV', 'S1_VH', 'S1_VV_VH_ratio'])

    s1_processed = s1.map(add_sar_features)
    return s1_processed

def get_terrain_data(roi):
    """
    Fetches and processes Terrain data: NASADEM, Slope, HAND, and Dynamic World.
    Returns a single image with bands: ['elevation', 'slope', 'hand', 'dw_water', 'dw_built']
    """
    # 1. NASADEM (Elevation)
    dem = ee.Image('NASA/NASADEM_HGT/001').select('elevation')
    
    # 2. Slope
    slope = ee.Terrain.slope(dem).unmask(0).rename('slope')
    
    # 3. HAND
    try:
        hand = ee.ImageCollection("users/gena/global-hand/hand-100").mosaic().unmask(0).rename('hand')
    except:
        hand = ee.Image(0).rename('hand')

    # 4. Dynamic World (Seasonality & Built)
    # Using 2023 Mode/Median.
    # We want 'water' probability for permanent water
    # We want 'built' probability to mask out roads
    dw_collection = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
        .filterDate('2023-01-01', '2024-01-01') \
        .filterBounds(roi)
        
    dw_image = dw_collection.select(['water', 'built']).median().unmask(0)
    dw_water = dw_image.select('water').rename('dw_water')
    dw_built = dw_image.select('built').rename('dw_built')
    
    # Clip to ROI and unmask any remaining
    terrain = ee.Image.cat([dem.unmask(0), slope, hand, dw_water, dw_built]).clip(roi)
    return terrain

def fuse_datasets(s2_collection, s1_collection, roi, start_date, end_date, terrain_img=None):
    """
    Creates daily composites fusing S2 and S1 data.
    Adds S2_MASK band (1=Valid, 0=Missing).
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
        
        # --- Handle S2 Validity ---
        # If we have an image, it's valid (1). If not, it's missing (0).
        # We also need to unmask the S2 data with 0 *after* creating the mask.
        
        s2_present = s2_daily.size().gt(0)
        
        # Create mask image: 1 if present, 0 if not.
        s2_mask = ee.Image.constant(1).rename('S2_MASK').updateMask(s2_present).unmask(0).clip(roi)
        
        # Get S2 data or empty (masked with 0)
        s2_img = ee.Image(s2_daily.median()).unmask(0)
        
        # Combine S2 + Mask
        s2_full = s2_img.addBands(s2_mask)
        
        # S1: Unmask with -50 (silence)
        s1_img = ee.Image(s1_daily.median()).unmask(-50)
        
        # Combine everything
        # We always return an image now, so the mask tells us if S2 is useful.
        # But for 'daily_fused' we might still want to filter completely empty days if S1 is also missing?
        # User wants robust dataset: even if S2 is missing, if S1 is there, it's useful.
        
        base_img = s2_full.addBands(s1_img) \
                .set('system:time_start', date.millis()) \
                .set('date', date.format('YYYY-MM-dd')) \
                .clip(roi)
        
        # Add Terrain if provided
        if terrain_img:
             return base_img.addBands(terrain_img)
        else:
             return base_img

    # Helper for empty image if needed, but returning null works better with filter
    def null_img(): return ee.Image()

    daily_fused = ee.ImageCollection.fromImages(
        days.map(process_day)
    )
    # Filter? If we want strictly days with *some* data.
    # For simplicity, let's keep all days requested to ensure time series continuity?
    # Or filter if both S1 and S2 are missing?
    # Let's keep it simple: Return all days in range.
    
    return daily_fused

def add_weak_labels(image, threshold=0.0):
    """
    Adds flood labels based on MNDWI threshold AND excludes Built areas.
    """
    # 1. Basic Flood: MNDWI > Threshold
    mndwi = image.select('S2_MNDWI')
    potential_flood = mndwi.gt(threshold)
    
    # 2. Exclude Built-up areas (Roads/Urban) using Dynamic World 'dw_built'
    # 'dw_built' is probability 0-1.
    # If built prob > 0.5, it's likely a road/building.
    if 'dw_built' in image.bandNames().getInfo():
        built_mask = image.select('dw_built').gt(0.4) # Strict on buildings
        # Flood = Potential AND NOT Built
        flood = potential_flood.And(built_mask.Not())
    else:
        flood = potential_flood
        
    # 3. Optional: Exclude Permanent Water if we only want NEW flood
    # But usually "Flood Label" includes all water.
    # User said: "Permanent water... not usually correct".
    # Let's stick to improved water detection for now.
    
    return image.addBands(flood.rename('LABEL_flood_raw'))

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

def process_and_split(src_path, filename, s1_dir, s2_dir, terrain_dir, label_dir):
    """
    Helper to split bands of a downloaded tile.
    Updated for new bands (S2_MASK, dw_built).
    """
    try:
        with rasterio.open(src_path) as src:
            data = src.read()
            profile = src.profile.copy()
            
            # Updated Band Order:
            # S2: 7 bands (B2, B3, B4, B8, B11, NDWI, MNDWI) + 1 Mask = 8 bands
            # S1: 3 bands (VV, VH, Ratio)
            # Terrain: 5 bands (DEM, Slope, HAND, DW_Water, DW_Built)
            # Label: 1 band
            
            s2_count = 8
            s1_count = 3
            terrain_count = 5
            label_count = 1
            
            current_idx = 0
            
            # --- Save S2 ---
            s2_data = data[current_idx : current_idx + s2_count, :, :] 
            profile.update(count=s2_count)
            with rasterio.open(os.path.join(s2_dir, filename), 'w', **profile) as dst:
                dst.write(s2_data)
            current_idx += s2_count
                
            # --- Save S1 ---
            s1_data = data[current_idx : current_idx + s1_count, :, :]
            profile.update(count=s1_count)
            with rasterio.open(os.path.join(s1_dir, filename), 'w', **profile) as dst:
                dst.write(s1_data)
            current_idx += s1_count
            
            # --- Save Terrain ---
            terrain_data = data[current_idx : current_idx + terrain_count, :, :]
            profile.update(count=terrain_count)
            with rasterio.open(os.path.join(terrain_dir, filename), 'w', **profile) as dst:
                dst.write(terrain_data)
            current_idx += terrain_count
            
            # --- Save Label ---
            label_data = data[current_idx : current_idx + label_count, :, :]
            profile.update(count=label_count, dtype=rasterio.uint8, nodata=None)
            with rasterio.open(os.path.join(label_dir, filename), 'w', **profile) as dst:
                dst.write(label_data.astype(rasterio.uint8))
                
    except Exception as e:
        print(f"Failed to split bands for {filename}: {e}")

def download_patches(collection, roi, output_dir, scale=10, rows=4, cols=4, overwrite=False):
    """
    Downloads the entire ImageCollection as patches (tiles) and splits them into S1, S2, Terrain, and Label folders.
    Files are named: tile_{tile_id}_{date}.tif
    
    Args:
        collection (ee.ImageCollection): The collection of daily fused images.
        roi (ee.Geometry): The region of interest.
        output_dir (str): Root directory for the dataset.
        scale (int): Scale in meters.
        rows (int): Number of rows in the grid.
        cols (int): Number of columns in the grid.
        overwrite (bool): If True, redownloads all tiles.
    """
    # 1. Create directory structure
    s1_dir = os.path.join(output_dir, 'S1')
    s2_dir = os.path.join(output_dir, 'S2')
    terrain_dir = os.path.join(output_dir, 'Terrain')
    label_dir = os.path.join(output_dir, 'labels')
    temp_dir = os.path.join(output_dir, 'temp_tiles')
    
    for d in [s1_dir, s2_dir, terrain_dir, label_dir, temp_dir]:
        if not os.path.exists(d):
            os.makedirs(d)
            
    print(f"Preparing to download time series to {output_dir}...")
    
    # 2. Get list of images (dates)
    # Convert collection to list to iterate client-side
    try:
        size = collection.size().getInfo()
        if size == 0:
            print("Collection is empty!")
            return
        image_list = collection.toList(size)
    except Exception as e:
        print(f"Error getting collection info: {e}")
        return

    print(f"Found {size} images (dates) in collection.")

    # 3. Define Grid
    grid = geemap.fishnet(roi, rows=rows, cols=cols)
    grid_count = grid.size().getInfo()
    grid_list = grid.toList(grid_count)
    print(f"Grid Size: {grid_count} tiles per image.")
    
    # 4. Iterate over Images (Dates)
    for i in range(size):
        img = ee.Image(image_list.get(i))
        
        # Get Date
        try:
            date_str = img.get('date').getInfo() # We set this earlier: YYYY-MM-DD
        except:
            date_str = f"date_{i}"
            
        print(f"Processing date: {date_str} ({i+1}/{size})")
        
        # 5. Iterate over Tiles for this Image
        for j in range(grid_count):
            feature = ee.Feature(grid_list.get(j))
            region = feature.geometry()
            
            # Filename: tile_{tile_id}_{date}.tif
            filename = f"tile_{j}_{date_str}.tif"
            temp_path = os.path.join(temp_dir, filename)
            
            # Check existence (Resume mode)
            # We check if the final split files exist to skip download
            final_exists = os.path.exists(os.path.join(s1_dir, filename))
            if not overwrite and final_exists:
                # print(f"  Tile {j} exists. Skipping.")
                continue
            
            # Download to temp
            try:
                geemap.download_ee_image(
                    img,
                    temp_path,
                    region=region,
                    scale=scale,
                    crs='EPSG:4326',
                    overwrite=True # Always overwrite temp
                )
            except Exception as e:
                print(f"  ERROR downloading tile {j} for {date_str}: {e}")
                continue
            
            # Post-process (Split Bands) immediately to save space? 
            # Or do batch? Doing immediately is safer for storage.
            if os.path.exists(temp_path):
                process_and_split(temp_path, filename, s1_dir, s2_dir, terrain_dir, label_dir)
                # Remove temp file to save space
                try:
                    os.remove(temp_path)
                except:
                    pass

    # shutil.rmtree(temp_dir) # Optional cleanup
    print("Time series download complete.")

