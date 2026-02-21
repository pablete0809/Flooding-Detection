import ee
import geemap

def debug_data():
    try:
        ee.Initialize()
    except:
        ee.Authenticate()
        ee.Initialize()

    # 1. Define ROI (Valencia) and Dates
    roi_coords = [
      [-0.5925608219299572, 39.05499447698859], 
      [-0.08444314614870718, 39.05499447698859], 
      [-0.08444314614870718, 39.48875100708455], 
      [-0.5925608219299572, 39.48875100708455], 
      [-0.5925608219299572, 39.05499447698859]
    ]
    roi = ee.Geometry.Polygon(roi_coords)
    start_date = '2024-10-25'
    end_date   = '2024-11-05'
    
    print(f"Checking data for period: {start_date} to {end_date}")

    # 2. Check Collections
    s2 = ee.ImageCollection('COPERNICUS/S2_SR') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 60))
        
    s1 = ee.ImageCollection('COPERNICUS/S1_GRD') \
        .filterBounds(roi) \
        .filterDate(start_date, end_date) \
        .filter(ee.Filter.eq('instrumentMode', 'IW')) \
        .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV'))
        
    print(f"S2 Images found: {s2.size().getInfo()}")
    print(f"S1 Images found: {s1.size().getInfo()}")
    
    if s2.size().getInfo() == 0:
        print("CRITICAL: No S2 images found. Check dates or cloud threshold.")
        return

    # 3. Check Intersection (Fusion Logic)
    # Replicate gee_pipeline logic roughly
    days = ee.List.sequence(
        ee.Date(start_date).millis(),
        ee.Date(end_date).millis(),
        24 * 60 * 60 * 1000
    )
    
    def check_day(day_millis):
        date = ee.Date(day_millis)
        s2_d = s2.filterDate(date, date.advance(1, 'day'))
        s1_d = s1.filterDate(date, date.advance(1, 'day'))
        return ee.Feature(None, {
            'date': date.format('YYYY-MM-dd'),
            's2_count': s2_d.size(),
            's1_count': s1_d.size()
        })
        
    checks = ee.FeatureCollection(days.map(check_day))
    infos = checks.getInfo()['features']
    
    print("\nDaily Availability:")
    for f in infos:
        p = f['properties']
        print(f"Date: {p['date']} | S2: {p['s2_count']} | S1: {p['s1_count']}")
        
    # 4. Probe Values for a valid day
    # Find a day with both
    valid_day = None
    for f in infos:
        if f['properties']['s2_count'] > 0 and f['properties']['s1_count'] > 0:
            valid_day = f['properties']['date']
            break
            
    if valid_day:
        print(f"\nProbing values for valid match on {valid_day}...")
        date_start = ee.Date(valid_day)
        date_end = date_start.advance(1, 'day')
        
        img_s2 = s2.filterDate(date_start, date_end).median().clip(roi)
        img_s1 = s1.filterDate(date_start, date_end).median().clip(roi)
        
        # S1 Value Check (Raw)
        # We need to see if it's Linear or dB
        # Reduce region
        s1_stats = img_s1.select(['VV', 'VH']).reduceRegion(
            reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), '', True),
            geometry=roi,
            scale=100, # coarse scale for speed
            bestEffort=True
        ).getInfo()
        
        print("S1 Raw Stats (before dB conversion):", s1_stats)
        
        # S2 Stats
        s2_stats = img_s2.select(['B4', 'B3', 'B2']).reduceRegion(
             reducer=ee.Reducer.mean(),
             geometry=roi,
             scale=100,
             bestEffort=True
        ).getInfo()
        print("S2 Raw Stats:", s2_stats)
        
    else:
        print("No coincident S1-S2 days found.")

if __name__ == "__main__":
    debug_data()
