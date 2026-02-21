# Manual Updates for `main.ipynb`

Please copy and paste the following code blocks into your `main.ipynb` notebook to update the functionality.

---

### **Cell 8: Generate Dataset (Updated)**
*Features: Cleans old data, Custom Grid (Rows/Cols), Custom MNDWI Threshold*

```python
import shutil

# 1. Clean up old dataset if it exists (Optional but recommended)
if os.path.exists(dataset_dir):
    print(f"Removing old dataset at {dataset_dir}...")
    shutil.rmtree(dataset_dir)

# 2. Generate new dataset
# You can customize rows and cols here.
# For your smaller ROI, try rows=2, cols=2 to get reasonable patch sizes.
# Threshold 0.0 is default for MNDWI.
download_patches(labeled_dataset, roi, dataset_dir, scale=10, rows=2, cols=2)

print("Dataset generation complete.")
```

---

### **Cell 9: Validate Dataset**

```python
import os

# Check if data was generated
if not os.path.exists(dataset_dir):
    print("Error: Dataset directory not found!")
else:
    s1_files = os.listdir(os.path.join(dataset_dir, 'S1'))
    s2_files = os.listdir(os.path.join(dataset_dir, 'S2'))
    lbl_files = os.listdir(os.path.join(dataset_dir, 'labels'))
    terrain_files = os.listdir(os.path.join(dataset_dir, 'Terrain'))

    print(f"Generated {len(s1_files)} S1 patches")
    print(f"Generated {len(s2_files)} S2 patches")
    print(f"Generated {len(lbl_files)} Label patches")
    print(f"Generated {len(terrain_files)} Terrain patches")
    
    # Check consistency
    if len(s1_files) == len(s2_files) == len(lbl_files) == len(terrain_files):
        print("Dataset is consistent.")
    else:
        print("Warning: Counts do not match!")
```

---

### **Cell 10: Visualize Data (Corrected for Dynamic World)**

```python
import matplotlib.pyplot as plt
import rasterio
import numpy as np
import os
import random

def normalize(band):
    band = band.astype(float)
    return (band - band.min()) / (band.max() - band.min() + 1e-6)

dataset_path = dataset_dir  # Ensure this points to your dataset folder
samples = os.listdir(os.path.join(dataset_path, 'S2'))

# Pick a random sample
filename = random.choice(samples)

def visualize_sample(dataset_path, filename):
    # Paths
    p_s2 = os.path.join(dataset_path, 'S2', filename)
    p_s1 = os.path.join(dataset_path, 'S1', filename)
    p_lbl = os.path.join(dataset_path, 'labels', filename)
    p_terrain = os.path.join(dataset_path, 'Terrain', filename)
    
    # Load S2
    with rasterio.open(p_s2) as src:
        s2 = src.read() # B2, B3, B4(Red)...
        # RGB = B4, B3, B2 => Indices 2, 1, 0
    
    # Load S1
    with rasterio.open(p_s1) as src:
        s1 = src.read() # VV, VH, Ratio
        
    # Load Label
    with rasterio.open(p_lbl) as src:
        label = src.read(1)
        
    # Load Terrain (Band 4 is Water Seasonality)
    with rasterio.open(p_terrain) as src:
        val = src.read()
        if val.shape[0] >= 4:
            water_prob = val[3, :, :] # Band 4 (Index 3)
        else:
            water_prob = np.zeros_like(label)

    # Plot
    fig, ax = plt.subplots(1, 4, figsize=(24, 6))
    
    # S2 RGB
    rgb = np.stack([s2[2], s2[1], s2[0]], axis=-1)
    ax[0].imshow(normalize(rgb))
    ax[0].set_title(f"Sentinel-2 RGB\n{filename}")
    ax[0].axis('off')
    
    # S1 False Color (VV, VH, Ratio)
    # Using simple normalization
    s1_rgb = np.stack([s1[0], s1[1], s1[0]/(s1[1]+0.1)], axis=-1)
    ax[1].imshow(normalize(s1_rgb))
    ax[1].set_title("Sentinel-1 (VV, VH, Ratio)")
    ax[1].axis('off')
    
    # Target Label
    ax[2].imshow(label, cmap='Blues', interpolation='nearest')
    ax[2].set_title("Target Label (MNDWI)")
    ax[2].axis('off')
    
    # Dynamic World Water
    # IMPORTANT: vmin=0, vmax=1 to display probability correctly
    ax[3].imshow(water_prob, cmap='Blues', vmin=0, vmax=1)
    ax[3].set_title("Dynamic World Water Prob.\n(Seasonality 2023)")
    ax[3].axis('off')
    
    plt.tight_layout()
    plt.show()

print(f"Visualizing: {filename}")
visualize_sample(dataset_path, filename)
```

---

### **Cell 11: MNDWI Histogram (Optional)**

```python
import matplotlib.pyplot as plt
import numpy as np

# Flatten all MNDWI pixels from a few samples to see distribution
mndwi_values = []

# Check if we have S2 images
s2_files = os.listdir(os.path.join(dataset_dir, 'S2'))
sample_files = s2_files[:50] # Check first 50 images

print("Calculating MNDWI histogram...")

for f in sample_files:
    path = os.path.join(dataset_dir, 'S2', f)
    with rasterio.open(path) as src:
        # S2 bands: B2(Blue), B3(Green), B4(Red), B8(NIR), B11(SWIR1), B12(SWIR2), MNDWI
        # If you exported 7 bands, MNDWI is likely the last one (index 6)
        if src.count >= 7:
            mndwi = src.read(7) # Read 7th band
            mndwi_values.extend(mndwi.flatten())

mndwi_values = np.array(mndwi_values)
print(f"Pixels analyzed: {len(mndwi_values)}")

if len(mndwi_values) > 0:
    plt.figure(figsize=(10, 6))
    plt.hist(mndwi_values, bins=100, range=(-1, 1), color='blue', alpha=0.7)
    plt.axvline(x=0.0, color='red', linestyle='--', label='Threshold 0.0')
    plt.title("MNDWI Histogram")
    plt.xlabel("MNDWI Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.show()
else:
    print("No MNDWI data found. Check if S2 export included MNDWI band.")
```
