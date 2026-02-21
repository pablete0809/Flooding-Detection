# Flood Detection Pipeline — Valencia DANA 2024

> A research project for **automatic flood detection** using multi-source satellite data (Sentinel-1 SAR + Sentinel-2 Optical) and deep learning, applied to the **Valencia DANA flood event of October 2024**.

---

## What does this project do?

This repository implements a complete **end-to-end pipeline** for flood detection from satellite imagery:

1. **Data Acquisition** — Downloads co-registered Sentinel-1 (radar) and Sentinel-2 (optical) images from Google Earth Engine (GEE) for a given region and time range.
2. **Dataset Generation** — Splits the data into structured patches (tiles) with all feature bands: `S1 VV`, `S1 VH`, `S2 optical`, `DEM`, `HAND`, `Slope`, `Permanent Water`, and a `S2 validity mask` for handling cloud-covered or missing optical data.
3. **Weak Label Generation** — Automatically generates flood labels using the MNDWI index, refined with Dynamic World data to exclude false positives over roads and urban areas.
4. **Model Training** — Trains a U-Net segmentation model on the generated dataset to predict flood extent.
5. **Ablation Study** — Runs a Random Forest ablation to quantify the contribution of each input feature (S1, terrain, HAND, etc.).
6. **Super-Resolution (Optional)** — Upsamples Sentinel-2 images from 10m to 2.5m resolution using the SEN2SR model.

---

## 📁 Repository Structure

```
.
├── gee_pipeline.py          # Core GEE data fetching & processing logic
├── main.ipynb               # Main notebook: data exploration, download, visualization
├── train.py                 # U-Net training script
├── train_ablation.py        # Random Forest ablation study
├── requirements.txt         # Python dependencies
├── src/
│   ├── dataset.py           # PyTorch Dataset class for training
│   └── model.py             # Simple U-Net model definition
├── scripts/
│   ├── pipeline_orchestrator.py   # Runs the full super-resolution pipeline
│   ├── apply_superres.py          # Applies SEN2SR super-resolution to S2 tiles
│   └── resize_s1_labels.py        # Rescales S1 and labels to match S2 high-res
└── dataset_sen12flood_v1/   # Generated dataset (git-ignored)
    ├── S1/                  # Sentinel-1: VV, VH, Ratio (3 bands)
    ├── S2/                  # Sentinel-2: B2,B3,B4,B8,B11,NDWI,MNDWI + S2_MASK (8 bands)
    ├── Terrain/             # DEM, Slope, HAND, DW_Water, DW_Built (5 bands)
    └── labels/              # Binary flood labels (1 band)
```

---

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- A valid [Google Earth Engine account](https://earthengine.google.com/) (free for research)
- *(Optional)* A CUDA-compatible GPU for training

### Setup

```bash
# 1. Clone the repository
git clone <repo-url>
cd TFG

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt

# 4. Authenticate with Google Earth Engine (one-time only)
earthengine authenticate
```

---

## Usage: Step-by-Step

### Step 1 — Explore & Download Data (`main.ipynb`)

Open the notebook to configure your region of interest and download the dataset:

```bash
jupyter notebook main.ipynb
```

The notebook walks you through:
- **Defining your ROI** (default: Valencia, Spain — Oct/Nov 2024 flood event)
- **Visualizing** Sentinel-1, Sentinel-2, MNDWI, DEM, HAND, and flood labels on an interactive map
- **Downloading the dataset** into `dataset_sen12flood_v1/` by running the `download_patches(...)` cell

> ⚠️ The download can take several minutes depending on the region size and number of dates.

The downloaded dataset will have the following band structure per tile:

| Modality | Bands | Notes |
|---|---|---|
| Sentinel-2 | 7 spectral + 1 mask | `S2_MASK=1` if valid, `S2_MASK=0` if cloudy/missing |
| Sentinel-1 | VV, VH, VV-VH ratio | In dB, water appears dark |
| Terrain | DEM, Slope, HAND, DW_Water, DW_Built | From NASADEM + Dynamic World |
| Label | Binary (0/1) | Flood = 1, generated from MNDWI + DW refinement |

---

### Step 2 — Train the Deep Learning Model (`train.py`)

Once the dataset is downloaded, train the U-Net segmentation model:

```bash
python3 train.py
```

This will:
- Load time-series patches from `dataset_sen12flood_v1/`
- Train a **U-Net** for 10 epochs to predict flood maps
- Save the trained model to `checkpoints/flood_forecast_model.pth`

**Model input configuration:**
- History length: **3 frames** per tile (temporal context)
- Input channels: **38** total `(3 × (8 S2 + 3 S1)) + 5 Terrain`
- Output channels: **5** (flood prediction for the next 5 days)

---

### Step 3 — Run the Ablation Study (`train_ablation.py`)

To understand which input features are most important for flood detection:

```bash
python3 train_ablation.py
```

This script:
- Trains a **Random Forest** classifier with different combinations of features
- Compares: SAR-only, SAR+HAND, SAR+DEM, SAR+Full Terrain
- Outputs F1/Precision/Recall metrics to `ablation_results.csv`
- Generates a **SHAP feature importance** plot (`shap_summary.png`)

---

### Step 4 *(Optional)* — Super-Resolution Pipeline

To upscale Sentinel-2 tiles from 10m → 2.5m resolution using the SEN2SR model:

```bash
python3 scripts/pipeline_orchestrator.py --dataset_dir dataset_sen12flood_v1
```

This will generate new `_HighRes` folders:
- `S2_HighRes/` — Sentinel-2 at 2.5m
- `S1_HighRes/` — Sentinel-1 rescaled to 2.5m
- `labels_HighRes/` — Labels rescaled to 2.5m (nearest-neighbor)

---

## Key Design Choices

### Handling Missing Sentinel-2 Data
Sentinel-2 is an optical sensor — it cannot see through clouds. This project handles this explicitly with a **`S2_MASK` band** exported alongside each tile:
- `S2_MASK = 1` → S2 data is valid and trustworthy
- `S2_MASK = 0` → Cloudy/missing; model should rely on S1 + Terrain instead

The model receives this mask as an input channel, allowing it to learn to weight features by their availability.

### Improved Water Labeling
Simple MNDWI thresholding tends to falsely label roads and waterlogged fields as flood. This pipeline refines labels by combining:
- **MNDWI > 0.0** (spectral water index)
- **Dynamic World `dw_built` < 0.4** (exclude urban/road pixels)

### Permanent Water from Dynamic World
Instead of using a static, low-resolution dataset, this project fetches the **Dynamic World V1** `water` probability band (10m resolution, ~2023 annual composite) as a proxy for permanent water extent — significantly more accurate for flood *change* detection.

---

## 📊 Dataset Band Reference

| Band | Source | Description |
|---|---|---|
| `S2_B2, B3, B4` | Sentinel-2 | RGB — Blue, Green, Red |
| `S2_B8` | Sentinel-2 | NIR |
| `S2_B11` | Sentinel-2 | SWIR |
| `S2_NDWI` | Sentinel-2 | Normalized Difference Water Index |
| `S2_MNDWI` | Sentinel-2 | Modified NDWI (better for turbid/flood water) |
| `S2_MASK` | Computed | 1=Valid, 0=Cloudy/Missing |
| `S1_VV` | Sentinel-1 | VV polarization (dB) |
| `S1_VH` | Sentinel-1 | VH polarization (dB) |
| `S1_VV_VH_ratio` | Sentinel-1 | VV−VH ratio (dB) |
| `elevation` | NASADEM | Terrain height (m) |
| `slope` | NASADEM | Terrain slope (°) |
| `hand` | Global HAND | Height Above Nearest Drainage (m) |
| `dw_water` | Dynamic World | Permanent water probability (0–1) |
| `dw_built` | Dynamic World | Built-up/urban probability (0–1) |
| `LABEL_flood_raw` | Computed | Binary flood label (0=No flood, 1=Flood) |
