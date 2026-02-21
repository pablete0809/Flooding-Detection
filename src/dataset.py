import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
import rasterio
from datetime import datetime

class FloodTimeSeriesDataset(Dataset):
    def __init__(self, root_dir, history_length=3, forecast_horizon=5, transform=None):
        """
        Dataset for Flood Forecasting using Time Series.
        
        Args:
            root_dir (str): Path to dataset root (e.g., 'dataset_sen12flood_v1')
            history_length (int): Number of past frames to use as input (T-n ... T)
            forecast_horizon (int): Number of future frames to predict (T+1 ... T+k)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.history_length = history_length
        self.forecast_horizon = forecast_horizon
        self.transform = transform
        
        self.s1_dir = os.path.join(root_dir, 'S1')
        self.s2_dir = os.path.join(root_dir, 'S2')
        self.terrain_dir = os.path.join(root_dir, 'Terrain')
        self.label_dir = os.path.join(root_dir, 'labels')
        
        self.samples = self._prepare_samples()
        print(f"Dataset initialized: {len(self.samples)} samples found.")

    def _prepare_samples(self):
        """
        Scans directories, groups files by Tile ID, sorts by Date, 
        and creates sliding window samples.
        """
        # 1. Gather all files
        # Assumption: Filename format is tile_{id}_{date}.tif
        # We use S2 dir as the source of truth
        if not os.path.exists(self.s2_dir):
            print(f"Warning: {self.s2_dir} does not exist.")
            return []
            
        files = [f for f in os.listdir(self.s2_dir) if f.endswith('.tif')]
        
        # 2. Parse and Group
        tile_groups = {} # {tile_id: [(date, filename), ...]}
        
        for f in files:
            parts = f.replace('.tif', '').split('_')
            # tile_{id}_{date}
            # parts[0] = tile, parts[1] = id, parts[2] = date
            if len(parts) >= 3:
                tile_id = parts[1]
                date_str = parts[2]
                try:
                    # Assuming date format YYYYMMDD or YYYY-MM-DD from GEE
                    # If GEE output is '2024-10-29', simple sort works
                    tile_groups.setdefault(tile_id, []).append((date_str, f))
                except Exception:
                    pass
        
        # 3. Create Sliding Windows per Tile
        samples = []
        required_len = self.history_length + self.forecast_horizon
        
        for tile_id, file_list in tile_groups.items():
            # Sort by date
            file_list.sort(key=lambda x: x[0])
            
            # Check if we have enough dates
            if len(file_list) < required_len:
                continue
                
            # Create windows
            for i in range(len(file_list) - required_len + 1):
                # Window slice
                window = file_list[i : i + required_len]
                
                # Split into Input and Target
                input_files = [x[1] for x in window[:self.history_length]]
                target_files = [x[1] for x in window[self.history_length:]]
                
                samples.append({
                    'tile_id': tile_id,
                    'input_files': input_files,
                    'target_files': target_files
                })
                
        return samples

    def __len__(self):
        return len(self.samples)

    def _load_tiff(self, path):
        with rasterio.open(path) as src:
            return src.read().astype(np.float32)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        
        # --- Load Inputs ---
        # Stack S1 + S2 for each history step
        # Dimensions: (History_Len * Channels, H, W)
        # S2: 8 bands (including Mask)
        # S1: 3 bands
        # Total per step: 11 bands
        input_tensors = []
        
        for fname in sample_info['input_files']:
            # Load S1 (3 bands)
            s1 = self._load_tiff(os.path.join(self.s1_dir, fname))
            # Load S2 (8 bands) - Band 7 is S2_MASK
            s2 = self._load_tiff(os.path.join(self.s2_dir, fname))
            
            # Concatenate bands: shape (11, H, W)
            # Order: S2 (8) + S1 (3)
            # Mask is channel 7
            combined = np.concatenate([s2, s1], axis=0)
            input_tensors.append(combined)
            
        # Stack along channel dimension
        # Result: (History * 11, H, W)
        x_dynamic = np.concatenate(input_tensors, axis=0)
        
        # --- Load Terrain (Static) ---
        # Terrain: 5 bands (DEM, Slope, HAND, DW_Water, DW_Built)
        last_input_file = sample_info['input_files'][-1]
        terrain = self._load_tiff(os.path.join(self.terrain_dir, last_input_file)) # (5, H, W)
        
        # Combine Dynamic + Static
        # Input Tensor: (History*11 + 5, H, W)
        x = np.concatenate([x_dynamic, terrain], axis=0)
        
        # --- Load Targets ---
        # Future Flood Labels
        targets = []
        for fname in sample_info['target_files']:
            lbl = self._load_tiff(os.path.join(self.label_dir, fname)) # (1, H, W)
            targets.append(lbl)
            
        # Stack targets: (Forecast_Horizon, H, W)
        y = np.concatenate(targets, axis=0)
        
        # Convert to Torch Tensors
        x_tensor = torch.from_numpy(x)
        y_tensor = torch.from_numpy(y)
        
        return x_tensor, y_tensor
