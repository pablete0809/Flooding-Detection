import os
import rasterio
import torch
import numpy as np
import mlstac
import sen2sr
from rasterio.enums import Resampling
import argparse
from pathlib import Path

def apply_superres(input_dir, output_dir, model_path="model/SEN2SRLite_RGBN"):
    """
    Applies SEN2SR super-resolution to Sentinel-2 images in input_dir.
    
    Args:
        input_dir (str): Directory containing 10m Sentinel-2 images (TIF).
        output_dir (str): Directory to save 2.5m Super-Resolved images.
        model_path (str): Path to the downloaded SEN2SR model.
    """
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Check for GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    print(f"Loading model from {model_path}...")
    try:
        if not os.path.exists(model_path):
             print(f"Model not found at {model_path}. Attempting download...")
             mlstac.download(
                file="https://huggingface.co/tacofoundation/sen2sr/resolve/main/SEN2SRLite/NonReference_RGBN_x4/mlm.json",
                output_dir=model_path
             )

        model = mlstac.load(model_path).compiled_model(device=device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Process images
    files = [f for f in os.listdir(input_dir) if f.endswith('.tif')]
    print(f"Found {len(files)} images to process.")

    for idx, filename in enumerate(files):
        src_path = os.path.join(input_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        
        if os.path.exists(dst_path):
            print(f"[{idx+1}/{len(files)}] Skipping {filename} (already exists)")
            continue
            
        print(f"[{idx+1}/{len(files)}] Processing {filename}...")
        
        try:
            with rasterio.open(src_path) as src:
                # Check band count
                if src.count < 4:
                    print(f"Skipping {filename}: Not enough bands (needs at least 4 for RGBN)")
                    continue

                # We need [Red, Green, Blue, NIR] -> [3, 2, 1, 4]
                img_data = src.read([3, 2, 1, 4]) 
                
                # Normalize
                if src.dtypes[0] == 'uint16':
                     img = img_data.astype('float32') / 10000.0
                else:
                     img = img_data.astype('float32')
                
                img = np.clip(img, 0, 1)

                # Prepare tensor: (C, H, W)
                X = torch.from_numpy(img).float().to(device)
                X = torch.nan_to_num(X)
                
                # Run Inference with predict_large for memory safety
                # predict_large handles tiling automatically
                with torch.no_grad():
                    sr_output = sen2sr.predict_large(
                        model=model,
                        X=X,
                        overlap=32
                    )
                    
                    # predict_large returns tensor on device usually, move to cpu numpy
                    if isinstance(sr_output, torch.Tensor):
                        sr_output = sr_output.cpu().numpy()
                
                # Save Result
                new_transform = src.transform * src.transform.scale(0.25, 0.25)
                new_height = sr_output.shape[1]
                new_width = sr_output.shape[2]
                
                profile = src.profile.copy()
                profile.update({
                    'height': new_height,
                    'width': new_width,
                    'transform': new_transform,
                    'count': sr_output.shape[0], # Should be 4
                    'dtype': 'float32',
                    'driver': 'GTiff'
                })
                
                with rasterio.open(dst_path, 'w', **profile) as dst:
                    dst.write(sr_output)
                    
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            # Optional: Clear CUDA cache on error
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean/Process Sentinel-2 Super-Resolution")
    parser.add_argument("--input_dir", type=str, required=True, help="Input directory (S2 10m)")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory (S2 2.5m)")
    args = parser.parse_args()
    
    apply_superres(args.input_dir, args.output_dir)
