import os
import rasterio
from rasterio.enums import Resampling
import argparse

def resize_images(target_ref_dir, source_dir, output_dir, is_label=False):
    """
    Resizes images in source_dir to match the resolution/transform of images in target_ref_dir.
    
    Args:
        target_ref_dir (str): Directory with 2.5m S2 images (reference).
        source_dir (str): Directory with 10m images (S1 or Labels) to resize.
        output_dir (str): Directory to save resized images.
        is_label (bool): If True, uses Nearest Neighbor resampling (for classes).
                         If False, uses Bilinear/Bicubic (for continuous data).
    """
    os.makedirs(output_dir, exist_ok=True)
    
    ref_files = [f for f in os.listdir(target_ref_dir) if f.endswith('.tif')]
    print(f"Found {len(ref_files)} reference files. aligning {source_dir}...")
    
    for filename in ref_files:
        ref_path = os.path.join(target_ref_dir, filename)
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(output_dir, filename)
        
        if not os.path.exists(src_path):
            print(f"Warning: Source file {src_path} not found. Skipping.")
            continue
            
        # Get reference metadata (Target Resolution/Dimensions)
        with rasterio.open(ref_path) as ref:
            target_height = ref.height
            target_width = ref.width
            target_transform = ref.transform
            target_crs = ref.crs
            
        # Resize Source Image
        resampling_method = Resampling.nearest if is_label else Resampling.bilinear
        
        try:
            with rasterio.open(src_path) as src:
                # Read and resample
                data = src.read(
                    out_shape=(
                        src.count,
                        target_height,
                        target_width
                    ),
                    resampling=resampling_method
                )
                
                profile = src.profile.copy()
                profile.update({
                    'height': target_height,
                    'width': target_width,
                    'transform': target_transform,
                    'crs': target_crs
                })
                
                with rasterio.open(dst_path, 'w', **profile) as dst:
                    dst.write(data)
                    
        except Exception as e:
            print(f"Error resizing {filename}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resize S1/Labels to match S2 Super-Resolution")
    parser.add_argument("--ref_dir", type=str, required=True, help="Reference directory (S2 2.5m)")
    parser.add_argument("--s1_dir", type=str, help="Input S1 directory (10m)")
    parser.add_argument("--s1_out", type=str, help="Output S1 directory (2.5m)")
    parser.add_argument("--label_dir", type=str, help="Input Label directory (10m)")
    parser.add_argument("--label_out", type=str, help="Output Label directory (2.5m)")
    
    args = parser.parse_args()
    
    if args.s1_dir and args.s1_out:
        print("Resizing Sentinel-1...")
        resize_images(args.ref_dir, args.s1_dir, args.s1_out, is_label=False)
        
    if args.label_dir and args.label_out:
        print("Resizing Labels...")
        resize_images(args.ref_dir, args.label_dir, args.label_out, is_label=True)
