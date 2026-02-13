import os
import subprocess
import argparse
import sys

def run_pipeline(base_dataset_dir):
    """
    Orchestrates:
    1. (Optional/Manual) gee_pipeline.py should have run first to generate S1/S2/labels.
    2. apply_superres.py -> Generates S2_HR
    3. resize_s1_labels.py -> Generates S1_HR and Labels_HR
    """
    
    s2_input = os.path.join(base_dataset_dir, "S2")
    s2_output = os.path.join(base_dataset_dir, "S2_HighRes")
    
    s1_input = os.path.join(base_dataset_dir, "S1")
    s1_output = os.path.join(base_dataset_dir, "S1_HighRes")
    
    label_input = os.path.join(base_dataset_dir, "labels")
    label_output = os.path.join(base_dataset_dir, "labels_HighRes")
    
    # Check if inputs exist
    if not os.path.exists(s2_input):
        print(f"Error: Input S2 directory {s2_input} does not exist.")
        print("Please run gee_pipeline.py (or main.ipynb) first to download data.")
        return

    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define paths to sibling scripts
    apply_superres_script = os.path.join(script_dir, "apply_superres.py")
    resize_s1_labels_script = os.path.join(script_dir, "resize_s1_labels.py")

    print("=== Step 1: Starting Sentinel-2 Super-Resolution (10m -> 2.5m) ===")
    subprocess.run([
        sys.executable, apply_superres_script,
        "--input_dir", s2_input,
        "--output_dir", s2_output
    ], check=True)
    
    print("\n=== Step 2: Resizing Sentinel-1 and Labels (to match 2.5m) ===")
    
    cmd = [
        sys.executable, resize_s1_labels_script,
        "--ref_dir", s2_output,
    ]
    
    if os.path.exists(s1_input):
        cmd.extend(["--s1_dir", s1_input, "--s1_out", s1_output])
    
    if os.path.exists(label_input):
        cmd.extend(["--label_dir", label_input, "--label_out", label_output])
        
    subprocess.run(cmd, check=True)
    
    print("\n=== Pipeline Completed ===")
    print(f"High Res Dataset available at:")
    print(f" - S2: {s2_output}")
    print(f" - S1: {s1_output}")
    print(f" - Labels: {label_output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Full Super-Res Pipeline")
    parser.add_argument("--dataset_dir", type=str, default="dataset_sen12flood_v1", help="Base dataset directory containing S1, S2, labels folders")
    args = parser.parse_args()
    
    run_pipeline(args.dataset_dir)
