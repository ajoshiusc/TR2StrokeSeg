#!/usr/bin/env python3
import os
import subprocess
import argparse
from pathlib import Path

"""
Script to run BrainSuite's BSE (Brain Surface Extractor) for skull stripping.
This script searches for T1-weighted images in specified directories and 
generates skull-stripped brain images and brain masks.
"""

def find_t1_images(search_dirs):
    """Finds all T1-weighted images in the given directories."""
    t1_patterns = ["*_0000.nii.gz", "*_T1w.nii.gz"]
    t1_files = []
    
    for d in search_dirs:
        path = Path(d)
        if not path.exists():
            print(f"Warning: Directory {d} does not exist. Skipping.")
            continue
        
        for pattern in t1_patterns:
            t1_files.extend(list(path.rglob(pattern)))
    
    # Use absolute paths and ensure uniqueness
    unique_files = {f.resolve(): f for f in t1_files}
    return sorted(list(unique_files.values()))

def run_skull_stripping(t1_files, bse_path):
    """Runs BSE on each T1 image."""
    print(f"Processing {len(t1_files)} images...")
    
    for i, t1_path in enumerate(t1_files):
        print(f"[{i+1}/{len(t1_files)}] Processing {t1_path.name}...")
        # Determine base name for output
        filename = t1_path.name
        if filename.endswith("_0000.nii.gz"):
            base_name = filename.replace("_0000.nii.gz", "")
        elif filename.endswith("_T1w.nii.gz"):
            base_name = filename.replace("_T1w.nii.gz", "")
        else:
            # Fallback for other .nii.gz files
            base_name = filename.split('.')[0]
            
        # Define output paths in the same directory as the T1 image
        output_dir = t1_path.parent
        brain_out = output_dir / f"{base_name}_brain.nii.gz"
        mask_out = output_dir / f"{base_name}_mask.nii.gz"
        
        # Skip if already exists
        if brain_out.exists() and mask_out.exists():
            continue
            
        # Build command
        # -i: input, -o: output brain, --mask: output mask, --auto: automatic parameter selection,
        # -p: dilate mask,
        cmd = [
            str(bse_path),
            "-i", str(t1_path),
            "-o", str(brain_out),
            "--mask", str(mask_out),
            "--auto",
            "-p",
        ]
        
        try:
            # Execute bse
            result = subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except subprocess.CalledProcessError as e:
            print(f"\nError processing {t1_path.name}:")
            print(e.stderr.decode())
        except Exception as e:
            print(f"\nUnexpected error processing {t1_path.name}: {e}")

def main():
    # Define default paths based on workspace structure
    workspace_root = Path("/home/ajoshi/Projects/TR2StrokeSeg")
    default_input_dirs = [
        str(workspace_root / "Stroke_selected_samples")
    ]
    default_bse_path = "/home/ajoshi/Software/BrainSuite23a/bin/bse"

    parser = argparse.ArgumentParser(description="Run BrainSuite BSE skull stripping on T1 images.")
    parser.add_argument("--input_dirs", nargs="+", default=default_input_dirs, 
                        help="Directories to search for T1 images (default: Stroke_selected_samples)")
    parser.add_argument("--bse_path", default=default_bse_path,
                        help=f"Path to the BSE executable (default: {default_bse_path})")

    args = parser.parse_args()
    
    # Check if BSE exists
    bse_exe = Path(args.bse_path)
    if not bse_exe.exists():
        print(f"Error: BSE executable not found at {args.bse_path}")
        print("Please check the path to BrainSuite.")
        return

    # Find images
    t1_files = find_t1_images(args.input_dirs)
    if not t1_files:
        print("No T1 images found in the specified directories.")
        return
        
    print(f"Found {len(t1_files)} unique T1 images.")
    
    # Run processing
    run_skull_stripping(t1_files, bse_exe)
    
    print(f"\nSkull stripping complete.")

if __name__ == "__main__":
    main()
