import os
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import sys
from tqdm import tqdm
from collections import defaultdict

def find_largest_lesions(data_dir, top_k=10):
    root_dir = Path(data_dir)
    if not root_dir.exists():
        print(f"Error: Directory {data_dir} does not exist.")
        return

    # If the user pointed directly to 'Training', we look inside.
    if (root_dir / "Training").exists():
         search_dir = root_dir / "Training"
    else:
         search_dir = root_dir

    print(f"Recursively searching for mask files in {search_dir}...")
    
    # Updated pattern based on user input
    # User example: sub-r002s001_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz
    # Use rglob to find all .nii.gz files, then filter for the specific pattern
    all_nii_files = list(search_dir.rglob('*.nii.gz'))
    print(f"Found {len(all_nii_files)} .nii.gz files in total.")
    
    mask_files = []
    for nii_file in all_nii_files:
        name = nii_file.name
        # Match "lesion_mask" as seen in the user example
        if 'lesion_mask.nii.gz' in name:
            mask_files.append(nii_file)
            
    print(f"Found {len(mask_files)} potential mask files.")

    # Group files by subject ID to handle potential split masks (e.g. L/R)
    subject_masks = defaultdict(list)
    
    for mask_file in mask_files:
        # extract sub-xxx for ID
        parts = mask_file.name.split('_')
        subject_id = "unknown"
        for part in parts:
            if part.startswith("sub-"):
                subject_id = part
                break
        
        if subject_id == "unknown":
             # fallback: try directory parts
             for part in reversed(mask_file.parts):
                 if part.startswith("sub-"):
                     subject_id = part
                     break
        
        if subject_id == "unknown":
            subject_id = mask_file.parent.name # fallback just in case

        subject_masks[subject_id].append(mask_file)
    
    print(f"Identified {len(subject_masks)} unique subjects.")
    
    subject_volumes = []
    
    print("Calculating volumes...")
    for subject_id, masks in tqdm(subject_masks.items()):
        total_volume_mm3 = 0.0
        
        for mask_file in masks:
            try:
                # Load with mmap=True to avoid reading full file into memory at once if possible, 
                # although get_fdata() will eventually read it. 
                # Converting to uint8 immediately might save memory if the mask is small integer.
                img = nib.load(mask_file)
                
                # Check zooms
                zooms = img.header.get_zooms()
                voxel_volume = np.prod(zooms[:3])
                
                # optimization: if we can use the dataobj directly without full float64 conversion
                data = img.get_fdata(dtype=np.float32) 
                
                lesion_voxels = np.count_nonzero(data > 0)
                volume_mm3 = lesion_voxels * voxel_volume
                
                total_volume_mm3 += volume_mm3
                
            except Exception as e:
                print(f"Error processing {mask_file}: {e}")
        
        subject_volumes.append({
            'subject': subject_id,
            'volume': total_volume_mm3,
            'paths': [str(p) for p in masks]
        })

    # Sort by volume descending
    subject_volumes.sort(key=lambda x: x['volume'], reverse=True)

    print("\n" + "="*80)
    print(f"Top {top_k} Brains with Largest Lesions")
    print("="*80)
    print(f"{'Rank':<5} {'Subject ID':<20} {'Volume (mm³)':<15} {'Volume (ml)':<15} {'File Count'}")
    print("-" * 90)
    
    for i, item in enumerate(subject_volumes[:top_k]):
        vol_ml = item['volume'] / 1000.0
        file_count = len(item['paths'])
        print(f"{i+1:<5} {item['subject']:<20} {item['volume']:<15.2f} {vol_ml:<15.2f} {file_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Path to data directory")
    parser.add_argument("--top_k", type=int, default=10, help="Number of top subjects to list")
    args = parser.parse_args()
    
    find_largest_lesions(args.data_dir, args.top_k)
