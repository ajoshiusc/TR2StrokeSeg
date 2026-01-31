import os
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import sys
from tqdm import tqdm
from collections import defaultdict
import shutil

def find_t1_for_mask(mask_path):
    # Mask: .../sub-r002s001_ses-1_space-MNI152NLin2009aSym_label-L_desc-T1lesion_mask.nii.gz
    # T1:   .../sub-r002s001_ses-1_space-MNI152NLin2009aSym_T1w.nii.gz
    
    # Strategy 1: Look in the same directory for *T1w.nii.gz
    parent_dir = mask_path.parent
    t1_candidates = list(parent_dir.glob('*T1w.nii.gz'))
    
    if len(t1_candidates) == 1:
        return t1_candidates[0]
    elif len(t1_candidates) > 1:
        # If multiple, try to match the prefix?
        # Usually looking for the one that shares the most common prefix
        # But in ATLAS, usually one T1w per anat folder.
        return t1_candidates[0]
        
    return None

def copy_subjects(subjects, category, output_base_dir):
    dest_dir = output_base_dir / category
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCopying {len(subjects)} subjects to {dest_dir}...")
    
    for subject in subjects:
        # subject dictionary from find_largest_lesions logic
        # {'subject': 'sub-xxx', 'volume': 123.0, 'paths': ['path/to/mask.nii.gz']}
        
        # We might have multiple masks (though typically 1 per session/anat)
        # We will copy all masks and their corresponding T1s
        
        for mask_path_str in subject['paths']:
            mask_path = Path(mask_path_str)
            
            # Find T1
            t1_path = find_t1_for_mask(mask_path)
            
            if t1_path and t1_path.exists():
                # Copy files
                # We retain the filename to ensure uniqueness
                try:
                    shutil.copy2(mask_path, dest_dir / mask_path.name)
                    shutil.copy2(t1_path, dest_dir / t1_path.name)
                    # print(f"  Copied {subject['subject']}")
                except Exception as e:
                    print(f"  Error copying {subject['subject']}: {e}")
            else:
                print(f"  Warning: T1 not found for {mask_path.name}")

def select_and_copy_samples(data_dir, output_dir, count=10):
    root_dir = Path(data_dir)
    output_base_dir = Path(output_dir)
    
    if not root_dir.exists():
        print(f"Error: Directory {data_dir} does not exist.")
        return

    # If the user pointed directly to 'Training', we look inside.
    if (root_dir / "Training").exists():
         search_dir = root_dir / "Training"
    else:
         search_dir = root_dir

    print(f"Scanning {search_dir}...")
    
    # 1. Find all mask files
    all_nii_files = list(search_dir.rglob('*.nii.gz'))
    mask_files = [f for f in all_nii_files if 'lesion_mask.nii.gz' in f.name]
    
    print(f"Found {len(mask_files)} mask files.")
    
    # 2. Group by subject
    subject_masks = defaultdict(list)
    for mask_file in mask_files:
        # Extract ID
        parts = mask_file.name.split('_')
        subject_id = "unknown"
        for part in parts:
            if part.startswith("sub-"):
                subject_id = part
                break
        if subject_id == "unknown":
             # fallback
             for part in reversed(mask_file.parts):
                 if part.startswith("sub-"):
                     subject_id = part
                     break
        if subject_id == "unknown":
            subject_id = mask_file.parent.name
            
        subject_masks[subject_id].append(mask_file)

    # 3. Calculate volumes
    print("Calculating volumes...")
    subject_volumes = []
    
    for subject_id, masks in tqdm(subject_masks.items()):
        total_volume_mm3 = 0.0
        valid_masks = []
        
        for mask_file in masks:
            try:
                # Optimized read
                img = nib.load(mask_file)
                zooms = img.header.get_zooms()
                voxel_volume = np.prod(zooms[:3])
                
                # Check header dims without reading data if possible? 
                # Nope, need content for lesion size.
                # Use mmap
                data = img.get_fdata(dtype=np.float32)
                
                lesion_voxels = np.count_nonzero(data > 0)
                volume_mm3 = lesion_voxels * voxel_volume
                
                # Only counts if it actually has volume
                if volume_mm3 > 0:
                    total_volume_mm3 += volume_mm3
                    valid_masks.append(mask_file)
                
            except Exception as e:
                print(f"Error processing {mask_file}: {e}")
        
        if total_volume_mm3 > 0:
            subject_volumes.append({
                'subject': subject_id,
                'volume': total_volume_mm3,
                'paths': [str(p) for p in valid_masks]
            })

    # 4. Sort
    subject_volumes.sort(key=lambda x: x['volume'], reverse=True)
    total_subjects = len(subject_volumes)
    print(f"\nTotal subjects with non-zero lesions: {total_subjects}")
    
    if total_subjects < count * 3:
        print(f"Warning: Not enough subjects for requested count {count} per group.")
    
    # 5. Select Groups
    largest = subject_volumes[:count]
    
    # For small, take the tail
    smallest = subject_volumes[-count:]
    
    # For medium, take the middle slice
    mid_idx = total_subjects // 2
    start_mid = max(0, mid_idx - (count // 2))
    end_mid = start_mid + count
    medium = subject_volumes[start_mid:end_mid]
    
    # Print selections
    print("\n--- Selection Summary ---")
    print(f"Largest (Range): {largest[0]['volume']:.2f} - {largest[-1]['volume']:.2f} mm3")
    print(f"Medium  (Range): {medium[0]['volume']:.2f} - {medium[-1]['volume']:.2f} mm3")
    print(f"Smallest (Range): {smallest[0]['volume']:.2f} - {smallest[-1]['volume']:.2f} mm3")
    
    # 6. Copy
    copy_subjects(largest, "large_lesions", output_base_dir)
    copy_subjects(medium, "medium_lesions", output_base_dir)
    copy_subjects(smallest, "small_lesions", output_base_dir)
    
    print("\nDone!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", help="Source Atlas2 data directory")
    parser.add_argument("output_dir", help="Destination directory for samples")
    parser.add_argument("--count", type=int, default=10, help="Number of subjects per group")
    args = parser.parse_args()
    
    select_and_copy_samples(args.data_dir, args.output_dir, args.count)
