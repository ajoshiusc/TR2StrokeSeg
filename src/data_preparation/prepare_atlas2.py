"""
Data preparation script for Atlas2 dataset to nn-UNet format.

This script converts Atlas2 stroke lesion dataset into the format required by nn-UNet.
Atlas2 dataset structure is expected to be:
    Atlas2/
        Training/
            R001/
                t1w.nii.gz (or similar naming)
                lesion_mask.nii.gz
            R002/
                ...

nn-UNet expects:
    nnUNet_raw/
        Dataset001_Atlas2/
            imagesTr/
                Atlas2_001_0000.nii.gz  (T1-weighted image)
            labelsTr/
                Atlas2_001.nii.gz       (lesion mask)
            dataset.json
"""

import os
import json
import shutil
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np


def create_dataset_json(output_folder, num_training_cases):
    """
    Create dataset.json file required by nn-UNet.
    
    Args:
        output_folder: Path to the dataset folder
        num_training_cases: Number of training cases
    """
    dataset_json = {
        "channel_names": {
            "0": "T1"
        },
        "labels": {
            "background": 0,
            "stroke_lesion": 1
        },
        "numTraining": num_training_cases,
        "file_ending": ".nii.gz",
        "name": "Atlas2",
        "description": "Atlas2 stroke lesion segmentation dataset",
        "reference": "Liew et al., 2018",
        "licence": "Public domain",
        "release": "2.0"
    }
    
    with open(os.path.join(output_folder, 'dataset.json'), 'w') as f:
        json.dump(dataset_json, f, indent=4)


def prepare_atlas2_dataset(atlas2_dir, output_dir, dataset_id=1):
    """
    Convert Atlas2 dataset to nn-UNet format.
    
    Args:
        atlas2_dir: Path to Atlas2 dataset root directory
        output_dir: Path to nnUNet_raw directory
        dataset_id: Dataset ID for nn-UNet (default: 1)
    """
    atlas2_dir = Path(atlas2_dir)
    output_dir = Path(output_dir)
    
    # Create dataset folder with proper naming
    dataset_name = f"Dataset{dataset_id:03d}_Atlas2"
    dataset_folder = output_dir / dataset_name
    images_tr_folder = dataset_folder / "imagesTr"
    labels_tr_folder = dataset_folder / "labelsTr"
    
    # Create directories
    images_tr_folder.mkdir(parents=True, exist_ok=True)
    labels_tr_folder.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing Atlas2 dataset from {atlas2_dir}")
    print(f"Output directory: {dataset_folder}")
    
    # Find all subject folders in Training directory
    training_dir = atlas2_dir / "Training"
    if not training_dir.exists():
        raise ValueError(f"Training directory not found: {training_dir}")
    
    subject_folders = sorted([d for d in training_dir.iterdir() if d.is_dir()])
    
    if len(subject_folders) == 0:
        raise ValueError(f"No subject folders found in {training_dir}")
    
    print(f"Found {len(subject_folders)} subjects")
    
    case_id = 0
    for subject_folder in subject_folders:
        subject_id = subject_folder.name
        print(f"Processing subject {subject_id}...")
        
        # Find T1-weighted image (common naming patterns)
        t1_patterns = ['t1w.nii.gz', 't1.nii.gz', 'T1.nii.gz', 'T1w.nii.gz']
        t1_file = None
        for pattern in t1_patterns:
            candidate = subject_folder / pattern
            if candidate.exists():
                t1_file = candidate
                break
        
        # If not found, try to find any .nii.gz file that's not a mask
        if t1_file is None:
            nii_files = list(subject_folder.glob("*.nii.gz"))
            for nii_file in nii_files:
                if 'mask' not in nii_file.name.lower() and 'label' not in nii_file.name.lower():
                    t1_file = nii_file
                    break
        
        # Find lesion mask
        mask_patterns = ['lesion_mask.nii.gz', 'mask.nii.gz', 'lesion.nii.gz', 'label.nii.gz']
        mask_file = None
        for pattern in mask_patterns:
            candidate = subject_folder / pattern
            if candidate.exists():
                mask_file = candidate
                break
        
        # If not found, try to find any file with 'mask' or 'label' in name
        if mask_file is None:
            for nii_file in subject_folder.glob("*.nii.gz"):
                if 'mask' in nii_file.name.lower() or 'label' in nii_file.name.lower():
                    mask_file = nii_file
                    break
        
        if t1_file is None:
            print(f"  Warning: T1 image not found for {subject_id}, skipping...")
            continue
        
        if mask_file is None:
            print(f"  Warning: Lesion mask not found for {subject_id}, skipping...")
            continue
        
        # Copy and rename files to nn-UNet format
        case_id += 1
        output_image_name = f"Atlas2_{case_id:03d}_0000.nii.gz"
        output_label_name = f"Atlas2_{case_id:03d}.nii.gz"
        
        shutil.copy2(t1_file, images_tr_folder / output_image_name)
        shutil.copy2(mask_file, labels_tr_folder / output_label_name)
        
        print(f"  Processed case {case_id}: {subject_id}")
    
    # Create dataset.json
    create_dataset_json(dataset_folder, case_id)
    
    print(f"\nDataset preparation complete!")
    print(f"Total cases processed: {case_id}")
    print(f"Dataset location: {dataset_folder}")
    print(f"\nNext steps:")
    print(f"1. Set environment variable: export nnUNet_raw='{output_dir}'")
    print(f"2. Run: nnUNetv2_plan_and_preprocess -d {dataset_id} --verify_dataset_integrity")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare Atlas2 dataset for nn-UNet training"
    )
    parser.add_argument(
        "--atlas2_dir",
        type=str,
        required=True,
        help="Path to Atlas2 dataset root directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to nnUNet_raw directory"
    )
    parser.add_argument(
        "--dataset_id",
        type=int,
        default=1,
        help="Dataset ID for nn-UNet (default: 1)"
    )
    
    args = parser.parse_args()
    
    prepare_atlas2_dataset(
        args.atlas2_dir,
        args.output_dir,
        args.dataset_id
    )


if __name__ == "__main__":
    main()
