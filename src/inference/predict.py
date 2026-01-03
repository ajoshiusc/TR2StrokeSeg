"""
Inference script for applying trained nn-UNet model to new datasets.

This script allows you to:
1. Apply a trained model to a different dataset (cross-dataset evaluation)
2. Batch process multiple subjects
3. Convert results back to original format
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
import shutil
import json


def prepare_test_data(input_dir, output_dir, modality_suffix="_0000"):
    """
    Prepare test data in nn-UNet format.
    
    Args:
        input_dir: Directory containing test images (*.nii.gz files)
        output_dir: Output directory for formatted test data
        modality_suffix: Suffix for modality (default: _0000 for single channel)
    
    Returns:
        List of case identifiers
    """
    print("\n" + "="*60)
    print("Step 1: Preparing Test Data")
    print("="*60)
    
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find all .nii.gz files
    nii_files = sorted(list(input_path.glob("**/*.nii.gz")))
    
    if len(nii_files) == 0:
        print(f"✗ No .nii.gz files found in {input_dir}")
        sys.exit(1)
    
    print(f"Found {len(nii_files)} NIfTI files")
    
    case_ids = []
    for idx, nii_file in enumerate(nii_files):
        # Skip mask/label files
        if any(x in nii_file.name.lower() for x in ['mask', 'label', 'seg']):
            print(f"  Skipping {nii_file.name} (appears to be a mask)")
            continue
        
        # Create case identifier
        case_id = f"case_{idx:04d}"
        case_ids.append(case_id)
        
        # Copy file with nn-UNet naming convention
        output_name = f"{case_id}{modality_suffix}.nii.gz"
        shutil.copy2(nii_file, output_path / output_name)
        print(f"  Prepared: {nii_file.name} -> {output_name}")
    
    print(f"✓ Prepared {len(case_ids)} cases for inference")
    return case_ids


def run_inference(input_dir, output_dir, model_dir, dataset_id, configuration="3d_fullres",
                  trainer="nnUNetTrainer", folds="0,1,2,3,4", checkpoint="checkpoint_final.pth"):
    """
    Run nn-UNet inference on test data.
    
    Args:
        input_dir: Directory containing prepared test images
        output_dir: Output directory for predictions
        model_dir: Path to trained model (nnUNet_results directory)
        dataset_id: Dataset ID
        configuration: Model configuration
        trainer: Trainer class name
        folds: Comma-separated fold numbers to use for ensemble
        checkpoint: Checkpoint name
    """
    print("\n" + "="*60)
    print("Step 2: Running Inference")
    print("="*60)
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Set environment variable if model_dir provided
    if model_dir:
        os.environ['nnUNet_results'] = str(model_dir)
        print(f"Using model from: {model_dir}")
    
    dataset_name = f"Dataset{dataset_id:03d}_Atlas2"
    
    cmd = [
        "nnUNetv2_predict",
        "-i", str(input_dir),
        "-o", str(output_dir),
        "-d", str(dataset_id),
        "-c", configuration,
        "-tr", trainer,
        "-f", folds,
        "-chk", checkpoint
    ]
    
    print(f"Running: {' '.join(cmd)}")
    print()
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("✗ Inference failed!")
        sys.exit(1)
    
    print("✓ Inference completed successfully")


def organize_predictions(prediction_dir, output_dir, case_mapping=None):
    """
    Organize predictions with original filenames.
    
    Args:
        prediction_dir: Directory containing predictions
        output_dir: Output directory for organized predictions
        case_mapping: Dictionary mapping case_ids to original names
    """
    print("\n" + "="*60)
    print("Step 3: Organizing Predictions")
    print("="*60)
    
    pred_path = Path(prediction_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    pred_files = sorted(list(pred_path.glob("*.nii.gz")))
    
    for pred_file in pred_files:
        if case_mapping and pred_file.stem in case_mapping:
            # Use original filename
            new_name = case_mapping[pred_file.stem] + "_prediction.nii.gz"
        else:
            # Keep prediction filename
            new_name = pred_file.name
        
        shutil.copy2(pred_file, output_path / new_name)
        print(f"  {pred_file.name} -> {new_name}")
    
    print(f"✓ Organized {len(pred_files)} prediction files")


def predict_on_dataset(test_dataset_dir, output_dir, model_dir, dataset_id,
                      configuration="3d_fullres", trainer="nnUNetTrainer",
                      folds="0,1,2,3,4", checkpoint="checkpoint_final.pth"):
    """
    Complete prediction pipeline on a test dataset.
    
    Args:
        test_dataset_dir: Directory containing test images
        output_dir: Output directory for all results
        model_dir: Path to trained model
        dataset_id: Dataset ID of trained model
        configuration: Model configuration
        trainer: Trainer class name
        folds: Folds to use for ensemble
        checkpoint: Checkpoint name
    """
    # Create working directories
    work_dir = Path(output_dir) / "work"
    prepared_dir = work_dir / "prepared_images"
    predictions_dir = work_dir / "predictions"
    final_dir = Path(output_dir) / "final_predictions"
    
    # Prepare test data
    case_ids = prepare_test_data(test_dataset_dir, prepared_dir)
    
    # Run inference
    run_inference(
        prepared_dir,
        predictions_dir,
        model_dir,
        dataset_id,
        configuration,
        trainer,
        folds,
        checkpoint
    )
    
    # Organize predictions
    organize_predictions(predictions_dir, final_dir)
    
    print("\n" + "="*60)
    print("Prediction pipeline completed successfully!")
    print("="*60)
    print(f"\nPredictions saved to: {final_dir}")
    print(f"Number of processed cases: {len(case_ids)}")


def main():
    parser = argparse.ArgumentParser(
        description="Run inference with trained nn-UNet model on new dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on test dataset using trained model
  python predict.py \\
    --test_dir /path/to/test/images \\
    --output_dir /path/to/output \\
    --model_dir /path/to/nnUNet_results \\
    --dataset_id 1

  # Use specific configuration and folds
  python predict.py \\
    --test_dir /path/to/test/images \\
    --output_dir /path/to/output \\
    --dataset_id 1 \\
    --configuration 3d_fullres \\
    --folds 0,1,2,3,4

  # Use single fold
  python predict.py \\
    --test_dir /path/to/test/images \\
    --output_dir /path/to/output \\
    --dataset_id 1 \\
    --folds 0
        """
    )
    
    parser.add_argument(
        "--test_dir",
        type=str,
        required=True,
        help="Directory containing test images (*.nii.gz files)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Output directory for predictions"
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="Path to nnUNet_results directory (if not set as env variable)"
    )
    
    parser.add_argument(
        "--dataset_id",
        type=int,
        required=True,
        help="Dataset ID of the trained model (e.g., 1 for Dataset001_Atlas2)"
    )
    
    parser.add_argument(
        "--configuration",
        type=str,
        default="3d_fullres",
        choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
        help="Model configuration (default: 3d_fullres)"
    )
    
    parser.add_argument(
        "--trainer",
        type=str,
        default="nnUNetTrainer",
        help="Trainer class name (default: nnUNetTrainer)"
    )
    
    parser.add_argument(
        "--folds",
        type=str,
        default="0,1,2,3,4",
        help="Comma-separated fold numbers for ensemble (default: 0,1,2,3,4)"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoint_final.pth",
        help="Checkpoint name (default: checkpoint_final.pth)"
    )
    
    args = parser.parse_args()
    
    # Check if nnUNet_results is set
    if args.model_dir is None and 'nnUNet_results' not in os.environ:
        print("✗ Error: Either --model_dir must be provided or nnUNet_results environment variable must be set")
        sys.exit(1)
    
    # Run prediction pipeline
    predict_on_dataset(
        args.test_dir,
        args.output_dir,
        args.model_dir,
        args.dataset_id,
        args.configuration,
        args.trainer,
        args.folds,
        args.checkpoint
    )


if __name__ == "__main__":
    main()
