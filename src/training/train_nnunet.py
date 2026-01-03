"""
Training script for nn-UNet on Atlas2 stroke dataset.

This script provides a wrapper around nn-UNet's training functionality
with common configurations for stroke lesion segmentation.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def check_nnunet_installation():
    """Check if nn-UNet is properly installed."""
    try:
        import nnunetv2
        # Some nnunetv2 package builds may not expose a __version__ attribute.
        # Try to read the attribute first, then fall back to importlib.metadata.
        version = getattr(nnunetv2, "__version__", None)
        if version is None:
            try:
                from importlib import metadata
            except Exception:
                try:
                    import importlib_metadata as metadata
                except Exception:
                    metadata = None

            if metadata is not None:
                try:
                    version = metadata.version("nnunetv2")
                except Exception:
                    version = "unknown"
            else:
                version = "unknown"

        print(f"✓ nn-UNet v2 is installed (version: {version})")
        return True
    except ImportError:
        print("✗ nn-UNet v2 is not installed")
        print("  Please install it with: pip install nnunetv2")
        return False


def check_environment_variables():
    """Check if required environment variables are set."""
    required_vars = ['nnUNet_raw', 'nnUNet_preprocessed', 'nnUNet_results']
    missing_vars = []
    
    for var in required_vars:
        if var not in os.environ:
            missing_vars.append(var)
        else:
            print(f"✓ {var} = {os.environ[var]}")
    
    if missing_vars:
        print(f"\n✗ Missing environment variables: {', '.join(missing_vars)}")
        print("\nPlease set them, for example:")
        print("  export nnUNet_raw='/path/to/nnUNet_raw'")
        print("  export nnUNet_preprocessed='/path/to/nnUNet_preprocessed'")
        print("  export nnUNet_results='/path/to/nnUNet_results'")
        return False
    
    return True


def plan_and_preprocess(dataset_id, verify_integrity=True):
    """
    Run nn-UNet's plan and preprocess step.
    
    Args:
        dataset_id: Dataset ID
        verify_integrity: Whether to verify dataset integrity
    """
    print("\n" + "="*60)
    print("Step 1: Planning and Preprocessing")
    print("="*60)
    
    cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d", str(dataset_id),
        "--verify_dataset_integrity" if verify_integrity else ""
    ]
    
    # Remove empty strings from command
    cmd = [c for c in cmd if c]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("✗ Planning and preprocessing failed!")
        sys.exit(1)
    
    print("✓ Planning and preprocessing completed successfully")


def train_model(dataset_id, configuration="3d_fullres", fold=0, trainer="nnUNetTrainer",
                num_epochs=None, continue_training=False):
    """
    Train nn-UNet model.
    
    Args:
        dataset_id: Dataset ID
        configuration: Model configuration (2d, 3d_fullres, 3d_lowres, 3d_cascade_fullres)
        fold: Fold number (0-4 for 5-fold cross-validation, or 'all' to train all folds)
        trainer: Trainer class name
        num_epochs: Number of epochs (None for default)
        continue_training: Whether to continue from existing checkpoint
    """
    print("\n" + "="*60)
    print(f"Step 2: Training (Configuration: {configuration}, Fold: {fold})")
    print("="*60)
    
    cmd = [
        "nnUNetv2_train",
        str(dataset_id),
        configuration,
        str(fold),
        "-tr", trainer
    ]
    
    if num_epochs is not None:
        cmd.extend(["--num_epochs", str(num_epochs)])
    
    if continue_training:
        cmd.append("-c")
    
    print(f"Running: {' '.join(cmd)}")
    print(f"\nNote: Training may take several hours or days depending on:")
    print("  - Dataset size")
    print("  - Hardware (GPU/CPU)")
    print("  - Number of epochs")
    print("\nYou can monitor progress in the terminal output.")
    print("Training logs and checkpoints will be saved to $nnUNet_results")
    print()
    
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("✗ Training failed!")
        sys.exit(1)
    
    print("✓ Training completed successfully")


def find_best_configuration(dataset_id):
    """
    Find the best configuration by running inference on validation set.
    
    Args:
        dataset_id: Dataset ID
    """
    print("\n" + "="*60)
    print("Step 3: Finding Best Configuration")
    print("="*60)
    
    cmd = [
        "nnUNetv2_find_best_configuration",
        str(dataset_id),
        "-c", "3d_fullres"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    
    if result.returncode != 0:
        print("✗ Finding best configuration failed!")
        sys.exit(1)
    
    print("✓ Best configuration determined successfully")


def main():
    parser = argparse.ArgumentParser(
        description="Train nn-UNet on Atlas2 stroke dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full training pipeline with preprocessing
  python train_nnunet.py --dataset_id 1 --do_preprocessing --fold 0

  # Train specific fold only (if preprocessing already done)
  python train_nnunet.py --dataset_id 1 --fold 0

  # Train all folds
  python train_nnunet.py --dataset_id 1 --fold all

  # Continue training from checkpoint
  python train_nnunet.py --dataset_id 1 --fold 0 --continue_training

  # Train with custom number of epochs
  python train_nnunet.py --dataset_id 1 --fold 0 --num_epochs 500
        """
    )
    
    parser.add_argument(
        "--dataset_id",
        type=int,
        required=True,
        help="Dataset ID (e.g., 1 for Dataset001_Atlas2)"
    )
    
    parser.add_argument(
        "--configuration",
        type=str,
        default="3d_fullres",
        choices=["2d", "3d_fullres", "3d_lowres", "3d_cascade_fullres"],
        help="Model configuration (default: 3d_fullres)"
    )
    
    parser.add_argument(
        "--fold",
        default="0",
        help="Fold number (0-4) or 'all' to train all folds (default: 0)"
    )
    
    parser.add_argument(
        "--trainer",
        type=str,
        default="nnUNetTrainer",
        help="Trainer class name (default: nnUNetTrainer)"
    )
    
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=None,
        help="Number of epochs (default: None, uses nn-UNet default)"
    )
    
    parser.add_argument(
        "--do_preprocessing",
        action="store_true",
        help="Run planning and preprocessing before training"
    )
    
    parser.add_argument(
        "--continue_training",
        action="store_true",
        help="Continue training from existing checkpoint"
    )
    
    parser.add_argument(
        "--find_best_config",
        action="store_true",
        help="Find best configuration after training"
    )
    
    parser.add_argument(
        "--skip_checks",
        action="store_true",
        help="Skip environment and installation checks"
    )

    parser.add_argument(
        "--nnunet_raw",
        type=str,
        default=None,
        help="Path to nnUNet_raw (overrides nnUNet_raw env var)"
    )

    parser.add_argument(
        "--nnunet_preprocessed",
        type=str,
        default=None,
        help="Path to nnUNet_preprocessed (overrides nnUNet_preprocessed env var)"
    )

    parser.add_argument(
        "--nnunet_results",
        type=str,
        default=None,
        help="Path to nnUNet_results (overrides nnUNet_results env var)"
    )
    
    args = parser.parse_args()

    # Allow overriding environment variables from CLI arguments
    if args.nnunet_raw:
        os.environ['nnUNet_raw'] = args.nnunet_raw
    if args.nnunet_preprocessed:
        os.environ['nnUNet_preprocessed'] = args.nnunet_preprocessed
    if args.nnunet_results:
        os.environ['nnUNet_results'] = args.nnunet_results
    
    # Perform checks
    if not args.skip_checks:
        print("Checking installation and environment...")
        if not check_nnunet_installation():
            sys.exit(1)
        if not check_environment_variables():
            sys.exit(1)
        print()
    
    # Run preprocessing if requested
    if args.do_preprocessing:
        plan_and_preprocess(args.dataset_id)
    
    # Train model
    if args.fold.lower() == "all":
        # Train all folds
        for fold in range(5):
            train_model(
                args.dataset_id,
                args.configuration,
                fold,
                args.trainer,
                args.num_epochs,
                args.continue_training
            )
    else:
        # Train single fold
        train_model(
            args.dataset_id,
            args.configuration,
            int(args.fold),
            args.trainer,
            args.num_epochs,
            args.continue_training
        )
    
    # Find best configuration if requested
    if args.find_best_config:
        find_best_configuration(args.dataset_id)
    
    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60)
    print(f"\nModel saved to: {os.environ.get('nnUNet_results', 'nnUNet_results')}")
    print("\nNext steps:")
    print("  1. Run inference on test data using src/inference/predict.py")
    print("  2. Evaluate model performance")


if __name__ == "__main__":
    main()
