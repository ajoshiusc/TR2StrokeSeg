"""
Evaluation utilities for comparing predictions with ground truth.

This script calculates common segmentation metrics:
- Dice Score
- Intersection over Union (IoU)
- Sensitivity (Recall)
- Specificity
- Precision
"""

import os
import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
from scipy.spatial.distance import directed_hausdorff


def load_nifti(file_path):
    """Load NIfTI file and return data array."""
    img = nib.load(file_path)
    return img.get_fdata()


def dice_score(pred, gt):
    """
    Calculate Dice Score.
    
    Args:
        pred: Predicted segmentation (binary)
        gt: Ground truth segmentation (binary)
    
    Returns:
        Dice score (float)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    intersection = np.sum(pred & gt)
    if np.sum(pred) + np.sum(gt) == 0:
        return 1.0  # Both empty
    
    return 2.0 * intersection / (np.sum(pred) + np.sum(gt))


def iou_score(pred, gt):
    """
    Calculate Intersection over Union (IoU).
    
    Args:
        pred: Predicted segmentation (binary)
        gt: Ground truth segmentation (binary)
    
    Returns:
        IoU score (float)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    intersection = np.sum(pred & gt)
    union = np.sum(pred | gt)
    
    if union == 0:
        return 1.0  # Both empty
    
    return intersection / union


def sensitivity_specificity(pred, gt):
    """
    Calculate sensitivity (recall) and specificity.
    
    Args:
        pred: Predicted segmentation (binary)
        gt: Ground truth segmentation (binary)
    
    Returns:
        Tuple of (sensitivity, specificity)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    tp = np.sum(pred & gt)
    tn = np.sum(~pred & ~gt)
    fp = np.sum(pred & ~gt)
    fn = np.sum(~pred & gt)
    
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    
    return sensitivity, specificity


def precision_score(pred, gt):
    """
    Calculate precision (positive predictive value).
    
    Args:
        pred: Predicted segmentation (binary)
        gt: Ground truth segmentation (binary)
    
    Returns:
        Precision score (float)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    tp = np.sum(pred & gt)
    fp = np.sum(pred & ~gt)
    
    if (tp + fp) == 0:
        return 0.0
    
    return tp / (tp + fp)


def hausdorff_distance(pred, gt):
    """
    Calculate Hausdorff distance.
    
    Args:
        pred: Predicted segmentation (binary)
        gt: Ground truth segmentation (binary)
    
    Returns:
        Hausdorff distance (float)
    """
    pred = pred.astype(bool)
    gt = gt.astype(bool)
    
    # Get surface points
    pred_points = np.argwhere(pred)
    gt_points = np.argwhere(gt)
    
    if len(pred_points) == 0 or len(gt_points) == 0:
        return float('inf')
    
    # Calculate directed Hausdorff distances
    hd1 = directed_hausdorff(pred_points, gt_points)[0]
    hd2 = directed_hausdorff(gt_points, pred_points)[0]
    
    return max(hd1, hd2)


def evaluate_prediction(pred_file, gt_file):
    """
    Evaluate a single prediction against ground truth.
    
    Args:
        pred_file: Path to prediction file
        gt_file: Path to ground truth file
    
    Returns:
        Dictionary of metrics
    """
    # Load images
    pred = load_nifti(pred_file)
    gt = load_nifti(gt_file)
    
    # Binarize (threshold at 0.5)
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)
    
    # Calculate metrics
    dice = dice_score(pred, gt)
    iou = iou_score(pred, gt)
    sens, spec = sensitivity_specificity(pred, gt)
    prec = precision_score(pred, gt)
    
    # Hausdorff distance (can be slow for large volumes)
    try:
        hd = hausdorff_distance(pred, gt)
    except Exception as e:
        print(f"  Warning: Could not calculate Hausdorff distance: {e}")
        hd = float('nan')
    
    return {
        'dice': dice,
        'iou': iou,
        'sensitivity': sens,
        'specificity': spec,
        'precision': prec,
        'hausdorff': hd
    }


def evaluate_dataset(pred_dir, gt_dir, output_file=None):
    """
    Evaluate all predictions in a directory.
    
    Args:
        pred_dir: Directory containing predictions
        gt_dir: Directory containing ground truth
        output_file: Optional CSV file to save results
    """
    pred_dir = Path(pred_dir)
    gt_dir = Path(gt_dir)
    
    pred_files = sorted(list(pred_dir.glob("*.nii.gz")))
    
    if len(pred_files) == 0:
        print(f"No prediction files found in {pred_dir}")
        return
    
    print(f"Evaluating {len(pred_files)} predictions...")
    print()
    
    all_metrics = []
    
    for pred_file in pred_files:
        # Try to find corresponding ground truth
        gt_file = gt_dir / pred_file.name
        
        # Try alternative naming if not found
        if not gt_file.exists():
            # Remove _prediction suffix if present
            alt_name = pred_file.name.replace('_prediction', '')
            gt_file = gt_dir / alt_name
        
        if not gt_file.exists():
            print(f"Warning: Ground truth not found for {pred_file.name}")
            continue
        
        print(f"Evaluating: {pred_file.name}")
        metrics = evaluate_prediction(pred_file, gt_file)
        metrics['filename'] = pred_file.name
        all_metrics.append(metrics)
        
        print(f"  Dice: {metrics['dice']:.4f}")
        print(f"  IoU: {metrics['iou']:.4f}")
        print(f"  Sensitivity: {metrics['sensitivity']:.4f}")
        print(f"  Specificity: {metrics['specificity']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print()
    
    # Calculate mean metrics
    if all_metrics:
        print("="*60)
        print("Mean Metrics:")
        print("="*60)
        mean_dice = np.mean([m['dice'] for m in all_metrics])
        mean_iou = np.mean([m['iou'] for m in all_metrics])
        mean_sens = np.mean([m['sensitivity'] for m in all_metrics])
        mean_spec = np.mean([m['specificity'] for m in all_metrics])
        mean_prec = np.mean([m['precision'] for m in all_metrics])
        
        print(f"Dice Score: {mean_dice:.4f} ± {np.std([m['dice'] for m in all_metrics]):.4f}")
        print(f"IoU: {mean_iou:.4f} ± {np.std([m['iou'] for m in all_metrics]):.4f}")
        print(f"Sensitivity: {mean_sens:.4f} ± {np.std([m['sensitivity'] for m in all_metrics]):.4f}")
        print(f"Specificity: {mean_spec:.4f} ± {np.std([m['specificity'] for m in all_metrics]):.4f}")
        print(f"Precision: {mean_prec:.4f} ± {np.std([m['precision'] for m in all_metrics]):.4f}")
        
        # Save to CSV if requested
        if output_file:
            import csv
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['filename', 'dice', 'iou', 
                                                       'sensitivity', 'specificity', 'precision', 'hausdorff'])
                writer.writeheader()
                writer.writerows(all_metrics)
            print(f"\nResults saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate stroke lesion segmentation predictions"
    )
    
    parser.add_argument(
        "--pred_dir",
        type=str,
        required=True,
        help="Directory containing predicted segmentations"
    )
    
    parser.add_argument(
        "--gt_dir",
        type=str,
        required=True,
        help="Directory containing ground truth segmentations"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for results (optional)"
    )
    
    args = parser.parse_args()
    
    evaluate_dataset(args.pred_dir, args.gt_dir, args.output)


if __name__ == "__main__":
    main()
