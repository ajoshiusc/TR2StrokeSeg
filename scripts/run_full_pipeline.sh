#!/bin/bash
# Complete workflow for training nn-UNet on Atlas2 and testing on another dataset
# This script demonstrates the full pipeline from data preparation to inference

set -e  # Exit on error

echo "=========================================="
echo "nn-UNet Training and Testing Pipeline"
echo "=========================================="
echo ""

# Configuration - MODIFY THESE PATHS
ATLAS2_DATA_DIR="/path/to/atlas2/dataset"
TEST_DATA_DIR="/path/to/test/dataset"
WORK_DIR="/path/to/working/directory"
DATASET_ID=1

# Setup working directories
export nnUNet_raw="$WORK_DIR/nnUNet_raw"
export nnUNet_preprocessed="$WORK_DIR/nnUNet_preprocessed"
export nnUNet_results="$WORK_DIR/nnUNet_results"

mkdir -p "$nnUNet_raw"
mkdir -p "$nnUNet_preprocessed"
mkdir -p "$nnUNet_results"

echo "Working directories:"
echo "  nnUNet_raw: $nnUNet_raw"
echo "  nnUNet_preprocessed: $nnUNet_preprocessed"
echo "  nnUNet_results: $nnUNet_results"
echo ""

# Step 1: Data Preparation
echo "=========================================="
echo "Step 1: Preparing Atlas2 Dataset"
echo "=========================================="
python src/data_preparation/prepare_atlas2.py \
    --atlas2_dir "$ATLAS2_DATA_DIR" \
    --output_dir "$nnUNet_raw" \
    --dataset_id $DATASET_ID

# Step 2: Planning and Preprocessing
echo ""
echo "=========================================="
echo "Step 2: Planning and Preprocessing"
echo "=========================================="
nnUNetv2_plan_and_preprocess -d $DATASET_ID --verify_dataset_integrity

# Step 3: Training
echo ""
echo "=========================================="
echo "Step 3: Training nn-UNet"
echo "=========================================="
echo "Training fold 0 (you can train all folds by changing --fold to 'all')"
python src/training/train_nnunet.py \
    --dataset_id $DATASET_ID \
    --fold 0 \
    --configuration 3d_fullres

# Optional: Train all folds for cross-validation
# Uncomment the following to train all folds:
# for fold in {0..4}; do
#     python src/training/train_nnunet.py \
#         --dataset_id $DATASET_ID \
#         --fold $fold \
#         --configuration 3d_fullres
# done

# Step 4: Inference on Test Dataset
echo ""
echo "=========================================="
echo "Step 4: Running Inference on Test Dataset"
echo "=========================================="
python src/inference/predict.py \
    --test_dir "$TEST_DATA_DIR" \
    --output_dir "$WORK_DIR/predictions" \
    --dataset_id $DATASET_ID \
    --configuration 3d_fullres \
    --folds 0

echo ""
echo "=========================================="
echo "Pipeline Completed Successfully!"
echo "=========================================="
echo ""
echo "Results:"
echo "  - Trained model: $nnUNet_results"
echo "  - Predictions: $WORK_DIR/predictions/final_predictions"
echo ""
echo "Next steps:"
echo "  1. Evaluate prediction quality"
echo "  2. Compare with ground truth if available"
echo "  3. Visualize results using ITK-SNAP or similar tools"
