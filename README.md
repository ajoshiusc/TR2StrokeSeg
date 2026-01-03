# TR2StrokeSeg

Stroke lesion segmentation using nn-UNet trained on Atlas2 dataset with cross-dataset evaluation capabilities.

## Overview

This repository provides a complete pipeline for:
1. **Training** nn-UNet on the Atlas2 stroke lesion dataset
2. **Testing** the trained model on other stroke datasets (cross-dataset evaluation)
3. **Inference** on new unseen data

The implementation uses [nn-UNet v2](https://github.com/MIC-DKFZ/nnUNet), a self-configuring deep learning framework for medical image segmentation.

## Features

- 🔧 Automated data preparation for Atlas2 dataset
- 🚀 Easy-to-use training scripts with sensible defaults
- 🎯 Cross-dataset inference capabilities
- 📊 Support for 5-fold cross-validation
- 🔄 Batch processing for multiple test subjects
- 📝 Comprehensive documentation and examples

## Installation

### Prerequisites

- Python 3.9 or higher
- CUDA-capable GPU (recommended for training)
- 16GB+ RAM (32GB+ recommended)

### Setup

1. Clone this repository:
```bash
git clone https://github.com/ajoshiusc/TR2StrokeSeg.git
cd TR2StrokeSeg
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up nn-UNet environment variables:
```bash
# Edit paths in scripts/setup_environment.sh
vim scripts/setup_environment.sh

# Source the script
source scripts/setup_environment.sh
```

Or manually set:
```bash
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

## Quick Start

### 1. Prepare Atlas2 Dataset

Download the Atlas2 dataset and prepare it for nn-UNet:

```bash
python src/data_preparation/prepare_atlas2.py \
    --atlas2_dir /path/to/atlas2/dataset \
    --output_dir $nnUNet_raw \
    --dataset_id 1
```

**Expected Atlas2 directory structure:**
```
atlas2/
├── Training/
│   ├── R001/
│   │   ├── t1w.nii.gz
│   │   └── lesion_mask.nii.gz
│   ├── R002/
│   │   ├── t1w.nii.gz
│   │   └── lesion_mask.nii.gz
│   └── ...
```

### 2. Train nn-UNet

#### Option A: Full training pipeline (with preprocessing)
```bash
python src/training/train_nnunet.py \
    --dataset_id 1 \
    --do_preprocessing \
    --fold 0
```

#### Option B: Train all folds for cross-validation
```bash
python src/training/train_nnunet.py \
    --dataset_id 1 \
    --do_preprocessing \
    --fold all
```

#### Option C: Continue training from checkpoint
```bash
python src/training/train_nnunet.py \
    --dataset_id 1 \
    --fold 0 \
    --continue_training
```

### 3. Run Inference on Test Dataset

Apply the trained model to a new dataset:

```bash
python src/inference/predict.py \
    --test_dir /path/to/test/images \
    --output_dir /path/to/output \
    --dataset_id 1 \
    --folds 0,1,2,3,4
```

For single fold inference:
```bash
python src/inference/predict.py \
    --test_dir /path/to/test/images \
    --output_dir /path/to/output \
    --dataset_id 1 \
    --folds 0
```

## Complete Pipeline Example

Run the full pipeline from data preparation to inference:

```bash
# Edit configuration in the script
vim scripts/run_full_pipeline.sh

# Run the pipeline
bash scripts/run_full_pipeline.sh
```

## Directory Structure

```
TR2StrokeSeg/
├── src/
│   ├── data_preparation/
│   │   └── prepare_atlas2.py      # Atlas2 dataset preparation
│   ├── training/
│   │   └── train_nnunet.py        # Training script
│   └── inference/
│       └── predict.py              # Inference script
├── configs/
│   └── atlas2_config.yaml          # Configuration file
├── scripts/
│   ├── setup_environment.sh        # Environment setup
│   └── run_full_pipeline.sh        # Complete pipeline script
├── requirements.txt                # Python dependencies
└── README.md                       # This file
```

## Usage Details

### Data Preparation

The `prepare_atlas2.py` script:
- Converts Atlas2 dataset to nn-UNet format
- Handles flexible file naming conventions
- Creates required `dataset.json` file
- Validates data integrity

### Training

The `train_nnunet.py` script:
- Wraps nn-UNet training with convenient options
- Supports single or multi-fold training
- Allows custom number of epochs
- Enables continuing from checkpoints
- Performs environment checks

**Training options:**
- `--dataset_id`: Dataset ID (e.g., 1)
- `--configuration`: Model configuration (2d, 3d_fullres, etc.)
- `--fold`: Fold number (0-4) or 'all'
- `--num_epochs`: Number of training epochs
- `--do_preprocessing`: Run preprocessing before training
- `--continue_training`: Continue from checkpoint

### Inference

The `predict.py` script:
- Prepares test data automatically
- Supports ensemble prediction from multiple folds
- Organizes predictions with meaningful names
- Works with any test dataset structure

**Inference options:**
- `--test_dir`: Directory with test images
- `--output_dir`: Output directory
- `--dataset_id`: Dataset ID of trained model
- `--folds`: Folds to use for ensemble (e.g., "0,1,2,3,4")
- `--checkpoint`: Checkpoint to use (default: checkpoint_final.pth)

## Cross-Dataset Evaluation

To evaluate the model trained on Atlas2 on a different stroke dataset:

1. Ensure test images are in NIfTI format (`.nii.gz`)
2. Run inference:
```bash
python src/inference/predict.py \
    --test_dir /path/to/other/dataset \
    --output_dir /path/to/results \
    --dataset_id 1
```

3. Predictions will be saved to `{output_dir}/final_predictions/`

## Tips and Best Practices

### Training
- **GPU Memory**: 3D full resolution requires significant GPU memory (11GB+). Use 2d or 3d_lowres for smaller GPUs.
- **Training Time**: Training can take 1-3 days depending on dataset size and hardware.
- **Cross-Validation**: Train all 5 folds for robust model evaluation.
- **Monitoring**: Check `$nnUNet_results` for training logs and progress.

### Inference
- **Ensemble**: Using multiple folds (--folds 0,1,2,3,4) generally improves prediction quality.
- **Batch Processing**: The script automatically processes all images in the test directory.
- **Format**: Input images should be in NIfTI format (.nii.gz).

### Data
- **Preprocessing**: nn-UNet automatically handles normalization and resampling.
- **Naming**: The scripts handle various naming conventions for image and mask files.
- **Validation**: Use `--verify_dataset_integrity` flag during preprocessing to check data quality.

## Troubleshooting

### Environment Variables Not Set
```bash
# Make sure to set and export environment variables
export nnUNet_raw="/path/to/nnUNet_raw"
export nnUNet_preprocessed="/path/to/nnUNet_preprocessed"
export nnUNet_results="/path/to/nnUNet_results"
```

### CUDA Out of Memory
- Use 2d configuration instead of 3d_fullres
- Reduce batch size (requires modifying nn-UNet configuration)
- Use a GPU with more memory

### Missing Dependencies
```bash
pip install -r requirements.txt
```

## References

- **nn-UNet**: Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
- **Atlas2**: Liew, S. L., et al. (2018). A large, open source dataset of stroke anatomical brain images and manual lesion segmentations. Scientific data, 5(1), 1-11.

## License

This project is licensed under the GNU General Public License v2.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{tr2strokeseg,
  title = {TR2StrokeSeg: Stroke Lesion Segmentation using nn-UNet},
  author = {Joshi, Anand},
  year = {2026},
  url = {https://github.com/ajoshiusc/TR2StrokeSeg}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Contact

For questions or issues, please open an issue on GitHub.
