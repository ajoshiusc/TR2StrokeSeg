# ATLAS3 Raw nnUNet Workflow

This folder contains a small nnUNet v2 workflow for training and inference on
ATLAS3 raw subject-space data. The code checks both standard data locations:

- local workstation: `/deneb_disk/ATLAS3`
- USC CARC: `/project2/ajoshi_1183/data/ATLAS3`

Set `ATLAS3_ROOT` or pass `--atlas3-root` only when the data is staged somewhere
else.

The default preparation is intentionally minimal:

- use `ATLAS3_Training_Raw`, not the MNI preprocessed tree
- create nnUNet `imagesTr` and `labelsTr` entries by symlink
- do not register or warp images
- let nnUNet plan/preprocess choose its internal patch size and spacing

External resampling is available with `--target-spacing`, but start without it.
nnUNet already resamples internally during preprocessing, and keeping the raw
dataset subject-space makes inference geometry cleaner.

## Prepare

```bash
python3 -m src.isle2026.atlas3_raw_nnunet prepare \
  --dataset-id 326 \
  --link-mode symlink
```

On the local workstation this creates:

```text
/deneb_disk/ATLAS3/nnunet_isle2026/
  nnUNet_raw/Dataset326_ATLAS3Raw/
    imagesTr/
    labelsTr/
    dataset.json
    case_mapping.tsv
  nnUNet_preprocessed/
  nnUNet_results/
```

For a smoke check without writing files:

```bash
python3 -m src.isle2026.atlas3_raw_nnunet prepare --dry-run
```

## Plan And Preprocess

```bash
python3 -m src.isle2026.atlas3_raw_nnunet plan \
  --dataset-id 326 \
  --configuration 3d_fullres \
  --num-processes 16
```

## Train

On a high-end GPU:

```bash
CUDA_VISIBLE_DEVICES=0 python3 -m src.isle2026.atlas3_raw_nnunet train \
  --dataset-id 326 \
  --configuration 3d_fullres \
  --fold 0 \
  --device cuda
```

For all five folds, run folds `0` through `4`. Use `--continue-training` to
resume a fold.

## CARC Slurm With `python3gpu.job`

The repo root has a `python3gpu.job` Slurm wrapper configured for USC CARC:

```text
account: ajoshi_1183
partition: gpu
gpu: 1
cpus: 16
time: 47:00:00
```

The helper scripts default to the repository path they are run from and choose
the first available ATLAS3/work directory. On CARC, that resolves to:

```bash
PROJECT_DIR=/project2/ajoshi_1183/Projects/TR2StrokeSeg
ATLAS3_ROOT=/project2/ajoshi_1183/data/ATLAS3
WORK_DIR=/project2/ajoshi_1183/data/ISLE2026_ATLAS3_nnunet
DATASET_ID=326
CONFIGURATION=3d_fullres
PYTHON_BIN=python3
```

On the local workstation, the defaults resolve to:

```bash
ATLAS3_ROOT=/deneb_disk/ATLAS3
WORK_DIR=/deneb_disk/ATLAS3/nnunet_isle2026
```

If ATLAS3 is staged elsewhere, override `ATLAS3_ROOT` when submitting or running
locally.

Prepare the raw nnUNet dataset and run nnUNet planning/preprocessing:

```bash
cd /project2/ajoshi_1183/Projects/TR2StrokeSeg
sbatch python3gpu.job src/isle2026/carc_prepare_plan_atlas3.sh
```

Submit one training fold:

```bash
cd /project2/ajoshi_1183/Projects/TR2StrokeSeg
sbatch --export=ALL,FOLD=0 python3gpu.job src/isle2026/carc_train_atlas3_fold.sh
```

Resume a fold from its latest checkpoint after a time limit:

```bash
cd /project2/ajoshi_1183/Projects/TR2StrokeSeg
sbatch --export=ALL,FOLD=0,CONTINUE_TRAINING=1 python3gpu.job src/isle2026/carc_train_atlas3_fold.sh
```

Submit all five folds:

```bash
cd /project2/ajoshi_1183/Projects/TR2StrokeSeg
for f in 0 1 2 3 4; do
  sbatch --export=ALL,FOLD=$f python3gpu.job src/isle2026/carc_train_atlas3_fold.sh
done
```

Or use the submit helper:

```bash
cd /project2/ajoshi_1183/Projects/TR2StrokeSeg
src/isle2026/carc_submit_atlas3.sh prepare-plan
src/isle2026/carc_submit_atlas3.sh train-fold 0
src/isle2026/carc_submit_atlas3.sh train-all
```

## Inference On Raw Images

Point `--input` at a raw ATLAS3-style folder or a single T1w NIfTI. The command
formats symlinked nnUNet input, runs prediction, and writes organized masks.

```bash
python3 -m src.isle2026.atlas3_raw_nnunet predict \
  --input /deneb_disk/ATLAS3/ATLAS3_Training_Raw/R046/sub-r046s004 \
  --output /tmp/atlas3_predictions \
  --dataset-id 326 \
  --configuration 3d_fullres \
  --folds 0 \
  --device cuda
```

If the model was trained with external `--target-spacing`, pass the same
spacing at prediction time and add `--restore-to-source-grid` to map masks back
onto the original T1 grid.
