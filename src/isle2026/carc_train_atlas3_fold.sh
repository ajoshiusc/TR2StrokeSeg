#!/usr/bin/env bash
set -euo pipefail

# CARC/local helper for one ATLAS3 raw nnUNet fold.
# Submit one fold:
#   sbatch --export=ALL,FOLD=0 python3gpu.job src/isle2026/carc_train_atlas3_fold.sh
#
# Submit all folds:
#   for f in 0 1 2 3 4; do sbatch --export=ALL,FOLD=$f python3gpu.job src/isle2026/carc_train_atlas3_fold.sh; done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
DATASET_ID="${DATASET_ID:-326}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
FOLD="${FOLD:-0}"
TRAINER="${TRAINER:-nnUNetTrainer}"
PLANS="${PLANS:-nnUNetPlans}"
DEVICE="${DEVICE:-cuda}"
NNUNET_N_PROC_DA="${NNUNET_N_PROC_DA:-8}"
CONTINUE_TRAINING="${CONTINUE_TRAINING:-0}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -z "${WORK_DIR:-}" ]]; then
    if [[ -d /deneb_disk/ATLAS3 ]]; then
        WORK_DIR="/deneb_disk/ATLAS3/nnunet_isle2026"
    else
        WORK_DIR="/project2/ajoshi_1183/data/ISLE2026_ATLAS3_nnunet"
    fi
fi

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"
export nnUNet_n_proc_DA="${NNUNET_N_PROC_DA}"

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "WORK_DIR=${WORK_DIR}"
echo "DATASET_ID=${DATASET_ID}"
echo "CONFIGURATION=${CONFIGURATION}"
echo "FOLD=${FOLD}"
echo "TRAINER=${TRAINER}"
echo "PLANS=${PLANS}"
echo "DEVICE=${DEVICE}"
echo "nnUNet_n_proc_DA=${nnUNet_n_proc_DA}"
echo "CONTINUE_TRAINING=${CONTINUE_TRAINING}"
echo "PYTHON_BIN=${PYTHON_BIN}"

train_args=(
    -m src.isle2026.atlas3_raw_nnunet train
    --work-dir "${WORK_DIR}"
    --dataset-id "${DATASET_ID}"
    --configuration "${CONFIGURATION}"
    --fold "${FOLD}"
    --trainer "${TRAINER}"
    --plans "${PLANS}"
    --device "${DEVICE}"
)

if [[ "${CONTINUE_TRAINING}" == "1" || "${CONTINUE_TRAINING}" == "true" ]]; then
    train_args+=(--continue-training)
fi

"${PYTHON_BIN}" "${train_args[@]}"
