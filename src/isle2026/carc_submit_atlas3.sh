#!/usr/bin/env bash
set -euo pipefail

# Thin submitter around the repository's python3gpu.job Slurm wrapper.
#
# Examples:
#   src/isle2026/carc_submit_atlas3.sh prepare-plan
#   src/isle2026/carc_submit_atlas3.sh train-fold 0
#   src/isle2026/carc_submit_atlas3.sh train-all

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
PYTHON3GPU_JOB="${PYTHON3GPU_JOB:-${PROJECT_DIR}/python3gpu.job}"
ACTION="${1:-}"

case "${ACTION}" in
    prepare-plan)
        sbatch "${PYTHON3GPU_JOB}" "${PROJECT_DIR}/src/isle2026/carc_prepare_plan_atlas3.sh"
        ;;
    train-fold)
        FOLD="${2:-${FOLD:-0}}"
        sbatch --export=ALL,FOLD="${FOLD}" "${PYTHON3GPU_JOB}" "${PROJECT_DIR}/src/isle2026/carc_train_atlas3_fold.sh"
        ;;
    train-all)
        for FOLD in 0 1 2 3 4; do
            sbatch --export=ALL,FOLD="${FOLD}" "${PYTHON3GPU_JOB}" "${PROJECT_DIR}/src/isle2026/carc_train_atlas3_fold.sh"
        done
        ;;
    *)
        echo "Usage: $0 {prepare-plan|train-fold [fold]|train-all}"
        echo
        echo "Optional env overrides:"
        echo "  PROJECT_DIR=${PROJECT_DIR}"
        echo "  ATLAS3_ROOT=/deneb_disk/ATLAS3 or /project2/ajoshi_1183/data/ATLAS3"
        echo "  WORK_DIR=/deneb_disk/ATLAS3/nnunet_isle2026 or /project2/ajoshi_1183/data/ISLE2026_ATLAS3_nnunet"
        echo "  DATASET_ID=326 CONFIGURATION=3d_fullres PYTHON_BIN=python3"
        exit 2
        ;;
esac
