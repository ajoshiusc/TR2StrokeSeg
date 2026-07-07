#!/usr/bin/env bash
set -euo pipefail

# CARC/local helper for ATLAS3 raw nnUNet dataset creation and preprocessing.
# Submit with:
#   sbatch python3gpu.job src/isle2026/carc_prepare_plan_atlas3.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd -P)}"
DATASET_ID="${DATASET_ID:-326}"
CONFIGURATION="${CONFIGURATION:-3d_fullres}"
NUM_PROCESSES="${NUM_PROCESSES:-16}"
LINK_MODE="${LINK_MODE:-symlink}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

if [[ -z "${ATLAS3_ROOT:-}" ]]; then
    if [[ -d /deneb_disk/ATLAS3 ]]; then
        ATLAS3_ROOT="/deneb_disk/ATLAS3"
    elif [[ -d /project2/ajoshi_1183/data/ATLAS3 ]]; then
        ATLAS3_ROOT="/project2/ajoshi_1183/data/ATLAS3"
    else
        echo "ATLAS3_ROOT was not set and no default ATLAS3 data root exists." >&2
        echo "Checked /deneb_disk/ATLAS3 and /project2/ajoshi_1183/data/ATLAS3." >&2
        exit 1
    fi
fi

if [[ -z "${WORK_DIR:-}" ]]; then
    case "${ATLAS3_ROOT%/}" in
        /deneb_disk/ATLAS3)
            WORK_DIR="/deneb_disk/ATLAS3/nnunet_isle2026"
            ;;
        /project2/ajoshi_1183/data/ATLAS3)
            WORK_DIR="/project2/ajoshi_1183/data/ISLE2026_ATLAS3_nnunet"
            ;;
        *)
            WORK_DIR="${ATLAS3_ROOT%/}/nnunet_isle2026"
            ;;
    esac
fi

cd "${PROJECT_DIR}"
export PYTHONPATH="${PROJECT_DIR}:${PYTHONPATH:-}"

echo "PROJECT_DIR=${PROJECT_DIR}"
echo "ATLAS3_ROOT=${ATLAS3_ROOT}"
echo "WORK_DIR=${WORK_DIR}"
echo "DATASET_ID=${DATASET_ID}"
echo "CONFIGURATION=${CONFIGURATION}"
echo "PYTHON_BIN=${PYTHON_BIN}"

"${PYTHON_BIN}" -m src.isle2026.atlas3_raw_nnunet prepare \
    --atlas3-root "${ATLAS3_ROOT}" \
    --work-dir "${WORK_DIR}" \
    --dataset-id "${DATASET_ID}" \
    --link-mode "${LINK_MODE}" \
    --overwrite

"${PYTHON_BIN}" -m src.isle2026.atlas3_raw_nnunet plan \
    --work-dir "${WORK_DIR}" \
    --dataset-id "${DATASET_ID}" \
    --configuration "${CONFIGURATION}" \
    --num-processes "${NUM_PROCESSES}"
