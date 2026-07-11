#!/usr/bin/env bash
# Submit one CARC Slurm GPU job per BIDS subject for the stroke-inpainting
# pipeline. Each subject job processes all matching sessions/T1w scans.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

INPUT_ROOT="${ARC_ROOT:-/project2/ajoshi_1183/data/ARC}"
OUTPUT_DIR="${STROKE_INPAINT_OUTPUT_DIR:-}"
PIPELINE_SCRIPT="$SCRIPT_DIR/run_subjectwise_stroke_inpainting.py"
JOB_SCRIPT="$SCRIPT_DIR/python3gpu.job"
LOG_DIR=""
MODALITY="T1w"
DRY_RUN=0
declare -a SBATCH_ARGS=()
declare -a PIPELINE_ARGS=()

usage() {
    cat <<'EOF'
Usage: submit_subjectwise_carc.sh [launcher options] [-- pipeline options]

Discover BIDS scans under INPUT_ROOT and submit one GPU job per sub-* folder.
All scans belonging to a subject are passed to the existing subjectwise
stroke-delineation and diffusion-inpainting pipeline. A subject is skipped
when OUTPUT_DIR/sub-* already exists, including partial or failed output.

Launcher options:
  --input-root PATH       BIDS input root
                          (default: $ARC_ROOT or /project2/ajoshi_1183/data/ARC)
  --output-dir PATH       Shared output root
                          (default: INPUT_ROOT/derivatives/stroke_inpainting)
  --log-dir PATH          Slurm log directory (default: OUTPUT_DIR/slurm_logs)
  --modality NAME         BIDS suffix to discover (default: T1w)
  --pipeline-script PATH  Pipeline Python script
  --job-script PATH       CARC sbatch script (default: python3gpu.job)
  --sbatch-arg ARG        Extra sbatch argument; repeat as needed
  --dry-run               Print submissions without calling sbatch
  -h, --help              Show this help

Arguments after -- are forwarded unchanged to the Python pipeline. For
example, use -- --overwrite or -- --inpainting-steps 250.

Examples:
  ./submit_subjectwise_carc.sh --dry-run
  ./submit_subjectwise_carc.sh --sbatch-arg=--mail-type=END,FAIL
  ./submit_subjectwise_carc.sh -- --disable-tta --inpainting-steps 500
EOF
}

die() {
    echo "ERROR: $*" >&2
    exit 1
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

while (($#)); do
    case "$1" in
        --input-root)
            (($# >= 2)) || die "$1 requires a path"
            INPUT_ROOT="$2"
            shift 2
            ;;
        --input-root=*)
            INPUT_ROOT="${1#*=}"
            shift
            ;;
        --output-dir)
            (($# >= 2)) || die "$1 requires a path"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-dir=*)
            OUTPUT_DIR="${1#*=}"
            shift
            ;;
        --log-dir)
            (($# >= 2)) || die "$1 requires a path"
            LOG_DIR="$2"
            shift 2
            ;;
        --log-dir=*)
            LOG_DIR="${1#*=}"
            shift
            ;;
        --modality)
            (($# >= 2)) || die "$1 requires a name"
            MODALITY="$2"
            shift 2
            ;;
        --modality=*)
            MODALITY="${1#*=}"
            shift
            ;;
        --pipeline-script)
            (($# >= 2)) || die "$1 requires a path"
            PIPELINE_SCRIPT="$2"
            shift 2
            ;;
        --pipeline-script=*)
            PIPELINE_SCRIPT="${1#*=}"
            shift
            ;;
        --job-script)
            (($# >= 2)) || die "$1 requires a path"
            JOB_SCRIPT="$2"
            shift 2
            ;;
        --job-script=*)
            JOB_SCRIPT="${1#*=}"
            shift
            ;;
        --sbatch-arg)
            (($# >= 2)) || die "$1 requires an argument"
            SBATCH_ARGS+=("$2")
            shift 2
            ;;
        --sbatch-arg=*)
            SBATCH_ARGS+=("${1#*=}")
            shift
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            PIPELINE_ARGS=("$@")
            break
            ;;
        *)
            die "unknown launcher option: $1 (put pipeline options after --)"
            ;;
    esac
done

[[ -d "$INPUT_ROOT" ]] || die "input root does not exist: $INPUT_ROOT"
[[ -f "$PIPELINE_SCRIPT" ]] || die "pipeline script does not exist: $PIPELINE_SCRIPT"
[[ -f "$JOB_SCRIPT" ]] || die "CARC job script does not exist: $JOB_SCRIPT"
[[ "$MODALITY" != */* ]] || die "modality must not contain a slash: $MODALITY"

INPUT_ROOT="$(cd -- "$INPUT_ROOT" && pwd -P)"
PIPELINE_SCRIPT="$(realpath -- "$PIPELINE_SCRIPT")"
JOB_SCRIPT="$(realpath -- "$JOB_SCRIPT")"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$INPUT_ROOT/derivatives/stroke_inpainting"
fi
OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"
if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="$OUTPUT_DIR/slurm_logs"
fi
LOG_DIR="$(realpath -m -- "$LOG_DIR")"

if ((DRY_RUN == 0)); then
    command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable; run this script on a CARC login node"
    mkdir -p "$LOG_DIR"
fi

for argument in "${PIPELINE_ARGS[@]}"; do
    case "$argument" in
        --input-root|--input-root=*|--output-dir|--output-dir=*|--modality|--modality=*|\
        --case|--case=*|--case-glob|--case-glob=*|--limit|--limit=*)
            die "$argument is controlled by the launcher and cannot be forwarded after --"
            ;;
    esac
done

mapfile -d '' SUBJECT_DIRS < <(
    find "$INPUT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'sub-*' -print0 | sort -z
)
((${#SUBJECT_DIRS[@]} > 0)) || die "no sub-* directories found under $INPUT_ROOT"

declare -A CASE_OWNERS=()
submitted=0
scans_found=0
subjects_with_scans=0
subjects_skipped=0
scans_skipped=0

for subject_dir in "${SUBJECT_DIRS[@]}"; do
    subject="$(basename -- "$subject_dir")"
    mapfile -d '' scans < <(
        find "$subject_dir" -mindepth 3 -maxdepth 3 -type f \
            -path "*/anat/*_${MODALITY}.nii.gz" -print0 | sort -z
    )
    ((${#scans[@]} > 0)) || continue
    ((subjects_with_scans += 1))

    subject_output_dir="$OUTPUT_DIR/$subject"
    if [[ -d "$subject_output_dir" ]]; then
        printf 'Skipping %-25s output directory exists: %s\n' \
            "$subject" "$subject_output_dir"
        ((subjects_skipped += 1))
        ((scans_skipped += ${#scans[@]}))
        continue
    fi

    declare -a case_args=()
    for scan in "${scans[@]}"; do
        case_id="$(basename -- "$scan")"
        case_id="${case_id%.nii.gz}"
        if [[ -n "${CASE_OWNERS[$case_id]:-}" ]]; then
            die "case ID $case_id occurs in both ${CASE_OWNERS[$case_id]} and $subject"
        fi
        CASE_OWNERS[$case_id]="$subject"
        case_args+=(--case "$case_id")
        ((scans_found += 1))
    done

    # Slurm job names accept a limited character set; normalize unusual BIDS IDs.
    safe_subject="${subject//[^[:alnum:]_.-]/_}"
    job_name="inpaint_${safe_subject}"
    command=(
        python "$PIPELINE_SCRIPT"
        --input-root "$INPUT_ROOT"
        --output-dir "$OUTPUT_DIR"
        --modality "$MODALITY"
        --nnunet-device cuda
        --inpainting-device cuda
        --disable-progress-bar
        "${case_args[@]}"
        "${PIPELINE_ARGS[@]}"
    )
    submission=(
        sbatch
        --parsable
        "${SBATCH_ARGS[@]}"
        --job-name "$job_name"
        --output "$LOG_DIR/%x-%j.out"
        --error "$LOG_DIR/%x-%j.err"
        "$JOB_SCRIPT"
        "${command[@]}"
    )

    if ((DRY_RUN)); then
        print_command "${submission[@]}"
    else
        job_id="$("${submission[@]}")"
        printf 'Submitted %-24s job=%s scans=%d\n' "$subject" "$job_id" "${#scans[@]}"
    fi
    ((submitted += 1))
done

((subjects_with_scans > 0)) || die "no *_${MODALITY}.nii.gz scans found in sub-*/ses-*/anat"

if ((DRY_RUN)); then
    printf 'Dry run: %d subject job(s), %d scan(s); skipped %d existing subject(s), %d scan(s)\n' \
        "$submitted" "$scans_found" "$subjects_skipped" "$scans_skipped"
else
    printf 'Submitted %d subject job(s) for %d scan(s)\n' "$submitted" "$scans_found"
    printf 'Skipped %d existing subject output(s), covering %d scan(s)\n' \
        "$subjects_skipped" "$scans_skipped"
    printf 'Slurm logs: %s\n' "$LOG_DIR"
fi
