#!/usr/bin/env bash
# Submit one restartable CARC GPU job per ARC subject for p_lesion/entropy output.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}"
MNI_ROOT="${STROKE_INPAINT_OUTPUT_DIR:-${ARC_ROOT:-/project2/ajoshi_1183/data/ARC}/derivatives/stroke_inpainting}"
OUTPUT_DIR="${LESION_UNCERTAINTY_OUTPUT_DIR:-}"
LOG_DIR=""
WORKER="$SCRIPT_DIR/run_arc_lesion_uncertainty.py"
JOB_SCRIPT="$SCRIPT_DIR/lesion_uncertainty_gpu.job"
DRY_RUN=0
declare -a SBATCH_ARGS=()
declare -a WORKER_ARGS=()

usage() {
    cat <<'EOF'
Usage: submit_arc_lesion_uncertainty_carc.sh [launcher options] [-- worker options]

Discover completed ARC 1 mm MNI T1 derivatives and submit one CARC GPU job per
subject. Each worker exports voxelwise lesion probability, normalized
predictive entropy, a p>=0.5 mask, scalar AQ features, and QC PNGs.

Launcher options:
  --mni-root PATH       Existing stroke_inpainting derivative root
  --output-dir PATH     Output root (default: sibling lesion_uncertainty)
  --log-dir PATH        Slurm logs (default: OUTPUT_DIR/slurm_logs)
  --worker PATH         Python worker script
  --job-script PATH     CARC Slurm job wrapper
  --sbatch-arg ARG      Extra sbatch option; repeatable
  --dry-run             Print commands without submitting
  -h, --help            Show this help

Options after -- are forwarded to run_arc_lesion_uncertainty.py. Examples:
  ./submit_arc_lesion_uncertainty_carc.sh --dry-run
  ./submit_arc_lesion_uncertainty_carc.sh -- --folds 0 --disable-tta
  ./submit_arc_lesion_uncertainty_carc.sh --sbatch-arg=--mail-type=END,FAIL
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
        --mni-root)
            (($# >= 2)) || die "$1 requires a path"
            MNI_ROOT="$2"
            shift 2
            ;;
        --mni-root=*) MNI_ROOT="${1#*=}"; shift ;;
        --output-dir)
            (($# >= 2)) || die "$1 requires a path"
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
        --log-dir)
            (($# >= 2)) || die "$1 requires a path"
            LOG_DIR="$2"
            shift 2
            ;;
        --log-dir=*) LOG_DIR="${1#*=}"; shift ;;
        --worker)
            (($# >= 2)) || die "$1 requires a path"
            WORKER="$2"
            shift 2
            ;;
        --worker=*) WORKER="${1#*=}"; shift ;;
        --job-script)
            (($# >= 2)) || die "$1 requires a path"
            JOB_SCRIPT="$2"
            shift 2
            ;;
        --job-script=*) JOB_SCRIPT="${1#*=}"; shift ;;
        --sbatch-arg)
            (($# >= 2)) || die "$1 requires an argument"
            SBATCH_ARGS+=("$2")
            shift 2
            ;;
        --sbatch-arg=*) SBATCH_ARGS+=("${1#*=}"); shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --)
            shift
            WORKER_ARGS=("$@")
            break
            ;;
        *) die "unknown launcher option: $1 (put worker options after --)" ;;
    esac
done

[[ -d "$MNI_ROOT" ]] || die "MNI derivative root does not exist: $MNI_ROOT"
[[ -f "$WORKER" ]] || die "worker does not exist: $WORKER"
[[ -f "$JOB_SCRIPT" ]] || die "job script does not exist: $JOB_SCRIPT"

MNI_ROOT="$(cd -- "$MNI_ROOT" && pwd -P)"
WORKER="$(realpath -- "$WORKER")"
JOB_SCRIPT="$(realpath -- "$JOB_SCRIPT")"
if [[ -z "$OUTPUT_DIR" ]]; then
    OUTPUT_DIR="$(dirname -- "$MNI_ROOT")/lesion_uncertainty"
fi
OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"
if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="$OUTPUT_DIR/slurm_logs"
fi
LOG_DIR="$(realpath -m -- "$LOG_DIR")"

for argument in "${WORKER_ARGS[@]}"; do
    case "$argument" in
        --mni-root|--mni-root=*|--output-dir|--output-dir=*|--subject|--subject=*|\
        --case|--case=*|--limit|--limit=*|--device|--device=*)
            die "$argument is controlled by the launcher"
            ;;
    esac
done

if ((DRY_RUN == 0)); then
    command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable; run on a CARC login node"
    mkdir -p "$LOG_DIR"
fi

mapfile -d '' SUBJECT_DIRS < <(
    find "$MNI_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'sub-*' -print0 | sort -z
)
((${#SUBJECT_DIRS[@]} > 0)) || die "no sub-* directories found under $MNI_ROOT"

submitted=0
skipped_complete=0
skipped_no_input=0
scans=0
declare -A SEEN_SUBJECTS=()
for subject_dir in "${SUBJECT_DIRS[@]}"; do
    top_level_name="$(basename -- "$subject_dir")"
    # Current derivatives use sub-*/ses-*/case/.  The legacy local sample uses
    # case/ directly; taking the first BIDS entity supports both layouts.
    subject="${top_level_name%%_*}"
    if [[ -n "${SEEN_SUBJECTS[$subject]:-}" ]]; then
        continue
    fi
    SEEN_SUBJECTS[$subject]=1
    scan_count=0
    while IFS= read -r -d '' candidate; do
        parent="$(basename -- "$(dirname -- "$candidate")")"
        candidate_case="$(basename -- "$candidate")"
        candidate_case="${candidate_case%_mni_1mm.nii.gz}"
        if [[ "$(basename -- "$candidate")" == "${parent}_mni_1mm.nii.gz" && \
              "$candidate_case" == "${subject}_"* ]]; then
            ((scan_count += 1))
        fi
    done < <(find "$MNI_ROOT" -type f -name "${subject}_*_mni_1mm.nii.gz" -print0)
    if ((scan_count == 0)); then
        printf 'Skipping %-24s no full-head MNI T1 derivative\n' "$subject"
        ((skipped_no_input += 1))
        continue
    fi
    if [[ -f "$OUTPUT_DIR/$subject/lesion_uncertainty_complete" ]]; then
        printf 'Skipping %-24s completion marker exists\n' "$subject"
        ((skipped_complete += 1))
        continue
    fi

    safe_subject="${subject//[^[:alnum:]_.-]/_}"
    command=(
        python "$WORKER"
        --mni-root "$MNI_ROOT"
        --output-dir "$OUTPUT_DIR"
        --subject "$subject"
        --device cuda
        --disable-progress-bar
        "${WORKER_ARGS[@]}"
    )
    submission=(
        sbatch
        --parsable
        "${SBATCH_ARGS[@]}"
        --job-name "pstroke_${safe_subject}"
        --output "$LOG_DIR/%x-%j.out"
        --error "$LOG_DIR/%x-%j.err"
        "$JOB_SCRIPT"
        "${command[@]}"
    )
    if ((DRY_RUN)); then
        print_command "${submission[@]}"
    else
        job_id="$("${submission[@]}")"
        printf 'Submitted %-23s job=%s scans=%d\n' "$subject" "$job_id" "$scan_count"
    fi
    ((submitted += 1))
    ((scans += scan_count))
done

if ((DRY_RUN)); then
    printf 'Dry run: %d job(s), %d scan(s); skipped %d complete and %d without input\n' \
        "$submitted" "$scans" "$skipped_complete" "$skipped_no_input"
else
    printf 'Submitted %d subject job(s) covering %d scan(s)\n' "$submitted" "$scans"
    printf 'Skipped %d complete subject(s) and %d without input; logs: %s\n' \
        "$skipped_complete" "$skipped_no_input" "$LOG_DIR"
fi
