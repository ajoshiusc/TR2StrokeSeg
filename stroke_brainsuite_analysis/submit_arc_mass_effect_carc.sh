#!/usr/bin/env bash
# Submit one restartable CPU job per ARC subject for deformation-proxy output.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PROJECT_DIR="${PROJECT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd -P)}"
ARC_ROOT="${ARC_ROOT:-/project2/ajoshi_1183/data/ARC}"
INPAINTED_ROOT=""
RAW_ROOT=""
INPAINTING_ROOT=""
OUTPUT_DIR=""
LOG_DIR=""
WORKER="$SCRIPT_DIR/extract_arc_mass_effect.py"
JOB_SCRIPT="$SCRIPT_DIR/mass_effect_cpu.job"
DRY_RUN=0
declare -a SBATCH_ARGS=()
declare -a WORKER_ARGS=()

usage() {
    cat <<'EOF'
Usage: submit_arc_mass_effect_carc.sh [launcher options] [-- worker options]

Discover completed inpainted ARC SVReg maps and submit one restartable CARC
CPU job per subject. The worker stores atlas-space deformation proxy maps,
shell summaries, registration-sensitivity QC, and a subject QC PNG.

Launcher options:
  --arc-root PATH          ARC root (default: /project2/ajoshi_1183/data/ARC)
  --inpainted-root PATH    Inpainted BrainSuite/SVReg derivative root
  --raw-root PATH          Direct non-inpainted BrainSuite derivative root
  --inpainting-root PATH   Stroke-inpainting derivative root
  --output-dir PATH        Output root (default: derivatives/lesion_mass_effect)
  --log-dir PATH           Slurm logs (default: OUTPUT_DIR/slurm_logs)
  --worker PATH            Python extraction script
  --job-script PATH        CARC Slurm job wrapper
  --sbatch-arg ARG         Extra sbatch option; repeatable
  --dry-run                Print commands without submitting
  -h, --help               Show this help

Options after -- are forwarded to extract_arc_mass_effect.py. Examples:
  ./submit_arc_mass_effect_carc.sh --dry-run
  ./submit_arc_mass_effect_carc.sh -- --no-raw-sensitivity
  ./submit_arc_mass_effect_carc.sh --sbatch-arg=--mail-type=END,FAIL
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
        --arc-root) (($# >= 2)) || die "$1 requires a path"; ARC_ROOT="$2"; shift 2 ;;
        --arc-root=*) ARC_ROOT="${1#*=}"; shift ;;
        --inpainted-root) (($# >= 2)) || die "$1 requires a path"; INPAINTED_ROOT="$2"; shift 2 ;;
        --inpainted-root=*) INPAINTED_ROOT="${1#*=}"; shift ;;
        --raw-root) (($# >= 2)) || die "$1 requires a path"; RAW_ROOT="$2"; shift 2 ;;
        --raw-root=*) RAW_ROOT="${1#*=}"; shift ;;
        --inpainting-root) (($# >= 2)) || die "$1 requires a path"; INPAINTING_ROOT="$2"; shift 2 ;;
        --inpainting-root=*) INPAINTING_ROOT="${1#*=}"; shift ;;
        --output-dir) (($# >= 2)) || die "$1 requires a path"; OUTPUT_DIR="$2"; shift 2 ;;
        --output-dir=*) OUTPUT_DIR="${1#*=}"; shift ;;
        --log-dir) (($# >= 2)) || die "$1 requires a path"; LOG_DIR="$2"; shift 2 ;;
        --log-dir=*) LOG_DIR="${1#*=}"; shift ;;
        --worker) (($# >= 2)) || die "$1 requires a path"; WORKER="$2"; shift 2 ;;
        --worker=*) WORKER="${1#*=}"; shift ;;
        --job-script) (($# >= 2)) || die "$1 requires a path"; JOB_SCRIPT="$2"; shift 2 ;;
        --job-script=*) JOB_SCRIPT="${1#*=}"; shift ;;
        --sbatch-arg) (($# >= 2)) || die "$1 requires an argument"; SBATCH_ARGS+=("$2"); shift 2 ;;
        --sbatch-arg=*) SBATCH_ARGS+=("${1#*=}"); shift ;;
        --dry-run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        --) shift; WORKER_ARGS=("$@"); break ;;
        *) die "unknown launcher option: $1 (put worker options after --)" ;;
    esac
done

INPAINTED_ROOT="${INPAINTED_ROOT:-$ARC_ROOT/derivatives/brainsuite_anatomical_bidsapp}"
RAW_ROOT="${RAW_ROOT:-$ARC_ROOT/derivatives/brainsuite_anatomical_raw}"
INPAINTING_ROOT="${INPAINTING_ROOT:-$ARC_ROOT/derivatives/stroke_inpainting}"
OUTPUT_DIR="${OUTPUT_DIR:-$ARC_ROOT/derivatives/lesion_mass_effect}"

[[ -d "$INPAINTED_ROOT" ]] || die "inpainted BrainSuite root does not exist: $INPAINTED_ROOT"
[[ -d "$INPAINTING_ROOT" ]] || die "inpainting root does not exist: $INPAINTING_ROOT"
[[ -f "$WORKER" ]] || die "worker does not exist: $WORKER"
[[ -f "$JOB_SCRIPT" ]] || die "job script does not exist: $JOB_SCRIPT"

ARC_ROOT="$(realpath -m -- "$ARC_ROOT")"
INPAINTED_ROOT="$(cd -- "$INPAINTED_ROOT" && pwd -P)"
RAW_ROOT="$(realpath -m -- "$RAW_ROOT")"
INPAINTING_ROOT="$(cd -- "$INPAINTING_ROOT" && pwd -P)"
OUTPUT_DIR="$(realpath -m -- "$OUTPUT_DIR")"
WORKER="$(realpath -- "$WORKER")"
JOB_SCRIPT="$(realpath -- "$JOB_SCRIPT")"
if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="$OUTPUT_DIR/slurm_logs"
fi
LOG_DIR="$(realpath -m -- "$LOG_DIR")"

for argument in "${WORKER_ARGS[@]}"; do
    case "$argument" in
        --arc-root|--arc-root=*|--inpainted-brainsuite-root|--inpainted-brainsuite-root=*|\
        --raw-brainsuite-root|--raw-brainsuite-root=*|--inpainting-root|--inpainting-root=*|\
        --output-dir|--output-dir=*|--subject|--subject=*|--case|--case=*|--limit|--limit=*)
            die "$argument is controlled by the launcher"
            ;;
    esac
done

if ((DRY_RUN == 0)); then
    command -v sbatch >/dev/null 2>&1 || die "sbatch is unavailable; run on a CARC login node"
    mkdir -p "$LOG_DIR"
fi

mapfile -d '' SUBJECT_DIRS < <(
    find "$INPAINTED_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'sub-*' -print0 | sort -z
)
((${#SUBJECT_DIRS[@]} > 0)) || die "no sub-* directories found under $INPAINTED_ROOT"

submitted=0
skipped_complete=0
skipped_no_map=0
for subject_dir in "${SUBJECT_DIRS[@]}"; do
    subject="$(basename -- "$subject_dir")"
    if ! find "$subject_dir" -type f -name '*_inpainted_mni_1mm.svreg.inv.map.nii.gz' -print -quit | grep -q .; then
        printf 'Skipping %-24s no completed inpainted inverse map\n' "$subject"
        ((skipped_no_map += 1))
        continue
    fi
    if [[ -f "$OUTPUT_DIR/$subject/mass_effect_complete" ]]; then
        printf 'Skipping %-24s completion marker exists\n' "$subject"
        ((skipped_complete += 1))
        continue
    fi

    safe_subject="${subject//[^[:alnum:]_.-]/_}"
    command=(
        python "$WORKER"
        --arc-root "$ARC_ROOT"
        --inpainted-brainsuite-root "$INPAINTED_ROOT"
        --raw-brainsuite-root "$RAW_ROOT"
        --inpainting-root "$INPAINTING_ROOT"
        --output-dir "$OUTPUT_DIR"
        --subject "$subject"
        "${WORKER_ARGS[@]}"
    )
    submission=(
        sbatch
        --parsable
        "${SBATCH_ARGS[@]}"
        --job-name "meffect_${safe_subject}"
        --output "$LOG_DIR/%x-%j.out"
        --error "$LOG_DIR/%x-%j.err"
        "$JOB_SCRIPT"
        "${command[@]}"
    )
    if ((DRY_RUN)); then
        print_command "${submission[@]}"
    else
        job_id="$("${submission[@]}")"
        printf 'Submitted %-23s job=%s\n' "$subject" "$job_id"
    fi
    ((submitted += 1))
done

if ((DRY_RUN)); then
    printf 'Dry run: %d job(s); skipped %d complete and %d without maps\n' \
        "$submitted" "$skipped_complete" "$skipped_no_map"
else
    printf 'Submitted %d subject job(s); logs: %s\n' "$submitted" "$LOG_DIR"
    printf 'Skipped %d complete subject(s) and %d without maps\n' \
        "$skipped_complete" "$skipped_no_map"
fi
