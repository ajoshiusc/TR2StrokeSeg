#!/usr/bin/env bash
# Submit one CARC Slurm job per subject for the full BrainSuite anatomical
# pipeline (BSE + cortical extraction + SVReg) on completed inpainted T1 scans.

set -euo pipefail

SCRIPT_PATH="$(realpath -- "${BASH_SOURCE[0]}")"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

print_command() {
    printf '%q ' "$@"
    printf '\n'
}

run_subject_worker() {
    (($# >= 4)) || die "internal worker invocation is incomplete"
    local inpaint_root="$1"
    local output_root="$2"
    local subject="$3"
    local brainsuite_home="$4"
    shift 4
    if [[ "${1:-}" == "--" ]]; then
        shift
    fi
    local -a brainsuite_args=("$@")

    local anatomical_pipeline="$brainsuite_home/bin/brainsuite_anatomical_pipeline.sh"
    [[ -x "$anatomical_pipeline" ]] || \
        die "BrainSuite anatomical pipeline is not executable: $anatomical_pipeline"

    local subject_input="$inpaint_root/$subject"
    local subject_output="$output_root/$subject"
    [[ -d "$subject_input" ]] || die "inpainted subject directory is missing: $subject_input"

    # This also prevents two independently launched jobs from processing the
    # same subject. A partial directory is intentionally considered claimed.
    if ! mkdir "$subject_output" 2>/dev/null; then
        echo "Skipping $subject: output directory already exists: $subject_output"
        return 0
    fi

    local -a brains=()
    local brain
    mapfile -d '' brains < <(
        find "$subject_input" -mindepth 3 -maxdepth 3 -type f \
            -name '*_brain_inpainted_mni_1mm.nii.gz' -print0 | sort -z
    )
    ((${#brains[@]} > 0)) || die "no completed inpainted brains found for $subject"

    local -a complete_brains=()
    for brain in "${brains[@]}"; do
        local candidate_dir candidate_name candidate_id candidate_full_t1
        candidate_dir="$(dirname -- "$brain")"
        candidate_name="$(basename -- "$brain")"
        candidate_id="${candidate_name%_brain_inpainted_mni_1mm.nii.gz}"
        candidate_full_t1="$candidate_dir/${candidate_id}_inpainted_mni_1mm.nii.gz"
        if [[ -f "$candidate_full_t1" ]]; then
            complete_brains+=("$brain")
        else
            echo "Skipping $candidate_id: full-head inpainted T1 is missing: $candidate_full_t1" >&2
        fi
    done
    ((${#complete_brains[@]} > 0)) || die "no complete inpainted scan pairs found for $subject"

    export BrainSuiteDir="$brainsuite_home/"
    export BrainSuiteBin="$brainsuite_home/bin/"
    export SVRegDir="$brainsuite_home/svreg/"
    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

    local completed=0
    for brain in "${complete_brains[@]}"; do
        local source_dir brain_name case_id session full_t1 case_output staged_t1
        source_dir="$(dirname -- "$brain")"
        brain_name="$(basename -- "$brain")"
        case_id="${brain_name%_brain_inpainted_mni_1mm.nii.gz}"
        session="$(basename -- "$(dirname -- "$source_dir")")"
        full_t1="$source_dir/${case_id}_inpainted_mni_1mm.nii.gz"

        case_output="$subject_output/$session/$case_id"
        mkdir -p "$case_output"
        staged_t1="$case_output/${case_id}_inpainted_mni_1mm.nii.gz"
        ln -s "$full_t1" "$staged_t1"

        echo "[$((completed + 1))/${#complete_brains[@]}] $case_id"
        echo "Input:  $full_t1"
        echo "Output: $case_output"
        "$anatomical_pipeline" "$staged_t1" "${brainsuite_args[@]}"
        touch "$case_output/brainsuite_anatomical_complete"
        ((completed += 1))
    done

    touch "$subject_output/brainsuite_anatomical_complete"
    printf 'Completed BrainSuite anatomical processing for %s (%d scan(s))\n' \
        "$subject" "$completed"
}

if [[ "${1:-}" == "--worker" ]]; then
    shift
    run_subject_worker "$@"
    exit $?
fi

INPAINT_ROOT="${STROKE_INPAINT_OUTPUT_DIR:-${ARC_ROOT:-/project2/ajoshi_1183/data/ARC}/derivatives/stroke_inpainting}"
OUTPUT_ROOT="${BRAINSUITE_ANATOMICAL_OUTPUT_DIR:-}"
LOG_DIR=""
BRAINSUITE_HOME_VALUE="${BRAINSUITE_HOME:-/project2/ajoshi_27/BrainSuite23a}"
ACCOUNT="ajoshi_1183"
CPUS="8"
MEMORY="32G"
WALLTIME="2:00:00"
DRY_RUN=0
declare -a SBATCH_ARGS=()
declare -a BRAINSUITE_ARGS=()

usage() {
    cat <<'EOF'
Usage: submit_brainsuite_anatomical_carc.sh [options] [-- BrainSuite/SVReg args]

Submit one CARC CPU job per subject for BrainSuite's full anatomical pipeline,
including BSE, cortical surface extraction, and SVReg. A scan is eligible only
when both of these inpainting outputs exist:

  *_brain_inpainted_mni_1mm.nii.gz  (completion gate)
  *_inpainted_mni_1mm.nii.gz        (full-head BrainSuite input)

A subject is skipped when OUTPUT_ROOT/sub-* already exists, including partial
or failed anatomical output. BrainSuite results are written under:

  OUTPUT_ROOT/sub-*/ses-*/<case-id>/

Options:
  --inpaint-root PATH     Stroke-inpainting output root
  --output-root PATH      BrainSuite output root
                         (default: sibling brainsuite_anatomical_inpainted)
  --log-dir PATH          Slurm logs (default: OUTPUT_ROOT/slurm_logs)
  --brainsuite-home PATH  BrainSuite installation
                         (default: $BRAINSUITE_HOME or CARC BrainSuite23a)
  --account NAME          Slurm account (default: ajoshi_1183)
  --cpus N                CPUs per subject job (default: 8)
  --mem SIZE              Memory per job (default: 32G)
  --time HH:MM:SS         Walltime per subject job (default: 2:00:00)
  --sbatch-arg ARG        Extra sbatch argument; repeat as needed
  --dry-run               Print submissions without calling sbatch
  -h, --help              Show this help

Arguments after -- are passed to brainsuite_anatomical_pipeline.sh after the
input image. For example, SVReg flags can be supplied as: -- -k -t

Examples:
  ./submit_brainsuite_anatomical_carc.sh --dry-run
  ./submit_brainsuite_anatomical_carc.sh --sbatch-arg=--mail-type=END,FAIL
  ./submit_brainsuite_anatomical_carc.sh --time 47:00:00 -- -k -t
EOF
}

while (($#)); do
    case "$1" in
        --inpaint-root)
            (($# >= 2)) || die "$1 requires a path"
            INPAINT_ROOT="$2"
            shift 2
            ;;
        --inpaint-root=*) INPAINT_ROOT="${1#*=}"; shift ;;
        --output-root)
            (($# >= 2)) || die "$1 requires a path"
            OUTPUT_ROOT="$2"
            shift 2
            ;;
        --output-root=*) OUTPUT_ROOT="${1#*=}"; shift ;;
        --log-dir)
            (($# >= 2)) || die "$1 requires a path"
            LOG_DIR="$2"
            shift 2
            ;;
        --log-dir=*) LOG_DIR="${1#*=}"; shift ;;
        --brainsuite-home)
            (($# >= 2)) || die "$1 requires a path"
            BRAINSUITE_HOME_VALUE="$2"
            shift 2
            ;;
        --brainsuite-home=*) BRAINSUITE_HOME_VALUE="${1#*=}"; shift ;;
        --account)
            (($# >= 2)) || die "$1 requires a name"
            ACCOUNT="$2"
            shift 2
            ;;
        --account=*) ACCOUNT="${1#*=}"; shift ;;
        --cpus)
            (($# >= 2)) || die "$1 requires a number"
            CPUS="$2"
            shift 2
            ;;
        --cpus=*) CPUS="${1#*=}"; shift ;;
        --mem)
            (($# >= 2)) || die "$1 requires a size"
            MEMORY="$2"
            shift 2
            ;;
        --mem=*) MEMORY="${1#*=}"; shift ;;
        --time)
            (($# >= 2)) || die "$1 requires a duration"
            WALLTIME="$2"
            shift 2
            ;;
        --time=*) WALLTIME="${1#*=}"; shift ;;
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
            BRAINSUITE_ARGS=("$@")
            break
            ;;
        *) die "unknown option: $1 (put BrainSuite/SVReg arguments after --)" ;;
    esac
done

[[ -d "$INPAINT_ROOT" ]] || die "inpainting output root does not exist: $INPAINT_ROOT"
[[ "$CPUS" =~ ^[1-9][0-9]*$ ]] || die "--cpus must be a positive integer"

INPAINT_ROOT="$(cd -- "$INPAINT_ROOT" && pwd -P)"
BRAINSUITE_HOME_VALUE="$(realpath -m -- "$BRAINSUITE_HOME_VALUE")"
if [[ -z "$OUTPUT_ROOT" ]]; then
    OUTPUT_ROOT="$(dirname -- "$INPAINT_ROOT")/brainsuite_anatomical_inpainted"
fi
OUTPUT_ROOT="$(realpath -m -- "$OUTPUT_ROOT")"
if [[ -z "$LOG_DIR" ]]; then
    LOG_DIR="$OUTPUT_ROOT/slurm_logs"
fi
LOG_DIR="$(realpath -m -- "$LOG_DIR")"

if ((DRY_RUN == 0)); then
    command -v sbatch >/dev/null 2>&1 || \
        die "sbatch is unavailable; run this script on a CARC login node"
    [[ -x "$BRAINSUITE_HOME_VALUE/bin/brainsuite_anatomical_pipeline.sh" ]] || \
        die "BrainSuite anatomical pipeline is unavailable under $BRAINSUITE_HOME_VALUE"
    mkdir -p "$LOG_DIR"
fi

mapfile -d '' subject_dirs < <(
    find "$INPAINT_ROOT" -mindepth 1 -maxdepth 1 -type d -name 'sub-*' -print0 | sort -z
)
((${#subject_dirs[@]} > 0)) || die "no sub-* directories found under $INPAINT_ROOT"

eligible=0
submitted=0
skipped_existing=0
skipped_incomplete=0
scans_submitted=0

for subject_dir in "${subject_dirs[@]}"; do
    subject="$(basename -- "$subject_dir")"
    mapfile -d '' brains < <(
        find "$subject_dir" -mindepth 3 -maxdepth 3 -type f \
            -name '*_brain_inpainted_mni_1mm.nii.gz' -print0 | sort -z
    )
    if ((${#brains[@]} == 0)); then
        printf 'Skipping %-25s no inpainted brain output\n' "$subject"
        ((skipped_incomplete += 1))
        continue
    fi

    complete_scan_count=0
    for brain in "${brains[@]}"; do
        source_dir="$(dirname -- "$brain")"
        brain_name="$(basename -- "$brain")"
        case_id="${brain_name%_brain_inpainted_mni_1mm.nii.gz}"
        if [[ -f "$source_dir/${case_id}_inpainted_mni_1mm.nii.gz" ]]; then
            ((complete_scan_count += 1))
        fi
    done
    if ((complete_scan_count == 0)); then
        printf 'Skipping %-25s no complete brain/full-head inpainted pair\n' "$subject"
        ((skipped_incomplete += 1))
        continue
    fi
    ((eligible += 1))

    subject_output="$OUTPUT_ROOT/$subject"
    if [[ -d "$subject_output" ]]; then
        printf 'Skipping %-25s output directory exists: %s\n' "$subject" "$subject_output"
        ((skipped_existing += 1))
        continue
    fi

    safe_subject="${subject//[^[:alnum:]_.-]/_}"
    submission=(
        sbatch
        --parsable
        --nodes=1
        --ntasks=1
        --cpus-per-task="$CPUS"
        --mem="$MEMORY"
        --time="$WALLTIME"
        --account="$ACCOUNT"
        "${SBATCH_ARGS[@]}"
        --job-name="bsanat_${safe_subject}"
        --output="$LOG_DIR/%x-%j.out"
        --error="$LOG_DIR/%x-%j.err"
        "$SCRIPT_PATH"
        --worker "$INPAINT_ROOT" "$OUTPUT_ROOT" "$subject" "$BRAINSUITE_HOME_VALUE"
        -- "${BRAINSUITE_ARGS[@]}"
    )

    if ((DRY_RUN)); then
        print_command "${submission[@]}"
    else
        job_id="$("${submission[@]}")"
        printf 'Submitted %-24s job=%s scans=%d\n' "$subject" "$job_id" "$complete_scan_count"
    fi
    ((submitted += 1))
    ((scans_submitted += complete_scan_count))
done

((eligible > 0)) || die "no completed inpainted brain outputs were found"

if ((DRY_RUN)); then
    printf 'Dry run: %d subject job(s), %d scan(s); skipped %d existing and %d incomplete subject(s)\n' \
        "$submitted" "$scans_submitted" "$skipped_existing" "$skipped_incomplete"
else
    printf 'Submitted %d BrainSuite subject job(s) for %d scan(s)\n' \
        "$submitted" "$scans_submitted"
    printf 'Skipped %d existing and %d incomplete subject(s)\n' \
        "$skipped_existing" "$skipped_incomplete"
    printf 'Slurm logs: %s\n' "$LOG_DIR"
fi
