#!/usr/bin/env bash
# Submit one CARC Slurm job per subject for the BrainSuite BIDS App pipeline
# on completed inpainted T1 scans.

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
    (($# >= 3)) || die "internal worker invocation is incomplete"
    local inpaint_root="$1"
    local output_root="$2"
    local subject="$3"
    shift 3
    if [[ "${1:-}" == "--" ]]; then
        shift
    fi
    local -a bidsapp_args=("$@")

    local bidsapp_image="/project2/ajoshi_27/Software/bids_brainsuite_v23a.simg"
    [[ -f "$bidsapp_image" ]] || \
        die "BrainSuite BIDS App image not found: $bidsapp_image"

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
        find "$subject_input" -mindepth 2 -maxdepth 3 -type l \
            -name '*_inpainted_mni_1mm.nii.gz' -print0 | sort -z
    )
    if ((${#brains[@]} == 0)); then
        # Check for regular files too if no symlinks found
        mapfile -d '' brains < <(
            find "$subject_input" -mindepth 2 -maxdepth 3 -type f \
                -name '*_inpainted_mni_1mm.nii.gz' -print0 | sort -z
        )
    fi
    ((${#brains[@]} > 0)) || die "no completed inpainted brains found for $subject"

    local -a complete_brains=()
    for brain in "${brains[@]}"; do
        # We assume if the T1 scan exists, it's ready for BrainSuite
        complete_brains+=("$brain")
    done
    ((${#complete_brains[@]} > 0)) || die "no complete inpainted scan pairs found for $subject"

    export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

    local completed=0
    for brain in "${complete_brains[@]}"; do
        local source_dir brain_name case_id session full_t1 case_output staged_t1
        source_dir="$(dirname -- "$brain")"
        brain_name="$(basename -- "$brain")"
        case_id="${brain_name%_inpainted_mni_1mm.nii.gz}"
        session="$(basename -- "$(dirname -- "$source_dir")")"
        full_t1="$brain"

        case_output="$subject_output/$session/$case_id"
        mkdir -p "$case_output"
        staged_t1="$case_output/${case_id}_inpainted_mni_1mm.nii.gz"
        ln -s "$full_t1" "$staged_t1"

        echo "[$((completed + 1))/${#complete_brains[@]}] $case_id"
        echo "Input:  $full_t1"
        echo "Output: $case_output"
        
        # Load apptainer and run via singularity
        module load apptainer || true
        # Need to mount the source directory of the symlink so singularity can resolve it
        local real_t1
        real_t1="$(readlink -f "$staged_t1")" || real_t1="$staged_t1"
        local source_mount
        source_mount="$(dirname -- "$real_t1")"
        
        singularity exec -B /tmp:/usr/share/applications/other \
            -B "${case_output}:${case_output}" \
            -B "${source_mount}:${source_mount}" \
            --nv "$bidsapp_image" brainsuite_anatomical_pipeline.sh "$staged_t1" "${bidsapp_args[@]}"
        
        touch "$case_output/brainsuite_anatomical_complete"
        ((completed += 1))
    done

    touch "$subject_output/brainsuite_anatomical_complete"
    printf 'Completed BrainSuite BIDS App anatomical processing for %s (%d scan(s))\n' \
        "$subject" "$completed"
}

if [[ "${1:-}" == "--worker" ]]; then
    shift
    run_subject_worker "$@"
    exit 0
fi

usage() {
    cat <<EOF
Usage: $0 [OPTIONS] INPAINT_DIR OUTPUT_DIR [BRAINSUITE_ARGS...]

Submit CARC jobs to run the BrainSuite BIDS App anatomical pipeline on inpainted
T1 subjects.

Arguments:
  INPAINT_DIR   Input directory containing sub-*/ses-*/... directories.
  OUTPUT_DIR    Output directory for the processed files.
  BRAINSUITE_ARGS Extra arguments passed directly to svreg.sh in the Apptainer.

Options:
  --account ACCT        Slurm account (default: from \$SLURM_JOB_ACCOUNT or none)
  --partition PART      Slurm partition (default: from \$SLURM_JOB_PARTITION,
                        or epyc-64)
  --time TIME           Slurm time limit (default: 04:00:00)
  --cpus CPUS           CPUs per subject job (default: 8)
  --mem MEM             Memory per subject job (default: 16G)
  --limit N             Only submit up to N jobs.

Examples:
  $0 . data/ARC/derivatives/inpainted data/ARC/derivatives/brainsuite_anatomical
EOF
}

slurm_account="${SLURM_JOB_ACCOUNT:-}"
slurm_partition="${SLURM_JOB_PARTITION:-epyc-64}"
slurm_time="04:00:00"
slurm_cpus=8
slurm_mem="16G"
job_limit=""

while [[ $# -gt 0 && "$1" == -* ]]; do
    case "$1" in
        --account) slurm_account="$2"; shift 2 ;;
        --partition) slurm_partition="$2"; shift 2 ;;
        --time) slurm_time="$2"; shift 2 ;;
        --cpus) slurm_cpus="$2"; shift 2 ;;
        --mem) slurm_mem="$2"; shift 2 ;;
        --limit) job_limit="$2"; shift 2 ;;
        --help) usage; exit 0 ;;
        --) shift; break ;;
        *) die "unknown option: $1" ;;
    esac
done

(($# >= 2)) || { usage >&2; exit 1; }

inpaint_dir="$1"
output_dir="$2"
shift 2

[[ -d "$inpaint_dir" ]] || die "inpaint directory not found: $inpaint_dir"
mkdir -p "$output_dir" || die "failed to create output directory: $output_dir"
inpaint_dir="$(realpath -- "$inpaint_dir")"
output_dir="$(realpath -- "$output_dir")"

echo "Input  Directory: $inpaint_dir"
echo "Output Directory: $output_dir"
echo "--- Slurm Configuration ---"
echo "Account:   ${slurm_account:-(default)}"
echo "Partition: $slurm_partition"
echo "Time:      $slurm_time"
echo "CPUs:      $slurm_cpus"
echo "Memory:    $slurm_mem"
echo "Limit:     ${job_limit:-(none)}"
echo "---------------------------"

cd "$inpaint_dir" || die "failed to enter $inpaint_dir"

readarray -t subjects < <(find . -maxdepth 1 -mindepth 1 -type d -name 'sub-*' | \
    sed 's|^\./||' | sort)

((${#subjects[@]} > 0)) || die "no sub-* directories found in $inpaint_dir"

submitted=0
for sub in "${subjects[@]}"; do
    sub_out="$output_dir/$sub"
    if [[ -d "$sub_out" ]]; then
        echo "Skipping $sub: output directory already exists"
        continue
    fi

    # Submit a job for the subject. Note: we are passing standard input from /dev/null
    # to avoid interference.
    sbatch_cmd=(sbatch
        --job-name="bs_$sub"
        --output="$output_dir/%x-%j.out"
        --error="$output_dir/%x-%j.err"
        --time="$slurm_time"
        --cpus-per-task="$slurm_cpus"
        --mem="$slurm_mem"
        --partition="$slurm_partition"
    )
    if [[ -n "$slurm_account" ]]; then
        sbatch_cmd+=(--account="$slurm_account")
    fi

    # The actual sbatch command takes the script path. We pass arguments
    # to *this* script inside the sbatch directive using --wrap.
    wrap_cmd=(bash "$SCRIPT_PATH" --worker "$inpaint_dir" "$output_dir" "$sub" -- "$@")

    printf "Submitting %s...\n" "$sub"
    "${sbatch_cmd[@]}" --wrap="$(print_command "${wrap_cmd[@]}")" </dev/null || \
        die "failed to submit job for $sub"

    ((submitted += 1))
    if [[ -n "$job_limit" ]] && ((submitted >= job_limit)); then
        echo "Reached submission limit of $job_limit jobs."
        break
    fi
done

echo "Submitted $submitted job(s)."