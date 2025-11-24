#!/bin/bash
#
# submit_microsam_array.sh
#
# Build the list of input patches and submit the microSAM Slurm array.
# Run this from any node (e.g. minilab-cpu), NOT on a GPU node.

# /home/ads4015/ssl_project/compare_methods/micro_sam/submit_micro_sam_batch_array_jobs.sh

set -euo pipefail

# ---- User-configurable settings ----

# Root for your input patches
IN_ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"

# Where to write the file list
FILE_LIST="/midtier/paetzollab/scratch/ads4015/compare_methods/micro_sam/microsam_input_list.txt"

# Slurm script to submit (the array job)
SLURM_SCRIPT="/home/ads4015/ssl_project/compare_methods/micro_sam/micro_sam_batch_array_job.sh"

# Number of patches each array task should process
PER_TASK=1   # must match --per_task in run_microsam_array.slurm AND run_microsam_batch.py

# ---- No changes needed below this line ----

echo "[submit_microsam_array] Working directory: $(pwd)"
echo "[submit_microsam_array] Input root: $IN_ROOT"
echo "[submit_microsam_array] File list:  $FILE_LIST"

# 1) Build list of all image patches (exclude *_label.nii.gz)
echo "[submit_microsam_array] Building file list..."
find "$IN_ROOT" \
  -type f -name "patch_*_ch*.nii.gz" \
  ! -name "*_label.nii.gz" \
  | sort > "$FILE_LIST"

NUM_FILES=$(wc -l < "$FILE_LIST")
echo "[submit_microsam_array] Found $NUM_FILES input patches."

if [ "$NUM_FILES" -eq 0 ]; then
    echo "[submit_microsam_array] ERROR: No input files found. Exiting."
    exit 1
fi

# 2) Compute number of array tasks
NUM_TASKS=$(( (NUM_FILES + PER_TASK - 1) / PER_TASK ))
echo "[submit_microsam_array] Using PER_TASK=$PER_TASK"
echo "[submit_microsam_array] Will submit array with $NUM_TASKS tasks (0..$((NUM_TASKS - 1)))."

# 3) Submit the array job
if [ ! -f "$SLURM_SCRIPT" ]; then
    echo "[submit_microsam_array] ERROR: Slurm script not found at $SLURM_SCRIPT"
    exit 1
fi

echo "[submit_microsam_array] Submitting Slurm array..."
sbatch --array=0-$((NUM_TASKS - 1)) "$SLURM_SCRIPT"

echo "[submit_microsam_array] Done."





