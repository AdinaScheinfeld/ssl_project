#!/bin/bash
#SBATCH --job-name=build_submit_unet_cv
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/compare_methods/unet/finetuned_cross_val/logs/build_submit_%j.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/compare_methods/unet/finetuned_cross_val/logs/build_submit_%j.err

# unet_cv_build_and_submit.sh
#
# End-to-end driver for the UNet cross-val sweep.
# It does two things:
#   1) BUILD folds json files (by calling your existing get_selma_cross_val_folds.py)
#   2) BUILD a single TASKS file, then SUBMIT a GPU array over those tasks
#
# This mirrors the structure you already use for other experiments.
#
# What you already have:
#   /home/ads4015/ssl_project/src/get_selma_cross_val_folds.py
#
# What this script submits:
#   /home/ads4015/ssl_project/compare_methods/unet/unet_cv_array_job.sh
# which runs:
#   /home/ads4015/ssl_project/compare_methods/unet/unet_train_eval_from_folds_task.py

set -euo pipefail

echo "[INFO] Starting UNet CV build+submit at $(date)"

# -------------------------
# Safe temp directory (prevents /tmp NFS issues)
# -------------------------
export SCRATCH_ROOT=/midtier/paetzollab/scratch/ads4015
export TMPDIR="${SCRATCH_ROOT}/.tmp/build_${SLURM_JOB_ID}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

# -------------------------
# Paths
# -------------------------
ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
OUT_ROOT="/midtier/paetzollab/scratch/ads4015/compare_methods/unet/finetuned_cross_val"
FOLDS_OUTDIR="${OUT_ROOT}/cv_folds"           # where folds json + tasks file live
TASKS_FILE="${FOLDS_OUTDIR}/unetcv_tasks.txt" # one line per GPU task
JOB_PREFIX="unetcv"                           # appears in slurm logs

# Scripts
FOLD_GEN_PY="/home/ads4015/ssl_project/src/get_selma_cross_val_folds.py"
ARRAY_SCRIPT="/home/ads4015/ssl_project/compare_methods/unet/unet_cv_train_array_job.sh"

mkdir -p "${OUT_ROOT}/logs" "${FOLDS_OUTDIR}"

# -------------------------
# Fold generation config
# -------------------------
REPEATS=3      # number of folds / repeats
SEED=100       # global seed used by fold generator and training script
CHANNELS="ALL" # fold generator channel filter
TEST_SIZE=2    # you want exactly 2 held-out samples

# Optional: array concurrency cap ("" means no cap)
MAX_CONCURRENT=""   # e.g. set to "3" for 3 concurrent tasks

# -------------------------
# Define which K values to sweep per datatype.
# IMPORTANT: In your setup, K means the size of the *pool* (train+val) in the folds JSON.
# The training script will split that pool into 80/20 train/val.
# -------------------------

declare -A COUNTS
COUNTS[amyloid_plaque_patches]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
COUNTS[c_fos_positive_patches]="1 2 3 4"
COUNTS[cell_nucleus_patches]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
COUNTS[vessels_patches]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

# -------------------------
# Load conda env (needed for fold generator)
# -------------------------
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# -------------------------
# Build folds + tasks
# -------------------------
: > "$TASKS_FILE"   # truncate tasks file

echo "[INFO] Building folds jsons and tasks list..."

for SUBTYPE in "${!COUNTS[@]}"; do
  for K in ${COUNTS[$SUBTYPE]}; do

    # File naming: keep consistent with what you already do
    FJSON="${FOLDS_OUTDIR}/${SUBTYPE}_folds_tr${K}_rep${REPEATS}.json"

    # Build folds json.
    # We use || true so one failing (e.g. not enough samples) doesn't kill the whole sweep.
    python "$FOLD_GEN_PY" \
      --root "$ROOT" \
      --subtypes "$SUBTYPE" \
      --channel_substr "$CHANNELS" \
      --train_limit "$K" \
      --repeats "$REPEATS" \
      --test_size "$TEST_SIZE" \
      --seed "$SEED" \
      --output_json "$FJSON" \
      || true

    # Only add tasks if the folds file exists.
    if [[ -f "$FJSON" ]]; then
      # Add one task line per fold id
      for ((FID=0; FID<REPEATS; FID++)); do
        # Format is: SUBTYPE K FOLD_ID FOLDS_JSON
        echo "$SUBTYPE $K $FID $FJSON" >> "$TASKS_FILE"
      done
    else
      echo "[WARN] Missing folds JSON: $FJSON (skip $SUBTYPE K=$K)" >&2
    fi

  done
done

NUM_TASKS=$(wc -l < "$TASKS_FILE" || echo 0)

echo "[INFO] Built tasks file: $TASKS_FILE"
echo "[INFO] NUM_TASKS=$NUM_TASKS"

if [[ "$NUM_TASKS" -eq 0 ]]; then
  echo "[INFO] No tasks to submit. Exiting."
  exit 0
fi

# -------------------------
# Build array spec and submit
# -------------------------
if [[ -n "$MAX_CONCURRENT" ]]; then
  ARRAY_SPEC="0-$((NUM_TASKS-1))%${MAX_CONCURRENT}"
  echo "[INFO] Submitting ${NUM_TASKS} GPU tasks as array ${ARRAY_SPEC} (cap %${MAX_CONCURRENT})"
else
  ARRAY_SPEC="0-$((NUM_TASKS-1))"
  echo "[INFO] Submitting ${NUM_TASKS} GPU tasks as array ${ARRAY_SPEC} (no cap)"
fi

# We pass SEED via --export so the GPU jobs have the same seed.
ARRAY_JOBID=$(sbatch --parsable \
  --job-name "${JOB_PREFIX}_sweep" \
  --array="${ARRAY_SPEC}" \
  --export=ALL,SEED="${SEED}" \
  "$ARRAY_SCRIPT" "$TASKS_FILE" "$JOB_PREFIX")

echo "[INFO] Submitted array job: ${ARRAY_JOBID}"
echo "[INFO] Done build+submit at $(date)"
