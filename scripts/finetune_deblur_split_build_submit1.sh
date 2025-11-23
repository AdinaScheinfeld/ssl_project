#!/bin/bash
#SBATCH --job-name=build_submit_deblur_cv
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_deblur_preds_autumn_sweep_27/logs/build_submit_deblur_cv_%j.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_deblur_preds_autumn_sweep_27/logs/build_submit_deblur_cv_%j.err
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00

# /home/ads4015/ssl_project/scripts/finetune_deblur_split_build_submit1.sh - build CV folds and submit deblurring finetune array jobs

set -euo pipefail
export TOKENIZERS_PARALLELISM=false

# indicate starting
echo "[INFO] Starting build & submit for deblur CV jobs..."

# root directory with *sharp* patches
ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"

# output directory for folds JSON and task list
OUTDIR="/midtier/paetzollab/scratch/ads4015/temp_selma_deblur_preds_autumn_sweep_27/cv_folds"

# Number of repeated CV folds per setting (same as inpainting script)
REPEATS=3
SEED=100
CHANNELS="ALL"
TEST_SIZE=2

# job prefix for array job naming / task file
JOB_PREFIX="deblur27"
ARRAY_SCRIPT="/home/ads4015/ssl_project/scripts/finetune_deblur_split_job_array1.sh"
MAX_CONCURRENT=""  # e.g. "6" to cap concurrency; empty for no cap

# per-subtype training subset sizes (same pattern as inpainting)
declare -A COUNTS
COUNTS[amyloid_plaque_patches]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
COUNTS[c_fos_positive_patches]="1 2 3 4"
COUNTS[cell_nucleus_patches]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
COUNTS[vessels_patches]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

# load env
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# prepare folders
mkdir -p "$OUTDIR"
mkdir -p "$(dirname "$OUTDIR")/logs"

TASKS="$OUTDIR/${JOB_PREFIX}_tasks.txt"
: > "$TASKS"  # truncate

# build folds and tasks list
echo "[INFO] Building folds and tasks at $(date) ..."
for SUBTYPE in "${!COUNTS[@]}"; do
  for K in ${COUNTS[$SUBTYPE]}; do
    FJSON="$OUTDIR/${SUBTYPE}_folds_tr${K}_rep${REPEATS}.json"

    python /home/ads4015/ssl_project/src/get_inpaint_cross_val_folds.py \
      --root "$ROOT" \
      --subtypes "$SUBTYPE" \
      --channel_substr "$CHANNELS" \
      --train_per_class "$K" \
      --repeats "$REPEATS" \
      --fixed_test_per_class "$TEST_SIZE" \
      --seed "$SEED" \
      --output_json "$FJSON" || true

    if [[ -f "$FJSON" ]]; then
      for ((FID=0; FID<REPEATS; FID++)); do
        echo "$SUBTYPE $K $FID $FJSON" >> "$TASKS"
      done
    else
      echo "[WARN] Missing folds JSON: $FJSON (skip $SUBTYPE K=$K)" >&2
    fi
  done
done

NUM_TASKS=$(wc -l < "$TASKS" || echo 0)
if [[ "$NUM_TASKS" -eq 0 ]]; then
  echo "[INFO] No tasks to submit. Exiting."
  exit 0
fi

# array spec
if [[ -n "$MAX_CONCURRENT" ]]; then
  ARRAY_SPEC="0-$((NUM_TASKS-1))%${MAX_CONCURRENT}"
  echo "[INFO] Submitting ${NUM_TASKS} GPU tasks as an array (max concurrency %${MAX_CONCURRENT})..."
else
  ARRAY_SPEC="0-$((NUM_TASKS-1))"
  echo "[INFO] Submitting ${NUM_TASKS} GPU tasks as an array (no explicit concurrency cap)..."
fi

ARRAY_JOBID=$(sbatch --parsable \
  --job-name "${JOB_PREFIX}_sweep" \
  --array="$ARRAY_SPEC" \
  "$ARRAY_SCRIPT" "$TASKS" "$JOB_PREFIX")

# final info
echo "[INFO] Submitted deblur array job: ${ARRAY_JOBID}"










