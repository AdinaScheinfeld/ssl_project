#!/bin/bash
#SBATCH --job-name=build_submit_unet_random
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_unet_random2/logs/build_submit_%j.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_unet_random2/logs/build_submit_%j.err
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=6G
#SBATCH --time=02:00:00

set -euo pipefail

export SCRATCH_ROOT=/midtier/paetzollab/scratch/ads4015

ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
OUTROOT="${SCRATCH_ROOT}/temp_selma_segmentation_preds_unet_random2"
OUTDIR="${OUTROOT}/cv_folds"
ARRAY_SCRIPT="/home/ads4015/ssl_project/scripts/finetune_and_inference_cv_array_job_unet_random.sh"

REPEATS=3
SEED=100
CHANNELS="ALL"
TEST_SIZE=2
JOB_PREFIX="unet_random"
MAX_CONCURRENT=""   # e.g. "6" to cap concurrency

mkdir -p "${OUTROOT}/logs" "$OUTDIR"
TASKS="${OUTDIR}/${JOB_PREFIX}_tasks.txt"
: > "$TASKS"

module load anaconda3/2022.10-34zllqw
source activate monai-env2

# ---- choose K values per subtype (edit as you like) ----
declare -A COUNTS
# COUNTS[amyloid_plaque_patches]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
# COUNTS[c_fos_positive_patches]="1 2 3 4"
# COUNTS[cell_nucleus_patches]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
# COUNTS[vessels_patches]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
COUNTS[amyloid_plaque_patches]="2 3 4 11 12 13 14 15 16 17 18 19"
COUNTS[cell_nucleus_patches]="2 3 4 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
COUNTS[vessels_patches]="2 3 4 11 12 13 14 15 16 17 18 19 20"

echo "[INFO] Building folds + tasks at $(date)..."
for SUBTYPE in "${!COUNTS[@]}"; do
  for K in ${COUNTS[$SUBTYPE]}; do
    FJSON="${OUTDIR}/${SUBTYPE}_folds_tr${K}_rep${REPEATS}.json"

    python /home/ads4015/ssl_project/src/get_selma_cross_val_folds.py \
      --root "$ROOT" \
      --subtypes "$SUBTYPE" \
      --channel_substr "$CHANNELS" \
      --train_limit "$K" \
      --repeats "$REPEATS" \
      --test_size "$TEST_SIZE" \
      --seed "$SEED" \
      --output_json "$FJSON" || true

    if [[ -f "$FJSON" ]]; then
      for ((FID=0; FID<REPEATS; FID++)); do
        echo "$SUBTYPE $K $FID $FJSON" >> "$TASKS"
      done
    else
      echo "[WARN] Missing folds JSON: $FJSON (skip $SUBTYPE K=$K)"
    fi
  done
done

NUM_TASKS=$(wc -l < "$TASKS" || echo 0)
if [[ "$NUM_TASKS" -eq 0 ]]; then
  echo "[INFO] No tasks to submit. Exiting."
  exit 0
fi

if [[ -n "$MAX_CONCURRENT" ]]; then
  ARRAY_SPEC="0-$((NUM_TASKS-1))%${MAX_CONCURRENT}"
else
  ARRAY_SPEC="0-$((NUM_TASKS-1))"
fi

echo "[INFO] Submitting ${NUM_TASKS} tasks as array: ${ARRAY_SPEC}"
ARRAY_JOBID=$(sbatch --parsable \
  --job-name "${JOB_PREFIX}_array" \
  --array="${ARRAY_SPEC}" \
  "$ARRAY_SCRIPT" "$TASKS" "$JOB_PREFIX")

echo "[INFO] Submitted array job: ${ARRAY_JOBID}"





