#!/bin/bash
#SBATCH --job-name=build_submit_deblur_unet_cv
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_deblur_preds_bright_sweep_26/logs/build_submit_deblur_unet_cv_%j.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_deblur_preds_bright_sweep_26/logs/build_submit_deblur_unet_cv_%j.err
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00


# *** USE THIS SCRIPT FOR IMAGE+CLIP UNET DEBLURRING ONLY ***

set -euo pipefail
export TOKENIZERS_PARALLELISM=false

echo "[INFO] Starting build & submit for UNet deblur CV jobs..."

ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
OUTROOT="/midtier/paetzollab/scratch/ads4015/temp_selma_deblur_preds_bright_sweep_26"
OUTDIR="${OUTROOT}/cv_folds"

REPEATS=3
SEED=100
CHANNELS="ALL"
TEST_SIZE=2

JOB_PREFIX="deblur_unet_bright_sweep_26"
ARRAY_SCRIPT="/home/ads4015/ssl_project/scripts/finetune_deblur_split_job_array2_unet.sh"
MAX_CONCURRENT=""

# match your deblur K grids (edit if desired)
declare -A COUNTS
# COUNTS[amyloid_plaque_patches]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
# COUNTS[c_fos_positive_patches]="1 2 3 4"
# COUNTS[cell_nucleus_patches]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
# COUNTS[vessels_patches]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"
COUNTS[amyloid_plaque_patches]="2 5 15 18"
COUNTS[c_fos_positive_patches]="1 2 3 4"
COUNTS[cell_nucleus_patches]="2 5 15 18 25"
COUNTS[vessels_patches]="2 5 15 18"

module load anaconda3/2022.10-34zllqw
source activate monai-env2

mkdir -p "$OUTDIR" "${OUTROOT}/logs" "${OUTROOT}/checkpoints" "${OUTROOT}/preds"

TASKS="$OUTDIR/${JOB_PREFIX}_tasks.txt"
: > "$TASKS"

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

echo "[INFO] Submitted UNet deblur array job: ${ARRAY_JOBID}"


