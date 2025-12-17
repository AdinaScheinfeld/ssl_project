#!/bin/bash
#SBATCH --job-name=build_submit_inpaint_cv
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_inpaint_preds_autumn_sweep_27_ntc/logs/build_submit_inpaint_cv_%j.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_inpaint_preds_autumn_sweep_27_ntc/logs/build_submit_inpaint_cv_%j.err
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00

# *** USE THIS FILE ONLY FOR AUTUMN_SWEEP_27 (IMAGE+CLIP) ***

# /home/ads4015/ssl_project/scripts/finetune_inpaint_split_build_submit1.sh - Build and submit inpainting CV jobs

# Build CV folds and submit inpainting array jobs
set -euo pipefail
export TOKENIZERS_PARALLELISM=false  # disable tokenizer parallelism warnings

# indicate starting
echo "[INFO] Starting build & submit for inpainting CV jobs..."

# parameters
ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches" # location of finetuning data
OUTDIR="/midtier/paetzollab/scratch/ads4015/temp_selma_inpaint_preds_autumn_sweep_27_ntc/cv_folds" # output dir for folds and tasks
REPEATS=3
SEED=100
CHANNELS="ALL"
TEST_SIZE=2
JOB_PREFIX="inpaint27"
ARRAY_SCRIPT="/home/ads4015/ssl_project/scripts/finetune_inpaint_split_job_array1.sh"
MAX_CONCURRENT=""   # set to "" for no cap

# sweep counts per subtype
declare -A COUNTS
COUNTS[amyloid_plaque_patches]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
COUNTS[c_fos_positive_patches]="1 2 3 4"
COUNTS[cell_nucleus_patches]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
COUNTS[vessels_patches]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

# load environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# prepare tasks file
mkdir -p "$OUTDIR"
TASKS="$OUTDIR/${JOB_PREFIX}_tasks.txt"
: > "$TASKS"

# build folds and tasks
echo "[INFO] Building folds and tasks at $(date) ..."
for SUBTYPE in "${!COUNTS[@]}"; do
  for K in ${COUNTS[$SUBTYPE]}; do
    FJSON="$OUTDIR/${SUBTYPE}_folds_tr${K}_rep${REPEATS}.json"

    # create folds json (reusing existing python utility from segmentation)
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

# submit array job
NUM_TASKS=$(wc -l < "$TASKS" || echo 0)
if [[ "$NUM_TASKS" -eq 0 ]]; then
  echo "[INFO] No tasks to submit. Exiting."
  exit 0
fi

# determine array spec
if [[ -n "$MAX_CONCURRENT" ]]; then
  ARRAY_SPEC="0-$((NUM_TASKS-1))%${MAX_CONCURRENT}"
  echo "[INFO] Submitting ${NUM_TASKS} GPU tasks as an array (max concurrency %${MAX_CONCURRENT})..."
else
  ARRAY_SPEC="0-$((NUM_TASKS-1))"
  echo "[INFO] Submitting ${NUM_TASKS} GPU tasks as an array (no explicit concurrency cap)..."
fi

# submit array job
ARRAY_JOBID=$(sbatch --parsable \
  --job-name "${JOB_PREFIX}_sweep" \
  --array="$ARRAY_SPEC" \
  "$ARRAY_SCRIPT" "$TASKS" "$JOB_PREFIX")


# indicate done
echo "[INFO] Submitted array job: ${ARRAY_JOBID}"
























