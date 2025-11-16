#!/bin/bash
#SBATCH --job-name=build_submit_ft_cv_rand
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_rand_v2/logs/build_submit_ft_cv_rand_%j.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_rand_v2/logs/build_submit_ft_cv_rand_%j.err
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=02:00:00

# finetune_and_inference_cv_build_and_submit_rand.sh - Script to build cross-validation folds and submit finetuning+inference array jobs using SLURM.


# indicate starting
echo "[INFO] Starting build and submit finetune+inference CV jobs..."


set -euo pipefail

# config
ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
OUTDIR="/midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_rand_v2/cv_folds" # output dir for folds jsons and tasks file
REPEATS=3
SEED=100
CHANNELS="ALL"
TEST_SIZE=2
JOB_PREFIX="rand"
ARRAY_SCRIPT="/home/ads4015/ssl_project/scripts/finetune_and_inference_cv_array_job_rand.sh"
MAX_CONCURRENT="" # array concurrency cap (set to "" for no cap, set to e.g. "3" for max 3 concurrent tasks)

# sweep counts per subtype
declare -A COUNTS
COUNTS[amyloid_plaque_patches]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
COUNTS[c_fos_positive_patches]="1 2 3 4"
COUNTS[cell_nucleus_patches]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
COUNTS[vessels_patches]="1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

# load conda env
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# prepare output dirs and tasks file
mkdir -p "$OUTDIR"
TASKS="${OUTDIR}/${JOB_PREFIX}_tasks.txt"
: > "$TASKS"

# build folds and tasks
echo "[INFO] Building folds and tasks list at $(date)..."
for SUBTYPE in "${!COUNTS[@]}"; do
  for K in ${COUNTS[$SUBTYPE]}; do
    FJSON="${OUTDIR}/${SUBTYPE}_folds_tr${K}_rep${REPEATS}.json"

    # build folds json
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

# build the array specification with optional concurrency cap
if [[ -n "$MAX_CONCURRENT" ]]; then
  ARRAY_SPEC="0-$((NUM_TASKS-1))%${MAX_CONCURRENT}"
  echo "[INFO] Submitting ${NUM_TASKS} GPU tasks as an array (max concurrency %${MAX_CONCURRENT})..."
else
  ARRAY_SPEC="0-$((NUM_TASKS-1))"
  echo "[INFO] Submitting ${NUM_TASKS} GPU tasks as an array (no explicit concurrency cap)..."
fi

# submit the array with the computed ARRAY_SPEC
ARRAY_JOBID=$(sbatch --parsable \
  --job-name "${JOB_PREFIX}_sweep" \
  --array="${ARRAY_SPEC}" \
  "$ARRAY_SCRIPT" "$TASKS" "$JOB_PREFIX")


# finalize job submission
echo "[INFO] Submitted array job: ${ARRAY_JOBID}"



















