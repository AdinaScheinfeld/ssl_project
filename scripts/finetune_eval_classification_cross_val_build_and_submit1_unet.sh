#!/bin/bash
#SBATCH --job-name=build_submit_cls_unet
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_super_sweep2/logs/build_submit_cls_unet_%j.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_super_sweep2/logs/build_submit_cls_unet_%j.err
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00



set -euo pipefail

echo "[INFO] Starting build+submit UNet classification sweep2 on $(date)..."

module load anaconda3/2022.10-34zllqw
source activate monai-env2

# -----------------------
# paths
# -----------------------
ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
OUTROOT="/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_super_sweep2"

LOGDIR="${OUTROOT}/logs"
FOLDDIR="${OUTROOT}/cv_folds"
CKPTDIR="${OUTROOT}/checkpoints"
METRICSROOT="${OUTROOT}/cls_metrics"

mkdir -p "$LOGDIR" "$FOLDDIR" "$CKPTDIR" "$METRICSROOT"

# tasks file (for the GPU array)
JOB_PREFIX="cls_unet_super_sweep2"
TASKS="${FOLDDIR}/${JOB_PREFIX}_tasks.txt"
: > "$TASKS"

# -----------------------
# extra classes (same as your other setup)
# -----------------------
MESO_ROOT="/midtier/paetzollab/scratch/ads4015/all_mesospim_patches"
ALLEN_ROOT="/midtier/paetzollab/scratch/ads4015/all_allen_human_patches"

EXTRA_CLASS_GLOBS=(
  "VIP_ASLM_off:${MESO_ROOT}/*VIP_ASLM_off*.nii*"
  "VIP_ASLM_on:${MESO_ROOT}/*VIP_ASLM_on*.nii*"
  "TPH2:${MESO_ROOT}/*TPH2*.nii*"
  "stain-CR:${ALLEN_ROOT}/*_cr_*ps96*.nii*"
  "stain-LEC:${ALLEN_ROOT}/*_lec_*ps96*.nii*"
  "stain-NN:${ALLEN_ROOT}/*_nn_*ps96*.nii*"
  "stain-NPY:${ALLEN_ROOT}/*_npy_*ps96*.nii*"
  "stain-YO:${ALLEN_ROOT}/*_yo_*ps96*.nii*"
)

EXTRA_CLASS_GLOBS_ARGS=()
for spec in "${EXTRA_CLASS_GLOBS[@]}"; do
  EXTRA_CLASS_GLOBS_ARGS+=(--extra_class_globs "$spec")
done

# -----------------------
# folds creation knobs
# -----------------------
REPEATS=3
TEST_FRAC=0.2
FIXED_TEST_PER_CLASS=2
LOCK_TEST=false
CHANNEL_SUBSTR=ALL
SEED=100

# which train-per-class values to run (edit as desired)
TRAIN_PER_CLASS_LIST=(2 5 15)

# -----------------------
# build folds + tasks
# -----------------------
echo "[INFO] Writing folds JSONs into: $FOLDDIR"
for TR in "${TRAIN_PER_CLASS_LIST[@]}"; do
  FJSON="${FOLDDIR}/cls_folds_tr${TR}_test${FIXED_TEST_PER_CLASS}_rep${REPEATS}.json"

  python /home/ads4015/ssl_project/src/get_classification_cross_val_folds.py \
    --root_dir "$ROOT" \
    --repeats "$REPEATS" \
    --fixed_test_per_class "$FIXED_TEST_PER_CLASS" \
    --train_per_class "$TR" \
    --test_frac "$TEST_FRAC" \
    $( $LOCK_TEST && echo "--lock_test" || true ) \
    --channel_substr "$CHANNEL_SUBSTR" \
    --seed "$SEED" \
    "${EXTRA_CLASS_GLOBS_ARGS[@]}" \
    --output_json "$FJSON"

  if [[ -f "$FJSON" ]]; then
    # add one task line per fold id
    for ((FID=0; FID<REPEATS; FID++)); do
      # format: TR FID FJSON
      echo "$TR $FID $FJSON" >> "$TASKS"
    done
  else
    echo "[WARN] Missing folds JSON: $FJSON (skip TR=$TR)"
  fi
done

NUM_TASKS=$(wc -l < "$TASKS" || echo 0)
if [[ "$NUM_TASKS" -eq 0 ]]; then
  echo "[ERROR] No tasks created. Exiting."
  exit 1
fi

echo "[INFO] Built tasks file: $TASKS"
echo "[INFO] NUM_TASKS=$NUM_TASKS"

# -----------------------
# submit GPU array
# -----------------------
ARRAY_SCRIPT="/home/ads4015/ssl_project/scripts/finetune_eval_classification_cross_val_job_array1_unet.sh"

# optional concurrency cap (set like "6" to limit how many run at once)
MAX_CONCURRENT=""
if [[ -n "$MAX_CONCURRENT" ]]; then
  ARRAY_SPEC="0-$((NUM_TASKS-1))%${MAX_CONCURRENT}"
else
  ARRAY_SPEC="0-$((NUM_TASKS-1))"
fi

echo "[INFO] Submitting GPU array: --array=${ARRAY_SPEC}"
ARRAY_JOBID=$(sbatch --parsable \
  --job-name "${JOB_PREFIX}_arr" \
  --array="${ARRAY_SPEC}" \
  "$ARRAY_SCRIPT" "$TASKS")

echo "[INFO] Submitted array job: $ARRAY_JOBID"
echo "[INFO] Done."
