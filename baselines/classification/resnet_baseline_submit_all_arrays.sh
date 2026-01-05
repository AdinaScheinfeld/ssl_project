#!/bin/bash
#SBATCH --job-name=resnet3d_cls
#SBATCH --partition=sablab-gpu
#SBATCH --account=sablab
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_resnet/logs/resnet_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_resnet/logs/resnet_%A_%a.err
#SBATCH --array=0-56

# /home/ads4015/ssl_project/baselines/classification/resnet_baseline_submit_all_arrays.sh
#
# Runs ResNet3D baseline across ALL folds and ALL JSONs using ONE array job

# submit like:
# sbatch baselines/classification/resnet_baseline_submit_all_arrays.sh /path/to/jsons/folder
# ex: sbatch baselines/classification/resnet_baseline_submit_all_arrays.sh /midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_resnet

# log starting time
echo "[INFO] Job started at $(date)"

set -euo pipefail

JSON_DIR="${1:?Provide directory with cls_folds_*.json}"
OUT_ROOT="/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_resnet"
mkdir -p "$OUT_ROOT/logs"

# ---- temp dir (safe for python/multiprocessing) ----
export SCRATCH_ROOT=/midtier/paetzollab/scratch/ads4015
export TMPDIR="${SCRATCH_ROOT}/.tmp/build_${SLURM_JOB_ID}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

export WANDB_DIR="$OUT_ROOT/wandb"
mkdir -p "$WANDB_DIR"

# ---- load modules and activate env ----
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# fold map file
MAP_FILE="$OUT_ROOT/fold_map.txt"


LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$MAP_FILE" || true)
if [[ -z "$LINE" ]]; then
  echo "[INFO] No more folds, exiting"
  exit 0
fi

IFS='|' read -r GID JSON_PATH FOLD_ID <<< "$LINE"

echo "[INFO] Running fold $FOLD_ID from $JSON_PATH"

python /home/ads4015/ssl_project/baselines/classification/resnet_baseline.py \
  --fold_json "$JSON_PATH" \
  --fold_id "$FOLD_ID" \
  --output_root "$OUT_ROOT" \
  --batch_size 4 \
  --epochs 500 \
  --patience 100 \
  --lr 1e-3 \
  --use_wandb

echo "[INFO] Done at $(date)"



