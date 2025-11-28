#!/bin/bash
#SBATCH --job-name=cellseg3d_cv
#SBATCH --partition=sablab-gpu
#SBATCH --account=sablab
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/compare_methods/cellseg3d/finetuned_cross_val/logs/cellseg3d_cv_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/compare_methods/cellseg3d/finetuned_cross_val/logs/cellseg3d_cv_%A_%a.err
#SBATCH --array=0-299

# /home/ads4015/ssl_project/compare_methods/cellseg3d/cellseg3d_finetune_eval_cv_job.sh

# ================================================================
#  Slurm array script for CellSeg3D WNet3D finetuning CV
# ================================================================
# Each array index corresponds to one entry in cv_index.json:
#   {
#     "pool_size": 2,
#     "fold_index": 0
#   }
# The Python script cellseg3d_finetune_cv.py performs:
#  - building NIfTI→TIF split
#  - training WNet3D
#  - wrapping checkpoint
#  - inference on held-out test volumes
#  - saving predictions + metrics
# ================================================================

set -euo pipefail

# -------------------------
# 1. Activate environment
# -------------------------
eval "$(/home/ads4015/bin/micromamba shell hook -s bash)"
micromamba activate cellseg3d-env1

echo "Environment activated: $(which python)"
python --version

echo "CUDA devices:"
nvidia-smi || true

# -------------------------
# 2. Paths
# -------------------------
INDEX_FILE="/midtier/paetzollab/scratch/ads4015/compare_methods/cellseg3d/cv_index.json"
SCRIPT_PATH="/home/ads4015/ssl_project/compare_methods/cellseg3d/cellseg3d_finetune_eval_cv.py"
OUTPUT_ROOT="/midtier/paetzollab/scratch/ads4015/compare_methods/cellseg3d/finetuned_cross_val"
DATA_ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"

# -------------------------
# 3. Load JSON parameters
# -------------------------
if [ ! -f "$INDEX_FILE" ]; then
    echo "ERROR: Missing CV index: $INDEX_FILE"
    exit 1
fi

TASK_JSON=$(jq ".[${SLURM_ARRAY_TASK_ID}]" "$INDEX_FILE")
POOL_SIZE=$(echo "$TASK_JSON" | jq -r ".pool_size")
FOLD_INDEX=$(echo "$TASK_JSON" | jq -r ".fold_index")

if [ "$POOL_SIZE" = "null" ]; then
    echo "ERROR: Array index $SLURM_ARRAY_TASK_ID is out of range." 
    exit 1
fi

# -------------------------
# 4. Echo task info
# -------------------------
echo "========================================================="
echo "SLURM ARRAY TASK ID : $SLURM_ARRAY_TASK_ID"
echo "POOL_SIZE           : $POOL_SIZE"
echo "FOLD_INDEX          : $FOLD_INDEX"
echo "DATA_ROOT           : $DATA_ROOT"
echo "OUTPUT_ROOT         : $OUTPUT_ROOT"
echo "========================================================="

# -------------------------
# 5. Run the experiment
# -------------------------
python "$SCRIPT_PATH" \
    --pool-size "$POOL_SIZE" \
    --fold-index "$FOLD_INDEX"

STATUS=$?

echo "========================================================="
echo "Task finished: pool_size=$POOL_SIZE fold=$FOLD_INDEX (exit $STATUS)"
echo "========================================================="
