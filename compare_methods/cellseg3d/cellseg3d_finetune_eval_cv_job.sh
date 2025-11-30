#!/bin/bash
#SBATCH --job-name=cellseg3d_cv
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=24G
#SBATCH --time=48:00:00
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/compare_methods/cellseg3d/finetuned_cross_val/logs/cellseg3d_cv_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/compare_methods/cellseg3d/finetuned_cross_val/logs/cellseg3d_cv_%A_%a.err

# ================================================================
# /home/ads4015/ssl_project/compare_methods/cellseg3d/cellseg3d_finetune_eval_cv_job.sh
# ================================================================
# Executes one fold of the CellSeg3D WNet3D finetuning CV pipeline.
# ================================================================

set -euo pipefail

echo "==========================================================="
echo "[INFO] Starting CellSeg3D CV task"
echo "SLURM ARRAY ID: ${SLURM_ARRAY_TASK_ID}"
echo "==========================================================="

# -------------------------------------------------------------------
# 1. Activate micromamba environment
# -------------------------------------------------------------------
eval "$(/home/ads4015/bin/micromamba shell hook -s bash)"
micromamba activate cellseg3d-env1

echo "[INFO] Python: $(which python)"
python --version

echo "[INFO] CUDA status:"
nvidia-smi || echo "[WARN] Could not run nvidia-smi"

# -------------------------------------------------------------------
# 2. Paths
# -------------------------------------------------------------------
INDEX_FILE="/midtier/paetzollab/scratch/ads4015/compare_methods/cellseg3d/cv_index.json"
SCRIPT_PATH="/home/ads4015/ssl_project/compare_methods/cellseg3d/cellseg3d_finetune_eval_cv.py"

DATA_ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
OUTPUT_ROOT="/midtier/paetzollab/scratch/ads4015/compare_methods/cellseg3d/finetuned_cross_val"

# -------------------------------------------------------------------
# 3. Validate cv_index.json exists
# -------------------------------------------------------------------
if [[ ! -f "$INDEX_FILE" ]]; then
    echo "[ERROR] Missing cv_index.json: $INDEX_FILE"
    exit 1
fi

# -------------------------------------------------------------------
# 4. Load JSON entry for this array task
# -------------------------------------------------------------------
TASK_JSON=$(jq ".[${SLURM_ARRAY_TASK_ID}]" "$INDEX_FILE")

POOL_SIZE=$(echo "$TASK_JSON" | jq -r ".pool_size")
FOLD_INDEX=$(echo "$TASK_JSON" | jq -r ".fold_index")

if [[ "$POOL_SIZE" == "null" ]]; then
    echo "[ERROR] Invalid array index ${SLURM_ARRAY_TASK_ID}"
    exit 1
fi

echo "-----------------------------------------------------------"
echo "[INFO] Running experiment:"
echo "POOL_SIZE  = $POOL_SIZE"
echo "FOLD_INDEX = $FOLD_INDEX"
echo "DATA_ROOT  = $DATA_ROOT"
echo "OUTPUT_ROOT= $OUTPUT_ROOT"
echo "-----------------------------------------------------------"

# -------------------------------------------------------------------
# 5. Run the finetune + eval experiment
# -------------------------------------------------------------------
python "$SCRIPT_PATH" \
    --pool-size "$POOL_SIZE" \
    --fold-index "$FOLD_INDEX"

EXIT_CODE=$?

echo "-----------------------------------------------------------"
echo "[INFO] Task finished."
echo "POOL_SIZE=$POOL_SIZE  FOLD_INDEX=$FOLD_INDEX"
echo "EXIT_CODE=$EXIT_CODE"
echo "-----------------------------------------------------------"
