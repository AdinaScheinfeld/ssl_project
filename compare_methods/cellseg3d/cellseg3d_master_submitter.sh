#!/bin/bash
#
# /home/ads4015/ssl_project/compare_methods/cellseg3d/cellseg3d_master_submitter.sh
#
# Master launcher for CellSeg3D CV finetuning experiments.
# 1. Generates cv_index.json
# 2. Counts number of experiments
# 3. Submits Slurm array job
#

set -euo pipefail

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
DATA_ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
CLASS_NAME="cell_nucleus_patches"

OUT_INDEX="/midtier/paetzollab/scratch/ads4015/compare_methods/cellseg3d/cv_index.json"

PYTHON_GENERATOR="/home/ads4015/ssl_project/compare_methods/cellseg3d/make_cellseg3d_cv_index.py"
SLURM_SCRIPT="/home/ads4015/ssl_project/compare_methods/cellseg3d/cellseg3d_finetune_eval_cv_job.sh"

# -------------------------------------------------------------------
# Activate micromamba
# -------------------------------------------------------------------
echo "[INFO] Activating micromamba…"
set +u
export MKL_INTERFACE_LAYER=GNU
export MKL_THREADING_LAYER=GNU
eval "$(/home/ads4015/bin/micromamba shell hook -s bash)"
micromamba activate cellseg3d-env1
set -u

echo "[INFO] Python: $(which python)"
python --version

# -------------------------------------------------------------------
# Step 1: Generate cv_index.json
# -------------------------------------------------------------------
echo "[INFO] Generating CV index JSON…"
python "$PYTHON_GENERATOR"

if [[ ! -f "$OUT_INDEX" ]]; then
    echo "[ERROR] cv_index.json was not created."
    exit 1
fi

echo "[INFO] cv_index.json created at: $OUT_INDEX"

# -------------------------------------------------------------------
# Step 2: Count tasks
# -------------------------------------------------------------------
NUM_TASKS=$(jq length "$OUT_INDEX")
LAST_INDEX=$((NUM_TASKS - 1))

if [[ "$NUM_TASKS" -lt 1 ]]; then
    echo "[ERROR] cv_index.json is empty."
    exit 1
fi

echo "[INFO] Number of tasks: $NUM_TASKS (array indices 0 … $LAST_INDEX)"

# -------------------------------------------------------------------
# Step 3: Submit Slurm array job
# -------------------------------------------------------------------
echo "[INFO] Submitting Slurm array job:"
echo "       sbatch --array=0-$LAST_INDEX $SLURM_SCRIPT"

sbatch --array=0-"$LAST_INDEX" "$SLURM_SCRIPT"

echo "[INFO] Done. Slurm jobs submitted."
