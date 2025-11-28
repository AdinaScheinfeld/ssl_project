#!/bin/bash
#
# /home/ads4015/ssl_project/compare_methods/cellseg3d/cellseg3d_master_submitter.sh
#
# This script:
#   1. Generates cv_index.json (pool_size × fold_index entries)
#   2. Counts the number of tasks
#   3. Submits the correct Slurm array job
#
# Usage:
#   bash master_submit_cellseg3d_cv.sh
#

set -euo pipefail

# ---------------------------------------------
# Configuration
# ---------------------------------------------
DATA_ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
CLASS_NAME="cell_nucleus_patches"

OUT_INDEX="/midtier/paetzollab/scratch/ads4015/compare_methods/cellseg3d/cv_index.json"

PYTHON_GENERATOR="/home/ads4015/ssl_project/compare_methods/cellseg3d/make_cellseg3d_cv_index.py"
SLURM_SCRIPT="/home/ads4015/ssl_project/compare_methods/cellseg3d/cellseg3d_finetune_eval_cv_job.sh"

# ---------------------------------------------
# Step 1: Activate micromamba environment
# ---------------------------------------------
echo "[INFO] Activating micromamba..."
set +u
export MKL_INTERFACE_LAYER=GNU
export MKL_THREADING_LAYER=GNU
eval "$(/home/ads4015/bin/micromamba shell hook -s bash)"
micromamba activate cellseg3d-env1
set -u

echo "[INFO] Python: $(which python)"
python --version

# ---------------------------------------------
# Step 2: Generate CV index JSON
# ---------------------------------------------
echo "[INFO] Generating cv_index.json ..."
python "$PYTHON_GENERATOR"

if [ ! -f "$OUT_INDEX" ]; then
    echo "[ERROR] cv_index.json was not created: $OUT_INDEX"
    exit 1
fi

echo "[INFO] cv_index.json created at: $OUT_INDEX"

# ---------------------------------------------
# Step 3: Count how many tasks we need
# ---------------------------------------------
NUM_TASKS=$(jq length "$OUT_INDEX")
LAST_INDEX=$((NUM_TASKS - 1))

echo "[INFO] Number of CV tasks: $NUM_TASKS (array 0..$LAST_INDEX)"

if [ "$NUM_TASKS" -lt 1 ]; then
    echo "[ERROR] cv_index.json is empty."
    exit 1
fi

# ---------------------------------------------
# Step 4: Submit Slurm array job
# ---------------------------------------------
echo "[INFO] Submitting Slurm array job..."
echo "       sbatch --array=0-$LAST_INDEX $SLURM_SCRIPT"

sbatch --array=0-$LAST_INDEX "$SLURM_SCRIPT"

echo "[INFO] Done. Slurm array submitted."
