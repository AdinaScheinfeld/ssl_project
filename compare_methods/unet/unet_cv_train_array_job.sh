#!/bin/bash
#SBATCH --job-name=unet_cv_task
#SBATCH --partition=sablab-gpu
#SBATCH --account=sablab
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/compare_methods/unet/finetuned_cross_val/logs/slurm_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/compare_methods/unet/finetuned_cross_val/logs/slurm_%A_%a.err

# unet_cv_array_job.sh
#
# Runs ONE array task.
#
# The build+submit script will pass:
#   $1 = TASKS file
#   $2 = JOB_PREFIX (for logging convenience)
#
# Each line of TASKS file must be:
#   <SUBTYPE> <K> <FOLD_ID> <FOLDS_JSON_PATH>
# Example:
#   amyloid_plaque_patches 10 0 /.../cv_folds/amyloid_plaque_patches_folds_tr10_rep3.json
#
# This script reads the line corresponding to SLURM_ARRAY_TASK_ID and runs:
#   unet_cv_train_one_task.py

set -euo pipefail

# -------------------------
# Args
# -------------------------
TASKS_FILE="${1:?Usage: sbatch ... unet_cv_array_job.sh <tasks.txt> <job_prefix>}"
JOB_PREFIX="${2:-unetcv}"

# -------------------------
# Temp directory (safe scratch for Python / dataloader)
# -------------------------
export SCRATCH_ROOT=/midtier/paetzollab/scratch/ads4015
export TMPDIR="/tmp/${USER}/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

export PYTHONWARNINGS="ignore"

# -------------------------
# Load env
# -------------------------
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# -------------------------
# Experiment configuration
# -------------------------
DATA_ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
OUT_ROOT="/midtier/paetzollab/scratch/ads4015/compare_methods/unet/finetuned_cross_val"
mkdir -p "${OUT_ROOT}/logs"

# W&B: keep all run files under OUT_ROOT/logs
export WANDB_DIR="${OUT_ROOT}/logs"
export WANDB_CACHE_DIR="${OUT_ROOT}/logs/.wandb_cache"
export WANDB_CONFIG_DIR="${OUT_ROOT}/logs/.wandb_config"

# Training hyperparams you want for the sweep
EPOCHS=10
BATCH_SIZE=2
NUM_WORKERS=4
LR=1e-3
WEIGHT_DECAY=1e-4
ROI_SIZE="96,96,96"
EARLY_STOP_PATIENCE=3

# Seed comes from build script via --export (defaults to 100 if not passed)
SEED="${SEED:-100}"

# -------------------------
# Pick this task from TASKS_FILE
# -------------------------
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS_FILE" || true)
if [[ -z "$LINE" ]]; then
  echo "[ERROR] Could not read task line for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID} from $TASKS_FILE" >&2
  exit 1
fi

# Parse fields
# shellcheck disable=SC2206
PARTS=($LINE)
DTYPE="${PARTS[0]}"
K="${PARTS[1]}"
FOLD_ID="${PARTS[2]}"
FOLDS_JSON="${PARTS[3]}"

echo "[INFO] ${JOB_PREFIX}: task_id=${SLURM_ARRAY_TASK_ID} dtype=${DTYPE} K=${K} fold=${FOLD_ID}" \
     "fold_json=${FOLDS_JSON}" \
     "seed=${SEED}" \
     "node=$(hostname)" \
     "start=$(date)" \
     | tee -a "${OUT_ROOT}/logs/${JOB_PREFIX}_array_meta.log"

# -------------------------
# Run the Python task runner
# -------------------------
python -u /home/ads4015/ssl_project/compare_methods/unet/unet_cv_train_one_task.py \
  --data_root "${DATA_ROOT}" \
  --out_root  "${OUT_ROOT}" \
  --fold_json "${FOLDS_JSON}" \
  --datatype  "${DTYPE}" \
  --fold      "${FOLD_ID}" \
  --pool_n    "${K}" \
  --seed      "${SEED}" \
  --epochs    "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}" \
  --lr "${LR}" \
  --weight_decay "${WEIGHT_DECAY}" \
  --roi_size "${ROI_SIZE}" \
  --early_stop_patience "${EARLY_STOP_PATIENCE}" \
  --wandb_project "selma3d_unet_cv"

echo "[INFO] Done task_id=${SLURM_ARRAY_TASK_ID} at $(date)"
