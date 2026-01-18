#!/bin/bash
#SBATCH --job-name=ft_unet_sweep_agent
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/finetune_sweep_unet/logs/agent_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/finetune_sweep_unet/logs/agent_%A_%a.err
#SBATCH --partition=sablab-gpu
#SBATCH --account=sablab
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3-00:00:00
#SBATCH --array=0-5
#SBATCH --requeue

set -euo pipefail

SWEEP_ID="${1:-}"
if [[ -z "$SWEEP_ID" ]]; then
  echo "USAGE: sbatch $0 <SWEEP_ID>"
  echo "Example: sbatch $0 adinas-wcm/selma3d_finetune_unet_stratified/abcd1234"
  exit 1
fi

# ---- dirs ----
export SCRATCH_ROOT="/midtier/paetzollab/scratch/ads4015"
export SWEEP_ROOT="${SCRATCH_ROOT}/finetune_sweep_unet"
mkdir -p "${SWEEP_ROOT}/wandb" "${SWEEP_ROOT}/logs" "${SWEEP_ROOT}/tmp"

# ---- temp dir (node-local preferred) ----
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
mkdir -p "$TMPDIR"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"

echo "[INFO] SLURM_JOB_ID=$SLURM_JOB_ID SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"
echo "[INFO] TMPDIR=$TMPDIR"
echo "[INFO] SWEEP_ID=$SWEEP_ID"

# ---- env ----
module load anaconda3/2022.10-34zllqw
source activate monai-env2

export TOKENIZERS_PARALLELISM=false
export WANDB_DIR="${SWEEP_ROOT}/wandb"
export WANDB_RESUME=allow
export PL_DISABLE_SLURM=1

# If Slurm sends TERM near time limit, exit cleanly so requeue can happen.
trap 'echo "[INFO] Caught termination signal; exiting to allow requeue."; exit 0' TERM INT

# Optional: make dataloading + NCCL less fragile
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=0
export NCCL_P2P_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run agent (each array task runs its own agent)
wandb agent "$SWEEP_ID"

echo "[INFO] Agent finished."
rm -rf "$TMPDIR" || true



