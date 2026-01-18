#!/bin/bash
#SBATCH --job-name=lsm_sweep_array_unet
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/pretrain_sweep_unet2/logs/wandb_array_unet_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/pretrain_sweep_unet2/logs/wandb_array_unet_%A_%a.err
#SBATCH --partition=sablab-gpu
#SBATCH --account=sablab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=3-00:00:00
#SBATCH --array=1-4
#SBATCH --requeue
#SBATCH --signal=B:TERM@120

# USAGE: sbatch sweep_unet/run_agent_array_unet.sh <SWEEP_ID>
SWEEP_ID="$1"
if [ -z "$SWEEP_ID" ]; then
  echo "ERROR: Provide SWEEP_ID (e.g., adinas-wcm/ibot-clip-pretrain-lsm-all-unet/abc123xy)"; exit 1
fi

echo "[Task $SLURM_ARRAY_TASK_ID] Starting W&B agent for $SWEEP_ID"

export PRETRAIN_SWEEP_DIR="/midtier/paetzollab/scratch/ads4015/pretrain_sweep_unet2"
mkdir -p "$PRETRAIN_SWEEP_DIR"/checkpoints "$PRETRAIN_SWEEP_DIR"/configs "$PRETRAIN_SWEEP_DIR"/wandb "$PRETRAIN_SWEEP_DIR"/logs "$PRETRAIN_SWEEP_DIR"/tmp

module load anaconda3/2022.10-34zllqw
source activate monai-env2

export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR="$PRETRAIN_SWEEP_DIR/wandb"
export WANDB_RESUME=allow
export PL_DISABLE_SLURM=1

# Exit cleanly on pre-timeout termination so Slurm requeue behaves nicely
trap 'echo "[INFO] Caught termination signal; exiting to allow requeue."; exit 0' TERM INT

# tell wrapper where the base config is
export BASE_CONFIG="/home/ads4015/ssl_project/configs/all_datasets_clip_pretrain_2_config_unet.yaml"

AGENT_COUNT=${AGENT_COUNT:-0}
if [ "$AGENT_COUNT" -gt 0 ]; then
  wandb agent --count "$AGENT_COUNT" "$SWEEP_ID"
else
  wandb agent "$SWEEP_ID"
fi

echo "[Task $SLURM_ARRAY_TASK_ID] W&B agent for $SWEEP_ID complete"

