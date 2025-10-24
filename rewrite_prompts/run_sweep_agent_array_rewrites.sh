#!/bin/bash
#SBATCH --job-name=lsm_sweep_array_rewrites
#SBATCH --output=logs/wandb_array_rewrites_%A_%a.out
#SBATCH --error=logs/wandb_array_rewrites_%A_%a.err
#SBATCH --partition=minilab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2 # each task needs 2 GPUs
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=7-00:00:00
#SBATCH --array=1-4 # submit 4 tasks

# USAGE: sbatch rewrite_prompts/run_agent_array_rewrites.sh <SWEEP_ID>
# EXAMPLE: sbatch rewrite_prompts/run_agent_array_rewrites.sh adinas-wcm/ibot-clip-pretrain-lsm-all/abc123xy
SWEEP_ID="$1"
if [ -z "$SWEEP_ID" ]; then
  echo "ERROR: Provide SWEEP_ID (e.g., adinas-wcm/ibot-clip-pretrain-lsm-all-rewrites/abc123xy)"; exit 1
fi

# indicate starting
echo "[Task $SLURM_ARRAY_TASK_ID] Starting W&B agent for $SWEEP_ID"

# base output directory (shared across tasks)
export PRETRAIN_SWEEP_DIR="/ministorage/adina/pretrain_sweep_rewrites"
mkdir -p "$PRETRAIN_SWEEP_DIR"/checkpoints "$PRETRAIN_SWEEP_DIR"/configs "$PRETRAIN_SWEEP_DIR"/wandb "$PRETRAIN_SWEEP_DIR"/logs "$PRETRAIN_SWEEP_DIR"/tmp

# activate conda environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# runtime/env niceties
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_DIR="/ministorage/adina/pretrain_sweep_rewrites/wandb"
export WANDB_RESUME=allow # force all wandb.init calls to reuse the agent's run
export PL_DISABLE_SLURM=1

# tell the wrapper where the base config lives
export BASE_CONFIG="/home/ads4015/ssl_project/rewrite_prompts/all_datasets_clip_pretrain_2_config_rewrites.yaml"

# optional: limit how many runs *this* agent executes before exiting
# AGENT_COUNT=0 means unlimited until sweep ends or walltime hits.
AGENT_COUNT=${AGENT_COUNT:-0}

if [ "$AGENT_COUNT" -gt 0 ]; then
  wandb agent --count "$AGENT_COUNT" "$SWEEP_ID"
else
  wandb agent "$SWEEP_ID"
fi

# indicate completion
echo "[Task $SLURM_ARRAY_TASK_ID] W&B agent for $SWEEP_ID complete"




