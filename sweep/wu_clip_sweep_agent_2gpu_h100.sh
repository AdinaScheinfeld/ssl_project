#!/bin/bash
#SBATCH --job-name=wb_agent2_h100
#SBATCH --output=logs/wb_agent2_h100_%j.out
#SBATCH --error=logs/wb_agent2_h100_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G

module load anaconda3/2022.10-34zllqw
source activate monai-env1

export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export START_EPOCH="$(date +%s)"

if [ -z "$SWEEP_ID_H100" ]; then
    echo "ERROR: export SWEEP_ID_H100='user/project/xxxx'"; exit 1;
fi

wandb agent "$SWEEP_ID_H100"






