#!/bin/bash
#SBATCH --job-name=pretrain_sweep_h100
#SBATCH --output=logs/sweep_h100_%j.out
#SBATCH --error=logs/sweep_h100_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=minilab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G


# indicate starting
echo "Starting H100 sweep..."


# load environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1

export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SWEEP_GROUP=ibot-clip-sweep
export HW_TAG=H100

# run agent
wandb agent --count 40 adinas-wcm/ssl_project-sweep/zdnbxohv


# indicate ending
echo "H100 sweep complete."







