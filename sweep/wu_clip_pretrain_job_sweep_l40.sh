#!/bin/bash
#SBATCH --job-name=pretrain_sweep_l40
#SBATCH --output=logs/sweep_l40_%j.out
#SBATCH --error=logs/sweep_l40_%j.err
#SBATCH --time=7-00:00:00
#SBATCH --partition=minilab-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G


# indicate starting
echo "Starting L40 sweep..."


# load environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1

export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=0
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SWEEP_GROUP=ibot-clip-sweep
export HW_TAG=L40

# run agent
wandb agent --count 40 adinas-wcm/ssl_project-sweep/0wu7hwf0


# indicate ending
echo "L40 sweep complete."







