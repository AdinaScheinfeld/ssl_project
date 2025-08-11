#!/bin/bash
#SBATCH --job-name=clip_pretrain
#SBATCH --output=logs/clip_pretrain_%j.out
#SBATCH --error=logs/clip_pretrain_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:h100:2
#SBATCH --mem=180G
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2 # 1 task per gpu


# indicate starting
echo "Starting LSM pretraining..."



# activate conda environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# wandb login
export WANDB_API_KEY=9778703390de02a48bdea1415c3c36c3bae408c0
wandb login $WANDB_API_KEY

export TOKENIZERS_PARALLELISM=false
export NCCL_P2P_DISABLE=0 # when set to 0, allows NCCL to use P2P communication for better performance (instead of using CPU for communication)
export NCCL_IB_DISABLE=0 # when set to 0, allows NCCL to use InfiniBand for better performance (instead of using TCP/IP for communication)
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1 # when set to 1, allows NCCL to handle errors asynchronously (can improve performance in some cases)

# set path to config file
CONFIG_PATH="/home/ads4015/ssl_project/configs/wu_clip_pretrain_config.yaml"

# clock job start time
export START_EPOCH="$(date +%s)"
echo "[INFO] Job runtime timer started at $(date -d @${START_EPOCH} '+%Y-%m-%d %H:%M:%S')"

# bridge slurm to torch env
export WORLD_SIZE=${SLURM_NTASKS} # total number of tasks (gpus) across all nodes

# allocator
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# run script
srun --label python /home/ads4015/ssl_project/src/wu_clip_pretrain.py --config $CONFIG_PATH



# indicate completion
echo 'LSM pretraining complete'













