#!/bin/bash
#SBATCH --job-name=clip_pretrain
#SBATCH --output=logs/clip_pretrain_%j.out
#SBATCH --error=logs/clip_pretrain_%j.err
#SBATCH --time=10:00:00
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --ntasks=1


# indicate starting
echo "Starting LSM pretraining..."



# activate conda environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# wandb login
export WANDB_API_KEY=9778703390de02a48bdea1415c3c36c3bae408c0
wandb login $WANDB_API_KEY

export TOKENIZERS_PARALLELISM=false

# set path to config file
CONFIG_PATH="/home/ads4015/ssl_project/configs/wu_clip_pretrain_config.yaml"

# run script
python /home/ads4015/ssl_project/src/wu_clip_pretrain.py --config $CONFIG_PATH



# indicate completion
echo 'LSM pretraining complete'













