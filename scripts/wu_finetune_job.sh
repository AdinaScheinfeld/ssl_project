#!/bin/bash
#SBATCH --job-name=finetune_lsm
#SBATCH --output=logs/finetune_lsm_%j.out
#SBATCH --error=logs/finetune_lsm_%j.err
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# indicate starting
echo "Starting the finetuning script..."


# activate conda environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# define path to config file
CONFIG_FILE="/home/ads4015/ssl_project/configs/wu_finetune_config.yaml"


# run finetuning script
python /home/ads4015/ssl_project/src/wu_finetune.py --config "$CONFIG_FILE"


# indicate completion
echo "Finetuning script completed."

