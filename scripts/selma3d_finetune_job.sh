#!/bin/bash
#SBATCH --job-name=finetune_lsm
#SBATCH --output=logs/finetune_lsm_%j.out
#SBATCH --error=logs/finetune_lsm_%j.err
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=48:00:00

# activate conda environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1


echo "Starting the finetuning script..."


# Run finetuning
python /home/ads4015/ssl_project/src/selma3d_finetune.py \
  --split_root /midtier/paetzollab/scratch/ads4015/data_selma3d/lsm_fm_selma3d_finetune_split \
  --pretrained_ckpt /home/ads4015/ssl_project/ibot-pretrain-lsm/n5q7rkh3/checkpoints/best-val-loss.ckpt \
  --batch_size 2 \
  --lr 1e-4 \
  --max_epochs 100 \
  --project_name lsm_finetune


  echo "Finetuning script completed successfully."
