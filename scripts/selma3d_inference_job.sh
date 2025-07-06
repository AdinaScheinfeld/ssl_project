#!/bin/bash
#SBATCH --job-name=lsm_inference
#SBATCH --output=logs/lsm_inference_%j.out
#SBATCH --error=logs/lsm_inference_%j.err
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00

# activate conda environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1

echo "Starting the inference script..."


# Run inference on folder
python /home/ads4015/ssl_project/src/selma3d_inference.py \
  --input_path /midtier/paetzollab/scratch/ads4015/data_selma3d/unlabeled_val \
  --checkpoint /home/ads4015/ssl_project/lsm_finetune/vtmxkihh/checkpoints/best.ckpt \
  --output_dir /midtier/paetzollab/scratch/ads4015/data_selma3d/unlabeled_val_preds


echo "Inference script completed successfully."




