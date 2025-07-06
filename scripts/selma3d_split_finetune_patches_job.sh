#!/bin/bash
#SBATCH --job-name=split_train_val
#SBATCH --output=logs/split_train_val_%j.out
#SBATCH --error=logs/split_train_val_%j.err
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=10:00:00

# activate conda environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1

echo "Starting the split script..."

# Run the script
python /home/ads4015/ssl_project/src/selma3d_split_finetune_patches.py \
  --input_root /midtier/paetzollab/scratch/ads4015/data_selma3d/lsm_fm_selma3d_finetune2 \
  --output_root /midtier/paetzollab/scratch/ads4015/data_selma3d/lsm_fm_selma3d_finetune_split \
  --val_frac 0.2 \
  --seed 100


  echo "Split script completed successfully."
