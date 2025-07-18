#!/bin/bash
#SBATCH --job-name=finetune_split
#SBATCH --output=logs/finetune_split_%j.out
#SBATCH --error=logs/finetune_split_%j.err
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00


# indicate starting
echo "Starting the split script..."


# activate conda environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# Run the script
python /home/ads4015/ssl_project/src/selma3d_split_finetune_patches.py \
  --input_root /midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches \
  --output_root /midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches_split \
  --val_frac 0.2 \
  --seed 100


# indicate completion
echo "Split script completed successfully."
