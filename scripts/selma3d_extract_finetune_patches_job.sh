#!/bin/bash
#SBATCH --job-name=extract_patches
#SBATCH --output=logs/extract_patches_%j.out
#SBATCH --error=logs/extract_patches_%j.err
#SBATCH --partition=minilab-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00


# activate conda environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1


# indicate starting
echo "Starting patch extraction..."


# run script
python /home/ads4015/ssl_project/src/selma3d_extract_finetune_patches.py


# indicate completion
echo 'Finishing patch extraction.'

