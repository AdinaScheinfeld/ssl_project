#!/bin/bash
#SBATCH --job-name=split_transform
#SBATCH --output=logs/split_transform_%j.out
#SBATCH --error=logs/split_transform_%j.err
#SBATCH --partition=minilab-cpu
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=500G


# indicate starting
echo "Beginning split transform..."


# load modules
module load anaconda3/2022.10-34zllqw
source activate monai-env1

python /home/ads4015/ssl_project/preprocess_patches/src/selma3d_split_and_transform.py


# indicate completion
echo "Split transform complete..."


























