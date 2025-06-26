#!/bin/bash
#SBATCH --job-name=get_patches
#SBATCH --output=logs/get_patches_%A_%a.out
#SBATCH --error=logs/get_patches_%A_%a.err
#SBATCH --partition=minilab-cpu
#SBATCH --array=14,22,27 # num samples - 1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G


# indicate starting
echo "Beginning patch extraction $SLURM_ARRAY_TASK_ID..."


# load modules
module load anaconda3/2022.10-34zllqw
source activate monai-env1

python /home/ads4015/ssl_project/preprocess_patches/src/get_patches.py --sample_index ${SLURM_ARRAY_TASK_ID}


# indicate completion
echo "Patch extraction complete."





















