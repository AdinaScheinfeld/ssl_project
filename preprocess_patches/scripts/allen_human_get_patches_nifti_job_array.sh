#!/bin/bash
#SBATCH --job-name=ah_get_patches
#SBATCH --output=logs/ah_get_patches_%A_%a.out
#SBATCH --error=logs/ah_get_patches_%A_%a.err
#SBATCH --partition=minilab-cpu
#SBATCH --array=0-47 # num samples - 1
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G


# allen_human_get_patches_job_array.sh - slurm job to get allen_human patches


# indicate starting
echo "Beginning patch extraction $SLURM_ARRAY_TASK_ID..."


# load modules
module load anaconda3/2022.10-34zllqw
source activate monai-env1

python /home/ads4015/ssl_project/preprocess_patches/src/allen_human_get_patches_nifti.py \
    --sample_index ${SLURM_ARRAY_TASK_ID} \
    --num_patches 1 \
    --min_fg 0.05 \
    --seed $SLURM_ARRAY_TASK_ID \
    --output_dir /midtier/paetzollab/scratch/ads4015/all_allen_human_patches \
    --patch_size 96


# indicate completion
echo "Patch extraction complete."





















