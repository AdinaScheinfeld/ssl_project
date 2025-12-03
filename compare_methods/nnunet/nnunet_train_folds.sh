#!/bin/bash
#SBATCH --job-name=nnunet_train
#SBATCH --output=logs/nnunet_train_%A_%a.out
#SBATCH --error=logs/nnunet_train_%A_%a.err
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-4

# indicate starting
echo "Starting nnUNet training for fold ${SLURM_ARRAY_TASK_ID} at $(date)"


# get fold ID
FOLD=$SLURM_ARRAY_TASK_ID

# set environment variables
export nnUNet_raw="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/nnUNet_preprocessed"
export nnUNet_results="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/nnUNet_results"

# load environment
module load anaconda3/2022.10-34zllqw
source activate nnunet2-env1

# run training
nnUNetv2_train 1 3d_fullres $FOLD --npz


# indicate ending
echo "Finished nnUNet training for fold ${SLURM_ARRAY_TASK_ID} at $(date)"





