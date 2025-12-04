#!/bin/bash
#SBATCH --job-name=nnunet_train_splits
#SBATCH --output=logs/nnunet_train_%A_%a.out
#SBATCH --error=logs/nnunet_train_%A_%a.err
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --array=1000-1252

echo "=== TRAINING Dataset ${SLURM_ARRAY_TASK_ID} ==="
echo "Start time: $(date)"

SPLIT_ID=${SLURM_ARRAY_TASK_ID}

# Environment variables for nnU-Net paths
export nnUNet_raw="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/cross_val/raw"
export nnUNet_preprocessed="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/cross_val/preprocessed"
export nnUNet_results="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/cross_val/results"

export nnUNet_use_progressive_dataloader="False"

# Load environment
module load anaconda3/2022.10-34zllqw
source activate nnunet2-env1

# Train with fold = -1  (use ALL training data)
nnUNetv2_train $SPLIT_ID 3d_fullres all --npz

echo "=== COMPLETED Dataset ${SPLIT_ID} at $(date) ==="

