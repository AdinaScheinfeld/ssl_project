#!/bin/bash
#SBATCH --job-name=nnunet_predict
#SBATCH --output=logs/nnunet_predict_%A_%a.out
#SBATCH --error=logs/nnunet_predict_%A_%a.err
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --array=1000-1252


# indicate starting
echo "=== INFERENCE for Dataset ${SLURM_ARRAY_TASK_ID} ==="
echo "Start time: $(date)"

DATASET_ID=${SLURM_ARRAY_TASK_ID}

###############################################
# Set nnU-Net paths (same as training)
###############################################
export nnUNet_raw="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/cross_val/raw"
export nnUNet_preprocessed="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/cross_val/preprocessed"
export nnUNet_results="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/cross_val/results"

###############################################
# Load environment
###############################################
module load anaconda3/2022.10-34zllqw
source activate nnunet2-env1

###############################################
# INPUT AND OUTPUT PATHS
###############################################
IN_DIR="${nnUNet_raw}/Dataset${DATASET_ID}/imagesTs"
OUT_DIR="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/cross_val/preds/Dataset${DATASET_ID}"
mkdir -p "$OUT_DIR"

###############################################
# RUN INFERENCE using fold 0 only
###############################################
nnUNetv2_predict \
    -d ${DATASET_ID} \
    -i "$IN_DIR" \
    -o "$OUT_DIR" \
    -c 3d_fullres \
    -f all


# indicate completion
echo "=== COMPLETED inference for Dataset ${DATASET_ID} at $(date) ==="





