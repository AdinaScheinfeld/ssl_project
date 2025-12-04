#!/bin/bash
#SBATCH --job-name=nnunet_plan_all
#SBATCH --output=logs/nnunet_plan_%A_%a.out
#SBATCH --error=logs/nnunet_plan_%A_%a.err
#SBATCH --partition=sablab-cpu
#SBATCH --account=sablab
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=1000-1252    # adjust to cover dataset ID range

# -----------------------------
# Environment Setup
# -----------------------------
module load anaconda3/2022.10-34zllqw
source activate nnunet2-env1

export nnUNet_raw="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/cross_val/raw"
export nnUNet_preprocessed="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/cross_val/preprocessed"
export nnUNet_results="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/cross_val/results"

# Dataset ID from SLURM array
ID=$SLURM_ARRAY_TASK_ID

DATASET_DIR="${nnUNet_raw}/Dataset${ID}"

if [[ ! -d "$DATASET_DIR" ]]; then
    echo "Dataset${ID} does not exist. Skipping."
    exit 0
fi

echo "==== Running planning & preprocessing for Dataset${ID} ===="

# -----------------------------
# Run Planning & Preprocessing
# -----------------------------
nnUNetv2_plan_and_preprocess \
    -d $ID \
    -c 3d_fullres \
    --verify_dataset_integrity

echo "==== Finished Dataset${ID} ===="
