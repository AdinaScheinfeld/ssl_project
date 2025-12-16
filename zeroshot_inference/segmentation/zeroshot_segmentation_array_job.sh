#!/bin/bash
#SBATCH --job-name=selma3d_zeroshot_array
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-87
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_zeroshot/logs/zeroshot_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_zeroshot/logs/zeroshot_%A_%a.err

# -----------------------------
# Environment
# -----------------------------
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# -----------------------------
# Run
# -----------------------------
python /home/ads4015/ssl_project/zeroshot_inference/segmentation/zeroshot_segmentation_single_patch.py \
    --task-id ${SLURM_ARRAY_TASK_ID}
