#!/bin/bash
#SBATCH --job-name=zeroshot_cls_embed
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --array=0-161
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_zeroshot/logs/zeroshot_classification_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_zeroshot/logs/zeroshot_classification_%A_%a.err

module load anaconda3/2022.10-34zllqw
source activate monai-env1

SCRIPT=/home/ads4015/ssl_project/zeroshot_inference/classification/zeroshot_inference_classification_embed_single_patch.py

# (Optional) ensure logs dir exists
mkdir -p /midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_zeroshot/logs

python ${SCRIPT} --task-id ${SLURM_ARRAY_TASK_ID}



