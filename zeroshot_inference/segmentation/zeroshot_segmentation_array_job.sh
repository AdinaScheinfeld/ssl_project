#!/bin/bash
#SBATCH --job-name=zeroshot_seg
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_zeroshot/logs/%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_zeroshot/logs/%A_%a.err

# /home/ads4015/ssl_project/zeroshot_inference/segmentation/zeroshot_segmentation_array_job.sh - SLURM array job script for zero-shot segmentation inference.

set -euo pipefail

TASKS=$1
IDX=$SLURM_ARRAY_TASK_ID

IMG=$(sed -n "$((IDX+1))p" "$TASKS")

if [[ -z "$IMG" ]]; then
  echo "[ERROR] No image for index $IDX"
  exit 1
fi

echo "[INFO] Running zeroshot inference for $IMG"

module load anaconda3/2022.10-34zllqw
source activate monai-env1

python /home/ads4015/ssl_project/zeroshot_inference/segmentation/zeroshot_segmentation_single_patch.py \
  --image "$IMG" \
  --out_root /midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_zeroshot \
  --clip_ckpt /midtier/paetzollab/scratch/ads4015/checkpoints/autumn_sweep_27/all_datasets_clip_pretrained-updated-epochepoch=354-val-reportval_loss_report=0.0968-stepstep=20590.ckpt \
  --imgonly_ckpt /ministorage/adina/pretrain_sweep_no_clip/checkpoints/r605gzgj/all_datasets_pretrained_no_clip-epochepoch=183-valval_loss=0.0201-stepstep=10672.ckpt
