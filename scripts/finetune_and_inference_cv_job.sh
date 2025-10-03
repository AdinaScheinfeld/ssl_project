#!/bin/bash
#SBATCH --job-name=finetune_infer_cv
#SBATCH --output=logs/finetune_infer_cv_%j.out
#SBATCH --error=logs/finetune_infer_cv_%j.err
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00


# finetune_and_inference_cv_job.sh - Script to finetune a pretrained model and perform inference on a dataset split into training and evaluation sets.

# indicate starting
echo "Starting finetune and inference..."


# set up environment
set -euo pipefail
module load anaconda3/2022.10-34zllqw
source activate monai-env1

SUBTYPE="${1:?need subtype}"
FOLD_ID="${2:?need fold id}"
FOLDS_JSON="${3:?need folds json}"

# set variables
ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
CKPT_DIR="/home/ads4015/ssl_project/checkpoints"
CKPT="/home/ads4015/ssl_project/checkpoints/all_datasets_clip_pretrained-v12.ckpt"
PRED_ROOT="/ministorage/adina/selma_segmentation_preds"

python /home/ads4015/ssl_project/src/finetune_and_inference_split.py \
  --root "$ROOT" \
  --subtypes "$SUBTYPE" \
  --ckpt_dir "$CKPT_DIR" \
  --pretrained_ckpt "$CKPT" \
  --val_percent 0.2 \
  --seed 100 --batch_size 4 --feature_size 24 --max_epochs 1000 \
  --freeze_encoder_epochs 5 --encoder_lr_mult 0.05 --loss_name dicece \
  --wandb_project selma3d_finetune --num_workers 4 \
  --channel_substr ALL \
  --preds_root "$PRED_ROOT" \
  --folds_json "$FOLDS_JSON" \
  --fold_id "$FOLD_ID"


# indicate ending
echo "Ending finetune and inference."








