#!/bin/bash
#SBATCH --job-name=finetune_deblur_unet
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_deblur_preds_super_sweep2_v2/logs/finetune_deblur_unet_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_deblur_preds_super_sweep2_v2/logs/finetune_deblur_unet_%A_%a.err
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00


# *** USE THIS SCRIPT FOR IMAGE+CLIP UNET DEBLURRING ONLY ***

set -euo pipefail
export TOKENIZERS_PARALLELISM=false

echo "[INFO] Starting finetune_deblur_unet job for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

TASKS_FILE="${1:?usage: $0 TASKS_FILE JOB_PREFIX}"
JOB_PREFIX="${2:?usage: $0 TASKS_FILE JOB_PREFIX}"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS_FILE" || true)"
if [[ -z "$LINE" ]]; then
  echo "[ERROR] No line for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID in $TASKS_FILE" >&2
  exit 1
fi

read -r SUBTYPE K FID FJSON <<< "$LINE"
echo "[INFO] Task: SUBTYPE=$SUBTYPE  K=$K  FID=$FID"
echo "[INFO] Folds JSON: $FJSON"

module load anaconda3/2022.10-34zllqw
source activate monai-env2

ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
BLUR_ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches_blurred2"

OUTROOT="/midtier/paetzollab/scratch/ads4015/temp_selma_deblur_preds_super_sweep2_v2"
CKPT_DIR="${OUTROOT}/checkpoints"
PRED_ROOT="${OUTROOT}/preds"

# YOUR pretrained UNet checkpoint
CKPT_PRETR="/midtier/paetzollab/scratch/ads4015/pretrain_sweep_unet/checkpoints/aaqkna34/all_datasets_clip_pretrained_unet_best.ckpt"

mkdir -p "$OUTROOT/logs" "$OUTROOT/cv_folds" "$CKPT_DIR" "$PRED_ROOT"

echo "[INFO] Starting UNet deblur finetune for ${SUBTYPE} (K=${K}, FID=${FID}) ..."

python /home/ads4015/ssl_project/src/finetune_deblur_split_unet.py \
  --data_root "$ROOT" \
  --blur_root "$BLUR_ROOT" \
  --subtypes "$SUBTYPE" \
  --ckpt_dir "$CKPT_DIR" \
  --preds_root "$PRED_ROOT" \
  --pretrained_ckpt_path "$CKPT_PRETR" \
  --val_fraction 0.2 \
  --seed 100 \
  --batch_size 2 \
  --max_epochs 500 \
  --early_stopping_patience 50 \
  --freeze_encoder_epochs 0 \
  --encoder_lr_mult 1.0 \
  --weight_decay 1e-5 \
  --freeze_bn_stats 0 \
  --wandb_project selma3d_deblur_unet \
  --num_workers 1 \
  --channel_substr ALL \
  --folds_json "$FJSON" \
  --fold_id "$FID" \
  --train_limit "$K" \
  --unet_channels "32,64,128,256,512" \
  --unet_strides "2,2,2,1" \
  --unet_num_res_units 2 \
  --unet_norm BATCH

echo "[INFO] Done: ${SUBTYPE} (K=${K}, FID=${FID})"

