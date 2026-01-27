#!/bin/bash
#SBATCH --job-name=finetune_inpaint_unet
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_inpaint_preds_bright_sweep_26_ntc/logs/finetune_inpaint_unet_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_inpaint_preds_bright_sweep_26_ntc/logs/finetune_inpaint_unet_%A_%a.err
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00


# *** USE THIS FILE FOR UNET IMAGE+CLIP ONLY ***
# /home/ads4015/ssl_project/scripts/finetune_inpaint_split_job_array1_unet.sh

set -euo pipefail
export TOKENIZERS_PARALLELISM=false

TASKS_FILE="${1:?usage: $0 TASKS_FILE JOB_PREFIX}"
JOB_PREFIX="${2:?usage: $0 TASKS_FILE JOB_PREFIX}"

LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS_FILE" || true)"
if [[ -z "${LINE}" ]]; then
  echo "[ERROR] No line for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID in $TASKS_FILE"
  exit 1
fi
read -r SUBTYPE K FID FJSON <<< "$LINE"

echo "[INFO] Task: SUBTYPE=$SUBTYPE  K=$K  FID=$FID"
echo "[INFO] Folds JSON: $FJSON"

module load anaconda3/2022.10-34zllqw
source activate monai-env2

ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
OUTROOT="/midtier/paetzollab/scratch/ads4015/temp_selma_inpaint_preds_bright_sweep_26_ntc"
CKPT_DIR="${OUTROOT}/checkpoints"
PRED_ROOT="${OUTROOT}/preds"
PRETRAIN_CKPT="/midtier/paetzollab/scratch/ads4015/pretrain_sweep_unet_no_text/checkpoints/t3cw8qnt/all_datasets_clip_pretrained_unet_best.ckpt"

mkdir -p "$OUTROOT"/{checkpoints,cv_folds,logs,preds}

# match your per-subtype mask choices from autumn sweep
case "$SUBTYPE" in
  amyloid_plaque_patches)
    MASK_MODE='fixed_size'
    MASK_RATIO=0.30
    MASK_RATIO_TEST=0.30
    MASK_FIXED_SIZE="24,24,12"
    MASK_FIXED_SIZE_TEST="24,24,12"
    NUM_MASK_BLOCKS=2
    NUM_MASK_BLOCKS_TEST=2
    ;;
  c_fos_positive_patches)
    MASK_MODE='fixed_size'
    MASK_RATIO=0.40
    MASK_RATIO_TEST=0.40
    MASK_FIXED_SIZE="12"
    MASK_FIXED_SIZE_TEST="12"
    NUM_MASK_BLOCKS=4
    NUM_MASK_BLOCKS_TEST=4
    ;;
  cell_nucleus_patches)
    MASK_MODE='fixed_size'
    MASK_RATIO=0.50
    MASK_RATIO_TEST=0.50
    MASK_FIXED_SIZE="18"
    MASK_FIXED_SIZE_TEST="18"
    NUM_MASK_BLOCKS=5
    NUM_MASK_BLOCKS_TEST=5
    ;;
  vessels_patches)
    MASK_MODE='fixed_size'
    MASK_RATIO=0.60
    MASK_RATIO_TEST=0.60
    MASK_FIXED_SIZE="12,12,48"
    MASK_FIXED_SIZE_TEST="12,12,48"
    NUM_MASK_BLOCKS=2
    NUM_MASK_BLOCKS_TEST=2
    ;;
  *) echo "[ERROR] Unknown subtype: $SUBTYPE"; exit 2 ;;
esac

echo "[INFO] Starting UNet inpainting finetune for ${SUBTYPE} (K=${K}, FID=${FID})..."

# # use when text conditioning is enabled
# python /home/ads4015/ssl_project/src/finetune_inpaint_split_unet.py \
#   --data_root "$ROOT" \
#   --subtypes "$SUBTYPE" \
#   --ckpt_dir "$CKPT_DIR" \
#   --preds_root "$PRED_ROOT" \
#   --pretrained_ckpt_path "$PRETRAIN_CKPT" \
#   --folds_json "$FJSON" \
#   --fold_id "$FID" \
#   --train_limit "$K" \
#   --val_percent 0.2 \
#   --seed 100 \
#   --batch_size 2 \
#   --max_epochs 500 \
#   --early_stopping_patience 50 \
#   --freeze_encoder_epochs 5 \
#   --encoder_lr_mult 0.05 \
#   --l1_weight_masked 1.0 \
#   --l1_weight_global 0.1 \
#   --weight_decay 1e-5 \
#   --wandb_project selma3d_inpaint_unet_bright_sweep_26 \
#   --num_workers 1 \
#   --channel_substr ALL \
#   --mask_mode "$MASK_MODE" \
#   --mask_ratio "$MASK_RATIO" \
#   --mask_ratio_test "$MASK_RATIO_TEST" \
#   --mask_fixed_size "$MASK_FIXED_SIZE" \
#   --mask_fixed_size_test "$MASK_FIXED_SIZE_TEST" \
#   --num_mask_blocks "$NUM_MASK_BLOCKS" \
#   --num_mask_blocks_test "$NUM_MASK_BLOCKS_TEST" \
#   --text_backend clip \
#   --clip_ckpt "$PRETRAIN_CKPT" \
#   --unet_channels "32,64,128,256,512" \
#   --unet_strides "2,2,2,1" \
#   --unet_num_res_units 2 \
#   --unet_norm "BATCH"


# use when text conditioning is disabled
python /home/ads4015/ssl_project/src/finetune_inpaint_split_unet.py \
  --data_root "$ROOT" \
  --subtypes "$SUBTYPE" \
  --ckpt_dir "$CKPT_DIR" \
  --preds_root "$PRED_ROOT" \
  --pretrained_ckpt_path "$PRETRAIN_CKPT" \
  --folds_json "$FJSON" \
  --fold_id "$FID" \
  --train_limit "$K" \
  --val_percent 0.2 \
  --seed 100 \
  --batch_size 2 \
  --max_epochs 500 \
  --early_stopping_patience 50 \
  --freeze_encoder_epochs 0 \
  --encoder_lr_mult 0.1 \
  --l1_weight_masked 1.0 \
  --l1_weight_global 0.1 \
  --lr 0.001 \
  --weight_decay 0.001 \
  --wandb_project selma3d_inpaint_unet_bright_sweep_26_ntc \
  --num_workers 1 \
  --channel_substr ALL \
  --mask_mode "$MASK_MODE" \
  --mask_ratio "$MASK_RATIO" \
  --mask_ratio_test "$MASK_RATIO_TEST" \
  --mask_fixed_size "$MASK_FIXED_SIZE" \
  --mask_fixed_size_test "$MASK_FIXED_SIZE_TEST" \
  --num_mask_blocks "$NUM_MASK_BLOCKS" \
  --num_mask_blocks_test "$NUM_MASK_BLOCKS_TEST" \
  --disable_text_cond \
  --unet_channels "32,64,128,256,512" \
  --unet_strides "2,2,2,1" \
  --unet_num_res_units 2 \
  --unet_norm "BATCH"

echo "[INFO] Done: ${SUBTYPE} (K=${K}, FID=${FID})"
