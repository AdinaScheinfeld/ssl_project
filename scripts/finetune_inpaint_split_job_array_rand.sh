#!/bin/bash
#SBATCH --job-name=finetune_inpaint_rand
#SBATCH --output=/ministorage/adina/selma_inpaint_preds_rand_ntc/logs/finetune_inpaint_rand_ntc_%A_%a.out
#SBATCH --error=/ministorage/adina/selma_inpaint_preds_rand_ntc/logs/finetune_inpaint_rand_ntc_%A_%a.err
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# finetune_inpaint_split_job_array_rand.sh - SLURM array job script for inpainting finetuning


# indicate starting
echo "[INFO] Starting finetune_inpaint_rand job for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

set -euo pipefail
export TOKENIZERS_PARALLELISM=false  # disable tokenizer parallelism warnings

# one array task = one (SUBTYPE, K, FID, FJSON) inpainting job
TASKS_FILE="${1:?usage: $0 TASKS_FILE JOB_PREFIX}"
JOB_PREFIX="${2:?usage: $0 TASKS_FILE JOB_PREFIX}"

# read line for this task
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS_FILE" || true)"
if [[ -z "${LINE}" ]]; then
  echo "[ERROR] No line for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID in $TASKS_FILE"
  exit 1
fi
read -r SUBTYPE K FID FJSON <<< "$LINE"

echo "[INFO] Task: SUBTYPE=$SUBTYPE  K=$K  FID=$FID"
echo "[INFO] Folds JSON: $FJSON"

# load environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# set paths
ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches" # root dir of finetuning data
CKPT_DIR="/ministorage/adina/selma_inpaint_preds_rand_ntc/checkpoints" # dir to save finetuning checkpoints
# CKPT_PRETR="/ministorage/adina/pretrain_sweep_updated/checkpoints/kjvlrs45/all_datasets_clip_pretrained-updated-epochepoch=354-val-reportval_loss_report=0.0968-stepstep=20590.ckpt" # path to pretrained checkpoint
PRED_ROOT="/ministorage/adina/selma_inpaint_preds_rand_ntc/preds" # dir to save finetuning predictions

case "$SUBTYPE" in
  amyloid_plaque_patches) PRETTY_SUBTYPE="amyloid_plaque" ;;
  c_fos_positive_patches) PRETTY_SUBTYPE="c_fos_positive" ;;
  cell_nucleus_patches)   PRETTY_SUBTYPE="cell_nucleus" ;;
  vessels_patches)        PRETTY_SUBTYPE="vessels" ;;
  *) echo "[ERROR] Unknown subtype: $SUBTYPE"; exit 2 ;;
 esac

echo "[INFO] Starting inpainting finetune for ${SUBTYPE} (K=${K}, FID=${FID})..."

# # use for finetuning with pretraining
# # feature size is 24 for image only, 36 for image+text
# python /home/ads4015/ssl_project/src/finetune_inpaint_split.py \
#   --data_root "$ROOT" \
#   --subtypes "$SUBTYPE" \
#   --ckpt_dir "$CKPT_DIR" \
#   --pretrained_ckpt_path "$CKPT_PRETR" \
#   --val_percent 0.2 \
#   --seed 100 \
#   --batch_size 2 \
#   --feature_size 36 \
#   --max_epochs 500 \
#   --freeze_encoder_epochs 5 \
#   --encoder_lr_mult 0.05 \
#   --l1_weight_masked 1.0 \
#   --l1_weight_global 0.1 \
#   --wandb_project selma3d_inpaint \
#   --num_workers 1 \
#   --channel_substr ALL \
#   --preds_root "$PRED_ROOT" \
#   --folds_json "$FJSON" \
#   --fold_id "$FID" \
#   --train_limit "$K" \
#   --text_backend clip \
#   --clip_ckpt "$CKPT_PRETR" \
#   --mask_ratio 0.3 --mask_ratio_test 0.3

# use for random init (no pretraining)
python /home/ads4015/ssl_project/src/finetune_inpaint_split.py \
  --data_root "$ROOT" \
  --subtypes "$SUBTYPE" \
  --ckpt_dir "$CKPT_DIR" \
  --val_percent 0.2 \
  --seed 100 \
  --batch_size 2 \
  --feature_size 36 \
  --max_epochs 500 \
  --freeze_encoder_epochs 0 \
  --encoder_lr_mult 1.0 \
  --l1_weight_masked 1.0 \
  --l1_weight_global 0.1 \
  --wandb_project selma3d_inpaint_rand \
  --num_workers 1 \
  --channel_substr ALL \
  --preds_root "$PRED_ROOT" \
  --folds_json "$FJSON" \
  --fold_id "$FID" \
  --train_limit "$K" \
  --mask_ratio 0.3 --mask_ratio_test 0.3 \
  --disable_text_cond

# indicate done
echo "[INFO] Done: ${SUBTYPE} (K=${K}, FID=${FID})"









