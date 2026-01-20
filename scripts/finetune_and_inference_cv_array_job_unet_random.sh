#!/bin/bash
#SBATCH --job-name=ft_infer_unet_array
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_unet_random2/logs/ft_infer_unet_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_unet_random2/logs/ft_infer_unet_%A_%a.err
#SBATCH --partition=sablab-gpu
#SBATCH --account=sablab
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=3-00:00:00

set -euo pipefail

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

# ---- temp dir (avoid NFS .nfs cleanup issues) ----
export SCRATCH_ROOT=/midtier/paetzollab/scratch/ads4015
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"
echo "[INFO] TMPDIR=$TMPDIR"

module load anaconda3/2022.10-34zllqw
source activate monai-env2

ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"

OUTROOT="/midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_unet_random2"
mkdir -p "${OUTROOT}/logs" "${OUTROOT}/checkpoints" "${OUTROOT}/preds" "${OUTROOT}/cv_folds"
    
python /home/ads4015/ssl_project/src/finetune_and_inference_split_unet.py \
  --root "$ROOT" \
  --subtypes "$SUBTYPE" \
  --out_root "$OUTROOT" \
  --channel_substr ALL \
  --folds_json "$FJSON" \
  --fold_id "$FID" \
  --train_limit "$K" \
  --init random \
  --wandb_project selma3d_unet_ft_infer_random \
  --seed 100 \
  --batch_size 2 \
  --num_workers 4 \
  --val_percent 0.2 \
  --infer_ckpt best \
  --lr 0.0003 \
  --weight_decay 0.00001 \
  --max_epochs 600 \
  --early_stopping_patience 200 \
  --freeze_encoder_epochs 0 \
  --encoder_lr_mult 0.2 \
  --freeze_bn_stats 1 \
  --loss_name dicefocal \
  --unet_channels "32,64,128,256" \
  --unet_strides "2,2,2,1" \
  --unet_num_res_units "2" \
  --unet_norm "BATCH"

rm -rf "$TMPDIR" || true
echo "[INFO] Done."





