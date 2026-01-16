#!/bin/bash
#SBATCH --job-name=finetune_infer_cv_array
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_snowy_sweep/logs/finetune_infer_cv_array_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_snowy_sweep/logs/finetune_infer_cv_array_%A_%a.err
#SBATCH --partition=sablab-gpu
#SBATCH --account=sablab
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=3-00:00:00

# finetune_and_inference_cv_array_job_unet.sh - Script to finetune a pretrained model and perform inference on a dataset split into training and evaluation sets, using SLURM array jobs.
# one array task = one (SUBTYPE, K, FID, FJSON) job.

set -euo pipefail


# ---- temp dir (critical: avoid NFS .nfs* cleanup errors) ----
export SCRATCH_ROOT=/midtier/paetzollab/scratch/ads4015

# Prefer node-local temp provided by Slurm, otherwise use /tmp on the node.
# Include ARRAY_TASK_ID so each task is isolated even within the same job allocation.
export TMPDIR="${SLURM_TMPDIR:-/tmp/$USER/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}}"
export TMP="$TMPDIR"
export TEMP="$TMPDIR"
mkdir -p "$TMPDIR"

echo "[INFO] TMPDIR=$TMPDIR"


# parse args
TASKS_FILE="${1:?usage: $0 TASKS_FILE JOB_PREFIX}"
JOB_PREFIX="${2:?usage: $0 TASKS_FILE JOB_PREFIX}"

# read the line corresponding to this array index (0-based)
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS_FILE" || true)"
if [[ -z "${LINE}" ]]; then
  echo "[ERROR] No line for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID in $TASKS_FILE"
  exit 1
fi

read -r SUBTYPE K FID FJSON <<< "$LINE"

echo "[INFO] Task: SUBTYPE=$SUBTYPE  K=$K  FID=$FID"
echo "[INFO] Folds JSON: $FJSON"

# load conda env
module load anaconda3/2022.10-34zllqw
source activate monai-env2

# define constants
ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
CKPT_DIR="/midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_snowy_sweep/checkpoints" # output dir for finetune checkpoints
CKPT="/midtier/paetzollab/scratch/ads4015/pretrain_sweep_unet/checkpoints/qsrufm3e/all_datasets_clip_pretrained_unet_best.ckpt" # Image+CLIP checkpoint
PRED_ROOT="/midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_snowy_sweep/preds" # output dir for preds

# pretty-name mapping for outputs
case "$SUBTYPE" in
  amyloid_plaque_patches) PRETTY_SUBTYPE="amyloid_plaque" ;;
  c_fos_positive_patches) PRETTY_SUBTYPE="c_fos_positive" ;;
  cell_nucleus_patches)   PRETTY_SUBTYPE="cell_nucleus" ;;
  vessels_patches)        PRETTY_SUBTYPE="vessels" ;;
  *) echo "[ERROR] Unknown subtype: $SUBTYPE"; exit 2 ;;
esac

# indicate starting
echo "[INFO] Starting finetune+infer for ${SUBTYPE} (K=${K}, FID=${FID})..."

# run finetuning and inference
python /home/ads4015/ssl_project/src/finetune_and_inference_split_unet.py \
  --root "$ROOT" \
  --subtypes "$SUBTYPE" \
  --ckpt_dir "$CKPT_DIR" \
  --pretrained_ckpt "$CKPT" \
  --val_percent 0.2 \
  --seed 100 \
  --batch_size 4 \
  --unet_channels 32,64,128,256,512 \
  --unet_strides 2,2,2,2 \
  --unet_num_res_units 2 \
  --unet_norm BATCH \
  --max_epochs 500 \
  --early_stopping_patience 45 \
  --freeze_encoder_epochs 10 \
  --encoder_lr_mult 0.05 \
  --loss_name dicece \
  --wandb_project selma3d_finetune_long \
  --num_workers 4 \
  --channel_substr ALL \
  --preds_root "$PRED_ROOT" \
  --preds_subtype "$PRETTY_SUBTYPE" \
  --folds_json "$FJSON" \
  --fold_id "$FID" \
  --train_limit "$K" \
  --infer_ckpt best


# optional: best-effort cleanup (ignore failure)
rm -rf "$TMPDIR" || true


# indicate done
echo "[INFO] Done: ${SUBTYPE} (K=${K}, FID=${FID})"



