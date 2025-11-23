#!/bin/bash
#SBATCH --job-name=finetune_deblur
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_deblur_preds_autumn_sweep_27/logs/finetune_deblur_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_deblur_preds_autumn_sweep_27/logs/finetune_deblur_%A_%a.err
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00

# /home/ads4015/ssl_project/scripts/finetune_deblur_split_job_array.sh - SLURM array script: one task = one (SUBTYPE, K, FID, FJSON) deblurring job.


set -euo pipefail
export TOKENIZERS_PARALLELISM=false

# indicate starting
echo "[INFO] Starting finetune_deblur job for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID"

TASKS_FILE="${1:?usage: $0 TASKS_FILE JOB_PREFIX}"
JOB_PREFIX="${2:?usage: $0 TASKS_FILE JOB_PREFIX}"

# read the line corresponding to this array index
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS_FILE" || true)"
if [[ -z "$LINE" ]]; then
  echo "[ERROR] No line for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID in $TASKS_FILE" >&2
  exit 1
fi

read -r SUBTYPE K FID FJSON <<< "$LINE"

echo "[INFO] Task: SUBTYPE=$SUBTYPE  K=$K  FID=$FID"
echo "[INFO] Folds JSON: $FJSON"

# load env
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# paths
ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches" # sharp patches
BLUR_ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches_blurred" # blurred patches
CKPT_DIR="/midtier/paetzollab/scratch/ads4015/temp_selma_deblur_preds_autumn_sweep_27/checkpoints" # location to save finetune checkpoints
PRED_ROOT="/midtier/paetzollab/scratch/ads4015/temp_selma_deblur_preds_autumn_sweep_27/preds" # location to save predictions

# existing pretrained checkpoint
CKPT_PRETR="/midtier/paetzollab/scratch/ads4015/checkpoints/autumn_sweep_27/all_datasets_clip_pretrained-updated-epochepoch=354-val-reportval_loss_report=0.0968-stepstep=20590.ckpt"

# customize per-subtype hyperparameters if desired
case "$SUBTYPE" in
  amyloid_plaque_patches)
    ;;
  c_fos_positive_patches)
    ;;
  cell_nucleus_patches)
    ;;
  vessels_patches)
    ;;
  *)
    echo "[WARN] Unknown subtype: $SUBTYPE, using default feature_size=24" >&2
    ;;
esac

# prepare output folders
mkdir -p "$CKPT_DIR" "$PRED_ROOT"

echo "[INFO] Starting deblur finetune for ${SUBTYPE} (K=${K}, FID=${FID}) ..."

# run finetune script
# feature size is 24 for image only, 36 for image+text
python /home/ads4015/ssl_project/src/finetune_deblur_split.py \
  --data_root "$ROOT" \
  --blur_root "$BLUR_ROOT" \
  --subtypes "$SUBTYPE" \
  --ckpt_dir "$CKPT_DIR" \
  --preds_root "$PRED_ROOT" \
  --pretrained_ckpt_path "$CKPT_PRETR" \
  --val_fraction 0.2 \
  --seed 100 \
  --batch_size 2 \
  --feature_size 36 \
  --max_epochs 500 \
  --freeze_encoder_epochs 5 \
  --encoder_lr_mult 0.05 \
  --wandb_project selma3d_deblur \
  --num_workers 1 \
  --channel_substr ALL \
  --folds_json "$FJSON" \
  --fold_id "$FID" \
  --train_limit "$K"

# indicate completion
echo "[INFO] Done: ${SUBTYPE} (K=${K}, FID=${FID})"























