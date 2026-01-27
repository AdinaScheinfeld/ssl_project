#!/bin/bash
#SBATCH --job-name=cls_unet_arr
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_classification_bright_sweep_26/logs/cls_unet_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_classification_bright_sweep_26/logs/cls_unet_%A_%a.err
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00

# *** USE THIS FILE FOR UNET IMAGE-ONLY FINETUNING ***

set -euo pipefail

# indicate starting
echo "[INFO] Starting classification finetune/eval UNet job array on $(date)..."

TASKS_FILE="${1:?usage: $0 TASKS_FILE}"
LINE="$(sed -n "$((SLURM_ARRAY_TASK_ID+1))p" "$TASKS_FILE" || true)"
if [[ -z "${LINE}" ]]; then
  echo "[ERROR] No line for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID in $TASKS_FILE"
  exit 1
fi

read -r TR FID FJSON <<< "$LINE"
echo "[INFO] Task: TR=$TR FID=$FID"
echo "[INFO] Folds JSON: $FJSON"

module load anaconda3/2022.10-34zllqw
source activate monai-env2

# -----------------------
# outputs (your requested structure)
# -----------------------
OUTROOT="/midtier/paetzollab/scratch/ads4015/temp_selma_classification_bright_sweep_26"
CKPT_DIR="${OUTROOT}/checkpoints/finetune_cls"
METRICS_ROOT="${OUTROOT}/cls_metrics"
mkdir -p "$CKPT_DIR" "$METRICS_ROOT" "${OUTROOT}/logs"

# -----------------------
# data root
# -----------------------
ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"

# -----------------------
# pretrained UNet backbone ckpt
# -----------------------
PRETRAINED_UNET="/midtier/paetzollab/scratch/ads4015/pretrain_sweep_unet_no_text/checkpoints/t3cw8qnt/all_datasets_clip_pretrained_unet_best.ckpt"

# -----------------------
# extra classes (must match folds creation!)
# (we don't need to pass these if fold_json already contains the paths,
#  but your finetune script uses them for discovery when fold_json is absent.
#  Keeping them here is harmless and keeps parity.)
# -----------------------
MESO_ROOT="/midtier/paetzollab/scratch/ads4015/all_mesospim_patches"
ALLEN_ROOT="/midtier/paetzollab/scratch/ads4015/all_allen_human_patches"
EXTRA_CLASS_GLOBS=(
  "VIP_ASLM_off:${MESO_ROOT}/*VIP_ASLM_off*.nii*"
  "VIP_ASLM_on:${MESO_ROOT}/*VIP_ASLM_on*.nii*"
  "TPH2:${MESO_ROOT}/*TPH2*.nii*"
  "stain-CR:${ALLEN_ROOT}/*_cr_*ps96*.nii*"
  "stain-LEC:${ALLEN_ROOT}/*_lec_*ps96*.nii*"
  "stain-NN:${ALLEN_ROOT}/*_nn_*ps96*.nii*"
  "stain-NPY:${ALLEN_ROOT}/*_npy_*ps96*.nii*"
  "stain-YO:${ALLEN_ROOT}/*_yo_*ps96*.nii*"
)
EXTRA_CLASS_GLOBS_ARGS=()
for spec in "${EXTRA_CLASS_GLOBS[@]}"; do
  EXTRA_CLASS_GLOBS_ARGS+=(--extra_class_globs "$spec")
done

# -----------------------
# run knobs
# -----------------------
INIT_MODE="pretrained"   # or "random"
SEED=100
BATCH_SIZE=8
MAX_EPOCHS=200
NUM_WORKERS=4
VAL_PERCENT=0.2

CLASS_WEIGHTING="inverse_freq"
BETA=0.9999

# UNet arch must match the UNet you pretrained (edit if needed)
UNET_CHANNELS="32,64,128,256,512"
UNET_STRIDES="2,2,2,1"
UNET_NUM_RES_UNITS=2
UNET_NORM="BATCH"

# -----------------------
# run classification
# IMPORTANT: train_per_class is baked into the fold JSON already.
# We include TR only in tags indirectly (via fold_json filename) and logs.
# -----------------------
srun python /home/ads4015/ssl_project/src/finetune_and_eval_classification_split_unet.py \
  --root_dir "$ROOT" \
  "${EXTRA_CLASS_GLOBS_ARGS[@]}" \
  --ckpt_dir "$CKPT_DIR" \
  --metrics_root "$METRICS_ROOT" \
  --seed "$SEED" \
  --batch_size "$BATCH_SIZE" \
  --lr 0.001 \
  --weight_decay 0.001 \
  --max_epochs "$MAX_EPOCHS" \
  --num_workers "$NUM_WORKERS" \
  --val_percent "$VAL_PERCENT" \
  --channel_substr ALL \
  --fold_json "$FJSON" \
  --fold_id "$FID" \
  --wandb_project selma3d_classify_unet_peach_sweep_25 \
  --class_weighting "$CLASS_WEIGHTING" --beta "$BETA" \
  --in_channels 1 \
  --unet_channels "$UNET_CHANNELS" \
  --unet_strides "$UNET_STRIDES" \
  --unet_num_res_units "$UNET_NUM_RES_UNITS" \
  --unet_norm "$UNET_NORM" \
  $( [[ "$INIT_MODE" == "pretrained" ]] && echo --init_mode pretrained --pretrained_ckpt "$PRETRAINED_UNET" --freeze_encoder_epochs 5 || echo --init_mode random )


# indicate done
echo "[INFO] Finished classification finetune/eval UNet job array task TR=$TR FID=$FID on $(date)"

