#!/bin/bash
#SBATCH --job-name=finetune_infer_cv_rand
#SBATCH --output=logs/finetune_infer_cv_rand_%j.out
#SBATCH --error=logs/finetune_infer_cv_rand_%j.err
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --time=48:00:00


# finetune_and_inference_cv_rand_job.sh - Script to finetune a pretrained model and perform inference on a dataset split into training and evaluation sets.

# indicate starting
echo "Starting finetune and inference..."


# set up environment
set -euo pipefail
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# configs
ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
CKPT_DIR="/ministorage/adina/checkpoints/finetune_infer_sweep_rand" # directory to save finetune checkpoints
PRED_ROOT="/ministorage/adina/selma_segmentation_preds_rand/preds"

# args
SUBTYPE="${1:?need subtype}"
FOLD_ID="${2:?need fold id}"
FOLDS_JSON="${3:?need folds json}"

echo "[INFO] Subtype: $SUBTYPE, Fold ID: $FOLD_ID, Folds JSON: $FOLDS_JSON"

# map data subtype to pretty name for outputs
case "$SUBTYPE" in
  amyloid_plaque_patches) PRETTY_SUBTYPE="amyloid_plaque" ;;
  c_fos_positive_patches) PRETTY_SUBTYPE="c_fos_positive" ;;
  cell_nucleus_patches) PRETTY_SUBTYPE="cell_nucleus" ;;
  vessels_patches) PRETTY_SUBTYPE="vessels" ;;
  *) echo "Unknown subtype: $SUBTYPE"; exit 1 ;;
esac

# set seed
case "$SUBTYPE" in
  amyloid_plaque_patches) SUBTYPE_IDX=0 ;;
  c_fos_positive_patches) SUBTYPE_IDX=1 ;;
  cell_nucleus_patches) SUBTYPE_IDX=2 ;;
  vessels_patches) SUBTYPE_IDX=3 ;;
  *) echo "Unknown subtype: $SUBTYPE"; exit 1 ;;
esac
SEED=$(( 1000 + SUBTYPE_IDX*100 + FOLD_ID )) # vary seed by fold for more randomness

# sanity check
[[ -f "$FOLDS_JSON" ]] || { echo "Folds json file $FOLDS_JSON does not exist!"; exit 1; }
mkdir -p "$CKPT_DIR" "$PRED_ROOT"
echo "[INFO] Subtype: $SUBTYPE, Fold ID: $FOLD_ID, Seed: $SEED"

# run finetuning and inference
srun --unbuffered python -u /home/ads4015/ssl_project/src/finetune_and_inference_split.py \
  --root "$ROOT" \
  --subtypes "$SUBTYPE" \
  --ckpt_dir "$CKPT_DIR" \
  --val_percent 0.2 \
  --seed "$SEED" \
  --batch_size 4 \
  --feature_size 24 \
  --max_epochs 1000 \
  --freeze_encoder_epochs 5 \
  --encoder_lr_mult 0.05 \
  --loss_name dicece \
  --wandb_project selma3d_finetune \
  --num_workers 4 \
  --channel_substr ALL \
  --preds_root "$PRED_ROOT" \
  --preds_subtype "$PRETTY_SUBTYPE" \
  --folds_json "$FOLDS_JSON" \
  --fold_id "$FOLD_ID"


# indicate ending
echo "Ending finetune and inference."








