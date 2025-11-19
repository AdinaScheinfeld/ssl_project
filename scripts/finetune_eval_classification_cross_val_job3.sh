#!/bin/bash
#SBATCH --job-name=classification_cross_val
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_random/logs/classification_cross_val_%j.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_random/logs/classification_cross_val_%j.err
#SBATCH --partition=sablab-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --account=sablab

# finetune_eval_classification_cross_val_job2.sh - Finetune and evaluate classification model on a single cross-validation fold.


# indicate starting
echo "[INFO] Starting classification cross-validation finetune/eval job on $(date)..."


# activate conda env
set -euo pipefail
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# args
FOLD_ID="${1:?Usage: $0 <fold_id> <fold_json> [pretrained|random] }" # which fold to run (0-indexed)
FOLD_JSON="${2:?need folds json path}" # path to folds json
INIT_MODE="${3:-pretrained}" # default: pretrained (use "random" for baseline)

# paths
ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches" # root data dir
CKPT_DIR="/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_random/checkpoints/finetune_cls" # checkpoint output dir
METRICS_ROOT="/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_random/cls_metrics" # metrics output dir

# additional roots for new classes
MESO_ROOT="/midtier/paetzollab/scratch/ads4015/all_mesospim_patches"
ALLEN_ROOT="/midtier/paetzollab/scratch/ads4015/all_allen_human_patches"

# extra class glob specifications (ClassName:glob_pattern)
EXTRA_CLASS_GLOBS=(

  # mesospim classes
  "VIP_ASLM_off:${MESO_ROOT}/*VIP_ASLM_off*.nii*"
  "VIP_ASLM_on:${MESO_ROOT}/*VIP_ASLM_on*.nii*"
  "TPH2:${MESO_ROOT}/*TPH2*.nii*"

  # allen human patch classes
  "stain-CR:${ALLEN_ROOT}/*_cr_*ps96*.nii*"
  "stain-LEC:${ALLEN_ROOT}/*_lec_*ps96*.nii*"
  "stain-NN:${ALLEN_ROOT}/*_nn_*ps96*.nii*"
  "stain-NPY:${ALLEN_ROOT}/*_npy_*ps96*.nii*"
  "stain-YO:${ALLEN_ROOT}/*_yo_*ps96*.nii*"
)

# PRETRAINED="/ministorage/adina/pretrain_sweep_updated/checkpoints/kjvlrs45/all_datasets_clip_pretrained-updated-epochepoch=354-val-reportval_loss_report=0.0968-stepstep=20590.ckpt" # pretrained backbone checkpoint (image+clip)
PRETRAINED="" # pretrained backbone checkpoint (image only)
FEATURE_SIZE=36 # feature size must match backbone pretraining (use 24 for no-clip pretraining, use 36 for clip pretraining)

# create output dirs
mkdir -p "$CKPT_DIR" "$METRICS_ROOT"

# imbalance strategy (loss weighting); sampler optional
CLASS_WEIGHTING="inverse_freq" # none | inverse_freq | effective_num
BETA=0.9999 # for effective_num
USE_SAMPLER=false # set true to also use WeightedRandomSampler

# common args
COMMON_ARGS=(
  --root_dir "$ROOT"
  --ckpt_dir "$CKPT_DIR"
  --metrics_root "$METRICS_ROOT"
  --seed 100 --batch_size 8 --max_epochs 200 --num_workers 4
  --val_percent 0.2
  --channel_substr ALL
  --feature_size "$FEATURE_SIZE"
  --fold_json "$FOLD_JSON" --fold_id "$FOLD_ID"
  --wandb_project selma3d_classify
  --class_weighting "$CLASS_WEIGHTING" --beta "$BETA"
)

# add extra class glob args
for spec in "${EXTRA_CLASS_GLOBS[@]}"; do
  COMMON_ARGS+=(--extra_class_globs "$spec")
done

# # Optional sampler flag
# $USE_SAMPLER && COMMON_ARGS+=(--use_weighted_sampler)

# choose init path
if [[ "$INIT_MODE" == "pretrained" ]]; then
  COMMON_ARGS+=(--init_mode pretrained --pretrained_ckpt "$PRETRAINED" --freeze_encoder_epochs 5)
elif [[ "$INIT_MODE" == "random" ]]; then
  COMMON_ARGS+=(--init_mode random)
else
  echo "[ERR] INIT_MODE must be 'pretrained' or 'random'"; exit 2
fi

# run finetune + eval for this fold
srun python /home/ads4015/ssl_project/src/finetune_and_eval_classification_split.py "${COMMON_ARGS[@]}"


# indicate done
echo "[INFO] Finished classification cross-validation finetune/eval job on $(date)."




