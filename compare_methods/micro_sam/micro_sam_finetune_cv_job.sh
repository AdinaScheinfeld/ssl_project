#!/bin/bash
#SBATCH --job-name=microsam_cv
#SBATCH --partition=sablab-gpu
#SBATCH --account=sablab
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=72:00:00
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/compare_methods/micro_sam/finetuned_cross_val_l/logs/microsam_cv_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/compare_methods/micro_sam/finetuned_cross_val_l/logs/microsam_cv_%A_%a.err
#SBATCH --array=0-227

# /home/ads4015/ssl_project/compare_methods/micro_sam/micro_sam_finetune_cv_job.sh

echo "SLURM ARRAY TASK ID: $SLURM_ARRAY_TASK_ID"

# ------------------------------------------------------------
# 1. Initialize micromamba
# ------------------------------------------------------------
eval "$(/home/ads4015/bin/micromamba shell hook -s bash)"
micromamba activate micro-sam-gpu

echo "Activated environment:"
echo "  $(which python)"
python --version

# ------------------------------------------------------------
# 2. Load the JSON entry for this array index
# ------------------------------------------------------------
INDEX_FILE="/midtier/paetzollab/scratch/ads4015/compare_methods/micro_sam/cv_index.json"

if [ ! -f "$INDEX_FILE" ]; then
    echo "ERROR: Missing index file: $INDEX_FILE"
    exit 1
fi

TASK_JSON=$(cat "$INDEX_FILE" | jq ".[$SLURM_ARRAY_TASK_ID]")

CLASS_NAME=$(echo "$TASK_JSON" | jq -r ".class_name")
POOL_SIZE=$(echo "$TASK_JSON" | jq -r ".pool_size")
FOLD_INDEX=$(echo "$TASK_JSON" | jq -r ".fold_index")

echo "Loaded parameters:"
echo "  CLASS_NAME = $CLASS_NAME"
echo "  POOL_SIZE  = $POOL_SIZE"
echo "  FOLD_INDEX = $FOLD_INDEX"

# ------------------------------------------------------------
# 3. Run finetuning cross-val for this split/fold
# ------------------------------------------------------------
python /home/ads4015/ssl_project/compare_methods/micro_sam/micro_sam_finetune_cv.py \
    --class-name "$CLASS_NAME" \
    --pool-size "$POOL_SIZE" \
    --fold-index "$FOLD_INDEX" \
    --data-root "/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches" \
    --output-root "/midtier/paetzollab/scratch/ads4015/compare_methods/micro_sam/finetuned_cross_val_l" \
    --epochs 500 \
    --early-stopping 50

echo "Task finished: class=$CLASS_NAME pool_size=$POOL_SIZE fold=$FOLD_INDEX"


