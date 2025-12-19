#!/bin/bash
#SBATCH --job-name=pca_lr_all
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_pca_lr/logs/pca_lr_all_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_pca_lr/logs/pca_lr_all_%A_%a.err
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-62 # adjust based on total number of folds across all JSONs


# /home/ads4015/ssl_project/baselines/classification/pca_lr_baseline_submit_all_arrays.sh


# Usage:
#   sbatch pca_lr_baseline_submit_all_arrays.sh /path/to/json_dir

set -euo pipefail

JSON_DIR="${1:?JSON directory not provided}"

# Load the list of all JSONs
shopt -s nullglob
JSON_FILES=( "$JSON_DIR"/cls_folds_tr*_test*_rep*.json )
shopt -u nullglob

if (( ${#JSON_FILES[@]} == 0 )); then
  echo "[ERROR] No JSON files found in ${JSON_DIR}"
  exit 1
fi

# Build a global mapping of array_index -> (JSON_PATH, FOLD_ID)
# and save into a temporary file for lookup
MAP_FILE="/tmp/pca_lr_map_${SLURM_JOB_ID}.txt"

if [[ ! -f "$MAP_FILE" ]]; then
  echo "[INFO] Building fold mapping file: $MAP_FILE"

  idx=0
  for JSON in "${JSON_FILES[@]}"; do
    NF=$(python - "$JSON" << 'PY'
import json, sys
d=json.load(open(sys.argv[1],'r'))
print(len(d["folds"]))
PY
    )

    for ((F=0; F< NF; F++)); do
      echo "$idx|$JSON|$F" >> "$MAP_FILE"
      idx=$((idx+1))
    done
  done

  echo "[INFO] Total folds = $idx"
fi

# Lookup entry for this SLURM array index
LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$MAP_FILE")

if [[ -z "$LINE" ]]; then
  echo "[ERROR] No mapping found for array index ${SLURM_ARRAY_TASK_ID}"
  exit 1
fi

IFS='|' read -r GLOBAL_ID JSON_PATH FOLD_ID <<< "$LINE"

echo "[INFO] Running GLOBAL_ID=$GLOBAL_ID JSON=$JSON_PATH FOLD_ID=$FOLD_ID"

module load anaconda3/2022.10-34zllqw
source activate monai-env1

python /home/ads4015/ssl_project/baselines/classification/pca_lr_baseline.py \
    --fold_json "$JSON_PATH" \
    --fold_id "$FOLD_ID" \
    --output_root "/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_pca_lr" \
    --pca_components 50 \
    --max_iter 500 \
    --seed 100

echo "[INFO] Done GLOBAL_ID=$GLOBAL_ID"












