#!/bin/bash
# submit_all_cls_folds.sh
#
# Submit a classification cross-val job array for every cls_folds_*.json
# in a given directory. Each JSON gets its own sbatch array with the
# correct 0..(num_folds-1) range.
#
# Usage:
#   ./submit_all_cls_folds.sh [FOLDS_DIR] [INIT_MODE]
#
# Examples:
#   ./submit_all_cls_folds.sh \
#       /midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_expert_sweep_31 \
#       pretrained
#
#   ./submit_all_cls_folds.sh   # uses defaults below

set -euo pipefail

# --- Config defaults ---

FOLDS_DIR="${1:-/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_expert_sweep_31}"
INIT_MODE="${2:-pretrained}"  # "pretrained" or "random"

# pattern for your folds JSONs; adjust if needed
FOLDS_PATTERN="cls_folds_tr*_test2_rep3.json"

# path to your existing array launcher script
ARRAY_SCRIPT="/home/ads4015/ssl_project/scripts/finetune_eval_classification_cross_val_job_array2.sh"

echo "[INFO] Submitting classification CV jobs from directory: $FOLDS_DIR"
echo "[INFO] Using INIT_MODE=$INIT_MODE"
echo "[INFO] Using array script: $ARRAY_SCRIPT"
echo

# --- Loop over all matching JSONs ---

shopt -s nullglob
JSON_FILES=( "$FOLDS_DIR"/$FOLDS_PATTERN )
shopt -u nullglob

if (( ${#JSON_FILES[@]} == 0 )); then
  echo "[WARN] No JSON files matching pattern '$FOLDS_PATTERN' found in $FOLDS_DIR"
  exit 0
fi

for JSON in "${JSON_FILES[@]}"; do
  echo "------------------------------------------------------------"
  echo "[INFO] Processing folds JSON: $JSON"

  # determine number of folds in this JSON
  NUM_FOLDS=$(python - "$JSON" <<'PY'
import json, sys
path = sys.argv[1]
with open(path, "r") as f:
    d = json.load(f)
folds = d.get("folds", [])
print(len(folds))
PY
)

  if [[ -z "$NUM_FOLDS" || "$NUM_FOLDS" == "0" ]]; then
    echo "[WARN] $JSON has 0 folds (folds list empty) – skipping."
    continue
  fi

  echo "[INFO] $JSON contains $NUM_FOLDS folds."
  echo "[INFO] Submitting sbatch array: 0-$((NUM_FOLDS-1))"

  # submit job array for this JSON, overriding the #SBATCH --array in the script
  sbatch --array=0-$((NUM_FOLDS-1)) "$ARRAY_SCRIPT" "$JSON" "$INIT_MODE"
done

echo
echo "[INFO] Done submitting all classification cross-val arrays."





