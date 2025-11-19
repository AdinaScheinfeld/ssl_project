#!/bin/bash
#SBATCH --job-name=classification_cross_val_arr
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_random/logs/classification_cross_val_arr_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_random/logs/classification_cross_val_arr_%A_%a.err
#SBATCH --partition=sablab-gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-4 # adjust based on number of folds
#SBATCH --account=sablab

# finetune_eval_classification_cross_val_job_array3.sh - Finetune and evaluate classification model across multiple cross-validation folds using a SLURM job array.
# launch with: sbatch scripts/finetune_eval_classification_cross_val_job_array3.sh <path/to/json/with/folds> <[pretrained|random]>
# ex: sbatch scripts/finetune_eval_classification_cross_val_job_array3.sh /ministorage/adina/classification_eval/selma_classification_folds.json pretrained

# indicate starting
echo "[INFO] Starting classification cross-validation finetune/eval job array on $(date)"

# activate conda env
set -euo pipefail
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# args
FOLD_JSON="${1:?Usage: $0 <fold_json> [pretrained|random]}"
INIT_MODE="${2:-pretrained}"

# detect number of folds if not provided by array range
NUM_FOLDS=$(python - "$FOLD_JSON" <<'PY'
import json,sys
with open(sys.argv[1],'r') as f:
    d=json.load(f)
print(len(d.get('folds',[])))
PY
)

# ensure SLURM_ARRAY_TASK_ID is set
if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
  echo "[ERR] Run with sbatch --array=0-$((NUM_FOLDS-1))%<concurrency> $0 $FOLD_JSON $INIT_MODE"; exit 3
fi

# which fold to run
FOLD_ID="$SLURM_ARRAY_TASK_ID"

# delegate to the single-fold job logic
bash /home/ads4015/ssl_project/scripts/finetune_eval_classification_cross_val_job3.sh "$FOLD_ID" "$FOLD_JSON" "$INIT_MODE"


# indicate done
echo "[INFO] Finished classification cross-validation finetune/eval job array on $(date)."











