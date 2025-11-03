#!/bin/bash
#SBATCH --job-name=mk_cls_folds
#SBATCH --output=logs/mk_cls_folds_%j.out
#SBATCH --error=logs/mk_cls_folds_%j.err
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00

# get_classification_cross_val_folds_job.sh - Create cross-validation folds for classification task.


# indicate starting
echo "[INFO] Starting classification cross-validation folds creation job on $(date)..."

# activate conda env
set -euo pipefail
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# paths
ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches" # root data dir
OUT_JSON="/ministorage/adina/classification_eval/selma_classification_folds.json"  # <- where to write folds
mkdir -p "$(dirname "$OUT_JSON")"

# Tuning knobs for folds creation
REPEATS=5 # number of folds (i.e., how many train/eval splits)
TEST_FRAC=0.2 # percent of data held out per fold (distributed per class)
LOCK_TEST=false # if true, same eval set for every repeat
CHANNEL_SUBSTR=ALL # filter e.g. "ch0,ch1" or use ALL for no filter
SEED=100

# build a single global, stratified split
python /home/ads4015/ssl_project/src/get_classification_cross_val_folds.py \
  --root_dir "$ROOT" \
  --repeats "$REPEATS" \
  --test_frac "$TEST_FRAC" \
  $( $LOCK_TEST && echo "--lock_test" || true ) \
  --channel_substr "$CHANNEL_SUBSTR" \
  --seed "$SEED" \
  --output_json "$OUT_JSON"


# indicate done
echo "[INFO] Wrote folds to: $OUT_JSON"




