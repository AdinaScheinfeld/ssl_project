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
OUTDIR="/ministorage/adina/selma_classification_preds_expert_sweep_31"  # location to write folds
mkdir -p "$OUTDIR"

# Tuning knobs for folds creation
REPEATS=3 # number of folds (i.e., how many train/eval splits)
TEST_FRAC=0.2 # percent of data held out per fold (distributed per class)
LOCK_TEST=false # if true, same eval set for every repeat
CHANNEL_SUBSTR=ALL # filter e.g. "ch0,ch1" or use ALL for no filter
SEED=100
FIXED_TEST_PER_CLASS=2 # fixed number of test samples per class (overrides TEST_FRAC if set; caps at available)

# build a single global, stratified split
for TR in 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20; do
  python /home/ads4015/ssl_project/src/get_classification_cross_val_folds.py \
    --root_dir "$ROOT" \
    --repeats "$REPEATS" \
    --fixed_test_per_class "$FIXED_TEST_PER_CLASS" \
    --train_per_class "$TR" \
    --test_frac "$TEST_FRAC" \
    $( $LOCK_TEST && echo "--lock_test" || true ) \
    --channel_substr "$CHANNEL_SUBSTR" \
    --seed "$SEED" \
    --output_json "$OUTDIR/cls_folds_tr${TR}_test${FIXED_TEST_PER_CLASS}_rep${REPEATS}.json"
done


# indicate done
echo "[INFO] Wrote folds to: $OUTDIR"




