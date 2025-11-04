#!/usr/bin/env bash
#SBATCH --job-name=launch_cls_sweep
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:30:00
#SBATCH --output=logs/launch_cls_sweep_%j.out
#SBATCH --error=logs/launch_cls_sweep_%j.err

# submit_classification_sweep_jobs.sh - Launch a sweep of classification cross-validation finetune/eval jobs for various training set sizes.

# indicate starting 
echo "[INFO] Starting submission of classification sweep jobs on $(date)..."

set -euo pipefail

# config
FOLDS_DIR="/ministorage/adina/classification_eval/folds3_test2"
REPEATS=3 # must equal number of repeat used when creating the JSONs
TRAIN_SIZES=(2 3 4 5 6 8 9 10 11 12 13 14 15 16 17 18 19 20)
ARRAY_RANGE="0-$((REPEATS-1))" # indices for the array jobs
CONCURRENCY=6 # limit concurrent tasks per array

ARRAY_SCRIPT="/home/ads4015/ssl_project/scripts/finetune_eval_classification_cross_val_job_array.sh"

# sanity check 
if [[ ! -d "$FOLDS_DIR" ]]; then 
    echo "[ERR] Folds dir not found: $FOLDS_DIR" >&2 
    exit 2 
fi

# submit pretrained + random for each train size
for TR in "${TRAIN_SIZES[@]}"; do 
    J="${FOLDS_DIR}/cls_folds_tr${TR}_test2_rep${REPEATS}.json" 
    if [[ ! -f "$J" ]]; then 
        echo "[WARN] Missing folds JSON for TR=$TR: $J (skipping)" 
        continue 
    fi

    # pretrained
    sbatch --array=${ARRAY_RANGE}%${CONCURRENCY} \
        --job-name="cls_pre_tr${TR}" \
        "$ARRAY_SCRIPT" "$J" pretrained

    # random init
    sbatch --array=${ARRAY_RANGE}%${CONCURRENCY} \
        --job-name="cls_rand_tr${TR}" \
        "$ARRAY_SCRIPT" "$J" random
done

# print summary
echo "[INFO] Submitted sweep for train sizes: ${TRAIN_SIZES[*]} (repeats=$REPEATS)"



# indicate done
echo "[INFO] Finished submission of classification sweep jobs on $(date)."

