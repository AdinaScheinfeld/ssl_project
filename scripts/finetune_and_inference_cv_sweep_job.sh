#!/bin/bash
#SBATCH --job-name=finetune_infer_cv_sweep
#SBATCH --output=logs/finetune_infer_cv_sweep_%j.out
#SBATCH --error=logs/finetune_infer_cv_sweep_%j.err
#SBATCH --partition=minilab-gpu
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:h100:1
#SBATCH --mem=128G
#SBATCH --time=48:00:00


# finetune_and_inference_cv_sweep_job.sh - Script to finetune a pretrained model and perform inference on a dataset split into training and evaluation sets.

# indicate starting
echo "Starting finetune and inference..."


# set up environment
set -euo pipefail
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# set variables
ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
OUTDIR="/ministorage/adina/selma_segmentation_preds/cv_folds"
JOB="/home/ads4015/ssl_project/scripts/finetune_and_inference_cv_job.sh"
REPEATS=10
SEED=100
CHANNELS="ALL"

# sweep counts per subtype
declare -A COUNTS
COUNTS[amyloid_plaque_patches]="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19"
COUNTS[c_fos_positive_patches]="0 1 2 3 4"
COUNTS[cell_nucleus_patches]="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25"
COUNTS[vessels_patches]="0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20"

# create output directory
mkdir -p "$OUTDIR"

# loop over subtypes and do sweeps
for SUBTYPE in "${!COUNTS[@]}"; do
    for K in ${COUNTS[$SUBTYPE]}; do
        FJSON="${OUTDIR}/${SUBTYPE}_folds_tr${K}_rep${REPEATS}.json" # folds json file

        # build cv folds
        python /home/ads4015/ssl_project/src/get_selma_cross_val_folds.py \
          --root "$ROOT" \
          --subtypes "$SUBTYPE" \
          --channel_substr "$CHANNELS" \
          --train_limit "$K" \
          --repeats "$REPEATS" \
          --seed "$SEED" \
          --output_json "$FJSON" || true

        # if json exists, launch jobs for each fold
        if [[ -f "$FJSON" ]]; then
            for ((FID=0; FID<REPEATS; FID++)); do
                sbatch --job-name "${SUBTYPE}_tr${K}_f${FID}" "$JOB" "$SUBTYPE" "$FID" "$FJSON"
            done
        else
            echo "[WARN] Folds JSON file $FJSON not found, skipping subtype $SUBTYPE with train count $K"
        fi
    done
done


# indicate ending
echo "Ending finetune and inference."




















