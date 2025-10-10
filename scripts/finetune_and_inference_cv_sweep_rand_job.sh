#!/bin/bash
#SBATCH --job-name=finetune_infer_cv_sweep_rand
#SBATCH --output=logs/finetune_infer_cv_sweep_rand_%j.out
#SBATCH --error=logs/finetune_infer_cv_sweep_rand_%j.err
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00


# finetune_and_inference_cv_sweep_rand_job.sh - Script to finetune a pretrained model and perform inference on a dataset split into training and evaluation sets.

# indicate starting
echo "Starting finetune and inference..."


# set up environment
set -euo pipefail
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# set variables
ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
OUTDIR="/ministorage/adina/selma_segmentation_preds_rand/cv_folds"
JOB="/home/ads4015/ssl_project/scripts/finetune_and_inference_cv_rand_job.sh"
NUM_FOLDS=5
SEED=100
CHANNELS="ALL"

# create list of train limits per subtype
declare -A TRAIN_LIMITS
TRAIN_LIMITS[amyloid_plaque_patches]="19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1"
TRAIN_LIMITS[c_fos_positive_patches]="4 3 2 1"
TRAIN_LIMITS[cell_nucleus_patches]="25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1"
TRAIN_LIMITS[vessels_patches]="20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1"

# create output directory
mkdir -p "$OUTDIR"

# create list of subtypes
for SUBTYPE in amyloid_plaque_patches c_fos_positive_patches cell_nucleus_patches vessels_patches; do
    for K in ${TRAIN_LIMITS[$SUBTYPE]}; do
        FJSON="${OUTDIR}/${SUBTYPE}_tr${K}_cv${NUM_FOLDS}.json"

        # build cv folds
        echo "[INFO] Generating CV folds for subtype ${SUBTYPE}, train limit=${K} - FJSON: ${FJSON}"
        python /home/ads4015/ssl_project/src/get_selma_cross_val_folds.py \
            --root "$ROOT" \
            --subtypes "$SUBTYPE" \
            --channel_substr "$CHANNELS" \
            --train_limit "$K" \
            --repeats "$NUM_FOLDS" \
            --seed "$SEED" \
            --output_json "$FJSON"

        # if json exists, launch jobs for each fold
        if [[ -f "$FJSON" ]]; then
            for ((FID=0; FID<NUM_FOLDS; FID++)); do
                sbatch --job-name "${SUBTYPE}_cv${NUM_FOLDS}_f${FID}" "$JOB" "$SUBTYPE" "$FID" "$FJSON"
            done
        else
            echo "[WARN] Folds JSON file $FJSON not found, skipping subtype $SUBTYPE"
        fi
    done
done


# indicate ending
echo "Ending finetune and inference."




















