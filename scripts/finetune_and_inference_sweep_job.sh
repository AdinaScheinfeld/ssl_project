#!/bin/bash
#SBATCH --job-name=finetune_infer_sweep
#SBATCH --output=logs/finetune_infer_sweep_%j.out
#SBATCH --error=logs/finetune_infer_sweep_%j.err
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:l40:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00


# indicate starting
echo "Starting finetune and inference sweep..."


# set up environment
set -euo pipefail
module load anaconda3/2022.10-34zllqw
source activate monai-env1


# path to job script
JOB_SCRIPT="/home/ads4015/ssl_project/scripts/finetune_and_inference_split_job.sh"

# amyloid_plaque_patches
for k in 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0; do
  sbatch --job-name "amyloid_${k}" "$JOB_SCRIPT" "amyloid_plaque_patches" "$k"
done

# c_fos_positive_patches
for k in 4 3 2 1 0; do
  sbatch --job-name "cfos_${k}" "$JOB_SCRIPT" "c_fos_positive_patches" "$k"
done

# cell_nucleus_patches
for k in 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0; do
  sbatch --job-name "nucleus_${k}" "$JOB_SCRIPT" "cell_nucleus_patches" "$k"
done

# vessels_patches: 10,5,2,1,0
for k in 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 0; do
  sbatch --job-name "vessels_${k}" "$JOB_SCRIPT" "vessels_patches" "$k"
done


# indicate ending
echo "Ending finetune and inference sweep."










