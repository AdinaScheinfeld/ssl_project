#!/bin/bash
#SBATCH --job-name=microsam_seg
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/compare_methods/micro_sam/logs/microsam_seg_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/compare_methods/micro_sam/logs/microsam_seg_%A_%a.err
#SBATCH --partition=sablab-gpu          # adjust if needed
#SBATCH --account=sablab                 # adjust if needed
#SBATCH --gres=gpu:1                     # or comment out for CPU-only
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
#SBATCH --array=0-0                      # will be overridden at submission

# /home/ads4015/ssl_project/compare_methods/micro_sam/micro_sam_batch_array_job.sh

# --- Paths ---
IN_ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
OUT_ROOT="/midtier/paetzollab/scratch/ads4015/compare_methods/micro_sam"
FILE_LIST="/midtier/paetzollab/scratch/ads4015/compare_methods/micro_sam/microsam_input_list.txt"
SCRIPT="/home/ads4015/ssl_project/compare_methods/micro_sam/micro_sam_batch.py"

# --- Micromamba setup ---
export MAMBA_ROOT_PREFIX="$HOME/micromamba"
eval "$($HOME/bin/micromamba shell hook -s bash)"
micromamba activate micro-sam-gpu

echo "Host: $(hostname)"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"

python "$SCRIPT" \
  --file_list "$FILE_LIST" \
  --output_root "$OUT_ROOT" \
  --model_type "vit_b_lm" \
  --device "auto" \
  --per_task 1





