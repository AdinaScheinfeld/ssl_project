#!/bin/bash
#SBATCH --job-name=make_blurred_patches
#SBATCH --output=logs/make_blurred_patches_%j.out
#SBATCH --error=logs/make_blurred_patches_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --partition=sablab-cpu
#SBATCH --account=sablab

set -euo pipefail

# indicate starting
echo "Starting blurred patch creation at: $(date)"


# load environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# set paths
INPUT_ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
OUTPUT_ROOT="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches_blurred"

# run script
python /home/ads4015/ssl_project/src/make_blurred_patches.py \
    --input_root "$INPUT_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --sigma 1.5 \
    --noise_std 1.5 \
    --overwrite

# indicate completion
echo "Completed blurred patch creation at: $(date)"















