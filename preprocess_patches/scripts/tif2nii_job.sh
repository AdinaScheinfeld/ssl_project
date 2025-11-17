#!/bin/bash
#SBATCH --job-name=tif2nii
#SBATCH --output=logs/tif2nii_%j.out
#SBATCH --error=logs/tif2nii_%j.err
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

# tif_to_nifti_job.sh - SLURM job to convert specific TIFF volumes to NIfTI

set -euo pipefail

# indicate starting
echo "[INFO] Starting tif2nii job on $(hostname)"
date

# load env
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# Output directory
OUT_DIR="/midtier/paetzollab/scratch/ads4015/cellseg3d_data/data_nifti"

# Path to the Python script (assumed to be in the same dir you submit from)
SCRIPT_PATH="/home/ads4015/ssl_project/preprocess_patches/src/tif2nii.py"

# TIFF files to convert
python "$SCRIPT_PATH" \
    --output_dir "$OUT_DIR" \
    --input_tifs \
    /midtier/paetzollab/scratch/ads4015/cellseg3d_data/data/c1image.tif \
    /midtier/paetzollab/scratch/ads4015/cellseg3d_data/data/c2image.tif \
    /midtier/paetzollab/scratch/ads4015/cellseg3d_data/data/c3image.tif \
    /midtier/paetzollab/scratch/ads4015/cellseg3d_data/data/c4image.tif \
    /midtier/paetzollab/scratch/ads4015/cellseg3d_data/data/c5image.tif \
    /midtier/paetzollab/scratch/ads4015/cellseg3d_data/data/visual.tif

echo "[INFO] Finished tif2nii job"
date
