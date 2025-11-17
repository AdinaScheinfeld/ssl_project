#!/usr/bin/env bash
#SBATCH --job-name=mesospim_patches
#SBATCH --partition=minilab-cpu
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --output=logs/mesospim_patches_%A_%a.out
#SBATCH --error=logs/mesospim_patches_%A_%a.err
#SBATCH --array=0-2

set -euo pipefail

# indicate starting
echo "[INFO] Starting Mesospim patch extraction job: JobID=${SLURM_JOB_ID:-NA}, ArrayIndex=${SLURM_ARRAY_TASK_ID:-NA}"

# load env
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# python script and args
PYTHON=python
PY_SCRIPT="/home/ads4015/ssl_project/preprocess_patches/src/tif_volume_get_patches.py"

# list of Mesospim TIFFs (indexed by SLURM_ARRAY_TASK_ID)
FILES=(
  "/midtier/paetzollab/scratch/ads4015/mesospim_raw/ExpA_VIP_ASLM_off.tif"
  "/midtier/paetzollab/scratch/ads4015/mesospim_raw/ExpA_VIP_ASLM_on.tif"
  "/midtier/paetzollab/scratch/ads4015/mesospim_raw/ExpC_TPH2_whole_brain.tif"
)

# which file to process
IDX=${SLURM_ARRAY_TASK_ID:-0}
INPUT_TIF="${FILES[$IDX]}"

echo "[INFO] Processing file: ${INPUT_TIF}"

# output dir
OUT_DIR="/midtier/paetzollab/scratch/ads4015/all_mesospim_patches"
mkdir -p "$OUT_DIR"

# patch extraction params
PATCH_SIZE=96 # patch size
NUM_PATCHES=20 # patches per file
MIN_FG=0.05 # at least 5% foreground
SEED=100

# build args array
ARGS=(
  "--input_tif" "$INPUT_TIF"
  "--output_dir" "$OUT_DIR"
  "--patch_size" "$PATCH_SIZE"
  "--num_patches" "$NUM_PATCHES"
  "--min_fg" "$MIN_FG"
  "--seed" "$SEED"
)

# run patch extraction
echo "[INFO] Running: $PYTHON $PY_SCRIPT ${ARGS[*]}"
set -x
"$PYTHON" "$PY_SCRIPT" "${ARGS[@]}"
set +x

# indicate done
echo "[INFO] Mesospim patch extraction complete for file: ${INPUT_TIF}"
echo "[INFO] Output dir: $OUT_DIR"






