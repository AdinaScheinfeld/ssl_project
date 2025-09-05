#!/usr/bin/env bash
#SBATCH --job-name=allen_conn_proj_patches
#SBATCH --partition=minilab-cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --array=0-5
#SBATCH --output=logs/allen_conn_proj_%A_%a.out
#SBATCH --error=logs/allen_conn_proj_%A_%a.err


# indicate starting
echo "Beginning Allen Connection Projection Patch Extraction"


set -euo pipefail

# config
PY_SCRIPT="/home/ads4015/ssl_project/preprocess_patches/src/tif_stack_get_patches_multichannel.py"
OUT_DIR="/midtier/paetzollab/scratch/ads4015/all_allen_connection_projection_patches"
PATCH_SIZE=96
NUM_PATCHES=10
MIN_FG=0.05
PATTERN="*.tif"
SORT_REGEX="(\d+)(?=\.tif$)"
CHANNELS=(0 1 2)

# input dirs
DIRS=(
  "/midtier/paetzollab/scratch/ads4015/allen_connection_projection/20211001_09_50_54_SW210318_07_LS_4X_2000z"
  "/midtier/paetzollab/scratch/ads4015/allen_connection_projection/20211020_11_21_05_SM210705_02_4x_2000z"
)

NUM_DIRS=${#DIRS[@]}
NUM_CHANS=${#CHANNELS[@]}

dir_idx=$(( (SLURM_ARRAY_TASK_ID) / NUM_CHANS ))
chan_idx=$(( (SLURM_ARRAY_TASK_ID) % NUM_CHANS ))

INPUT_DIR="${DIRS[$dir_idx]}"
C="${CHANNELS[$chan_idx]}"

# output dir
RUN_OUT_DIR="${OUT_DIR}/$(basename "${INPUT_DIR}")/c${C}"
mkdir -p "${RUN_OUT_DIR}"

# log
echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}"
echo "  INPUT_DIR   : ${INPUT_DIR}"
echo "  OUTPUT_DIR  : ${RUN_OUT_DIR}"
echo "  PATCH_SIZE  : ${PATCH_SIZE}"
echo "  NUM_PATCHES : ${NUM_PATCHES}"
echo "  MIN_FG      : ${MIN_FG}"
echo "  PATTERN     : ${PATTERN}"
echo "  SORT_REGEX  : ${SORT_REGEX}"
echo "  PY_SCRIPT   : ${PY_SCRIPT}"
echo "  CHANNEL     : ${C}"

# env
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# run script
python "${PY_SCRIPT}" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${RUN_OUT_DIR}" \
  --patch_size "${PATCH_SIZE}" \
  --num_patches "${NUM_PATCHES}" \
  --min_fg "${MIN_FG}" \
  --pattern "${PATTERN}" \
  --sort_regex "${SORT_REGEX}" \
  --seed "${SLURM_ARRAY_TASK_ID}" \
  --channels "${C}"


# indicate completion
echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} done."


