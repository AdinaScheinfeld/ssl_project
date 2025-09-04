#!/usr/bin/env bash
#SBATCH --job-name=allen_human2_patches
#SBATCH --partition=minilab-cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=200G
#SBATCH --output=logs/allen_human2_%A_%a.out
#SBATCH --error=logs/allen_human2_%A_%a.err
#SBATCH --array=1-110

# allen_human2_get_images_job.sh - extract 1 patch per OME-TIFF using tif_volume_get_patches.py, one file per array task

set -euo pipefail

# indicate starting
echo "[INFO] Starting task: JobID=${SLURM_JOB_ID:-NA}, ArrayIndex=${SLURM_ARRAY_TASK_ID:-NA}"

# env
module load anaconda3/2022.10-34zllqw
source activate monai-env1

PYTHON=python
PY_SCRIPT="/home/ads4015/ssl_project/preprocess_patches/src/tif_volume_get_patches.py"

# inputs
SUBJECT_DIRS=(
  "/ministorage/adina/allen_human2/sub-138"
  "/ministorage/adina/allen_human2/sub-145"
  "/ministorage/adina/allen_human2/sub-146"
)

# output root; results will go under $OUT_ROOT/<subject-name>/
OUT_ROOT="/midtier/paetzollab/scratch/ads4015/all_allen_human2_patches"
mkdir -p "$OUT_ROOT"

# Patch extraction params
PATCH_SIZE=96
NUM_PATCHES=1
MIN_FG=0.05
SEED=100
STRIDE=""    # e.g., STRIDE=64 to force overlap; empty = default (no overlap)
CHANNEL=""   # set to 0/1/... if volumes are 4D (Z,Y,X,C); empty if 3D

# build a global, deterministic list of all .ome.tif files
LIST_FILE="$(mktemp "/tmp/allen_human2_ome_list_${SLURM_JOB_ID:-$$}_XXXX.txt")"
# cleanup temp file on exit
cleanup() { rm -f "$LIST_FILE"; }
trap cleanup EXIT

# collect files from each subject dir (top-level only). Remove -maxdepth to recurse.
{
  find "${SUBJECT_DIRS[0]}" -maxdepth 1 -type f \( -name "*.ome.tif" -o -name "*.tif" -o -name "*.tiff" -o -name "*.nii" -o -name "*.nii.gz" \) -print0
  find "${SUBJECT_DIRS[1]}" -maxdepth 1 -type f \( -name "*.ome.tif" -o -name "*.tif" -o -name "*.tiff" -o -name "*.nii" -o -name "*.nii.gz" \) -print0
  find "${SUBJECT_DIRS[2]}" -maxdepth 1 -type f \( -name "*.ome.tif" -o -name "*.tif" -o -name "*.tiff" -o -name "*.nii" -o -name "*.nii.gz" \) -print0
} | LC_ALL=C sort -z | tr '\0' '\n' > "$LIST_FILE"

NUM_FILES=$(wc -l < "$LIST_FILE" | tr -d '[:space:]')
TASK_ID=${SLURM_ARRAY_TASK_ID:-1}

echo "[INFO] Total files found: ${NUM_FILES}. This task index: ${TASK_ID}."

# if the array index exceeds the number of files, exit gracefully (useful when --array is a safe upper bound).
if (( TASK_ID < 1 || TASK_ID > NUM_FILES )); then
  echo "[INFO] No file mapped to this array index (${TASK_ID}). Exiting."
  exit 0
fi

# select the Nth file (1-based)
INPUT_TIF=$(sed -n "${TASK_ID}p" "$LIST_FILE")
if [[ -z "${INPUT_TIF}" ]]; then
  echo "[ERROR] Failed to resolve input file for index ${TASK_ID}."
  exit 1
fi

echo "[INFO] Processing file: ${INPUT_TIF}"

# derive subject name from the file path (assumes parent folder is sub-###)
SUBJECT_NAME="$(basename "$(dirname "$INPUT_TIF")")"
OUT_DIR="$OUT_ROOT/$SUBJECT_NAME"
mkdir -p "$OUT_DIR"

# build args for the Python script
ARGS=( "--input_tif" "$INPUT_TIF"
       "--output_dir" "$OUT_DIR"
       "--patch_size" "$PATCH_SIZE"
       "--num_patches" "$NUM_PATCHES"
       "--min_fg" "$MIN_FG"
       "--seed" "$SEED" )

if [[ -n "$STRIDE" ]]; then
  ARGS+=( "--stride" "$STRIDE" )
fi

if [[ -n "$CHANNEL" ]]; then
  ARGS+=( "--channel" "$CHANNEL" )
fi

# run the script with args
set -x
"$PYTHON" "$PY_SCRIPT" "${ARGS[@]}"
set +x


# indicate completion
echo "[INFO] Task complete. Output dir: $OUT_DIR"







