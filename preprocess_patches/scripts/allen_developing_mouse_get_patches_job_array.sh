#!/usr/bin/env bash
#SBATCH --job-name=dev_mouse
#SBATCH --partition=minilab-cpu
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --output=logs/dev_mouse_%A_%a.out
#SBATCH --error=logs/dev_mouse_%A_%a.err
#SBATCH --array=1-42


# indicate starting
echo "Starting developing mouse brain patch extraction"

set -euo pipefail

# config
LIST_FILE="/ministorage/adina/allen_developing_mouse/selected_images.txt"
OUT_DIR="/midtier/paetzollab/scratch/ads4015/all_allen_developing_mouse_patches"
PY_SCRIPT="/home/ads4015/ssl_project/preprocess_patches/src/tif_stack_get_patches.py"
PATCH_SIZE=${PATCH_SIZE:-96}
MIN_FG=${MIN_FG:-0.05}
PATTERN=${PATTERN:-"Z*_ch*.tif"}

# select folder for this array index (1-based)
INPUT_DIR=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST_FILE")
if [[ -z "${INPUT_DIR:-}" ]]; then
  echo "[ERROR] No line ${SLURM_ARRAY_TASK_ID} in ${LIST_FILE}" >&2
  exit 1
fi

# matching for category-based patch counts (case-insensitive)
shopt -s nocasematch
NUM_PATCHES=1
if [[ "$INPUT_DIR" =~ (artery|syto16|pericyte) ]]; then
  NUM_PATCHES=2
elif [[ "$INPUT_DIR" =~ (microtubule|neurofilament) ]]; then
  NUM_PATCHES=10
fi
shopt -u nocasematch

# create output directory
mkdir -p logs "${OUT_DIR}"

# log
echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID}"
echo "  INPUT_DIR     : ${INPUT_DIR}"
echo "  OUTPUT_DIR    : ${OUT_DIR}"
echo "  NUM_PATCHES   : ${NUM_PATCHES}  (rules: Artery/Syto16/Pericyte=2; Microtubule/Neurofilament=10; else=1)"
echo "  PATCH_SIZE    : ${PATCH_SIZE}"
echo "  MIN_FG        : ${MIN_FG}"
echo "  PATTERN       : ${PATTERN}"
echo "  PY_SCRIPT     : ${PY_SCRIPT}"


# load modules
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# use SLURM_ARRAY_TASK_ID to perturb the deterministic per-folder seed
python "${PY_SCRIPT}" \
  --input_dir "${INPUT_DIR}" \
  --output_dir "${OUT_DIR}" \
  --patch_size "${PATCH_SIZE}" \
  --num_patches "${NUM_PATCHES}" \
  --min_fg "${MIN_FG}" \
  --pattern "${PATTERN}" \
  --seed "${SLURM_ARRAY_TASK_ID}"


# indicate completion
echo "[$(date)] Task ${SLURM_ARRAY_TASK_ID} done."











