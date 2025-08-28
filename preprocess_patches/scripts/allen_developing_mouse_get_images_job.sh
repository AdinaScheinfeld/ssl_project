#!/bin/bash
#SBATCH --job-name=allen_lsfm_dl
#SBATCH --output=logs/allen_lsfm_dl_%A_%a.out
#SBATCH --error=logs/allen_lsfm_dl_%A_%a.err
#SBATCH --time=12:00:00
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --array=1-31    # adjust this to the number of E folders

set -euo pipefail

DEST="/ministorage/adina/allen_developing_mouse"
BASE_URL="https://download.brainimagelibrary.org/0d/89/0d89ff2f52ee3323"

cd "$DEST"

# Generate the URL list only once
if [ ! -f lsfm_links.txt ]; then
  curl -k -s "$BASE_URL/" \
    | grep -oE 'E[^/]+/' \
    | sort -u \
    | sed "s#^#$BASE_URL/#;s#\$#/LSFM/#" > lsfm_links.txt
fi

# Get the URL for this array index
URL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" lsfm_links.txt)

# ---- Print at the very beginning ----
echo "[INFO] SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
echo "[INFO] This task will download: $URL"

# ---- Run download ----
wget -r -np -c --no-check-certificate \
  --reject "index.html*" \
  --reject-regex '.*/(Background|Background2|Max)(/|$)' \
  -nH --cut-dirs=3 \
  "$URL"

  echo "[INFO] Download complete for SLURM_ARRAY_TASK_ID=${SLURM_ARRAY_TASK_ID}"
