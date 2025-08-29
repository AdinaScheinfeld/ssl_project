#!/usr/bin/env bash
#SBATCH --job-name=dl_sub146_arr
#SBATCH --partition=minilab-cpu
#SBATCH --time=05:00:00
#SBATCH --cpus-per-task=2
#SBATCH --output=logs/dl_sub146_%A_%a.out
#SBATCH --error=logs/dl_sub146_%A_%a.err
#SBATCH --array=1-40


# indicate starting
echo "Beginning download for task ${SLURM_ARRAY_TASK_ID}..."


set -euo pipefail

# config

# # 138
# URL_FILE="/ministorage/adina/allen_human2/sub-138/urls_sub138.txt" # file with URLs
# TARGET_DIR="/ministorage/adina/allen_human2/sub-138" # output dir


# 146
URL_FILE="/ministorage/adina/allen_human2/sub-146/urls_sub146.txt" # file with URLs
TARGET_DIR="/ministorage/adina/allen_human2/sub-146" # output dir

# select url for this task
URL=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$URL_FILE")

# change to correct directory
cd "$TARGET_DIR"

# -L : follow redirects
# -J : use filename from server
# -O : write output to file (with -J = Content-Disposition name)
# --retry 6 : retry up to 6 times
# --retry-all-errors : retry on all transient errors (including 403, 500)
# --connect-timeout 30 : give up if can't connect in 30s
# --max-time 0 : no overall timeout
curl -L -J -O \
  --fail \
  --retry 6 \
  --retry-all-errors \
  --connect-timeout 30 \
  --max-time 0 \
  "$URL"


# indicate completion
echo "[$(date)] (${HOSTNAME}) Task ${SLURM_ARRAY_TASK_ID}: download complete."
