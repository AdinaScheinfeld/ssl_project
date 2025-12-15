#!/bin/bash

# /home/ads4015/ssl_project/zeroshot_inference/segmentation/submit_zeroshot_segmentation.sh - Submit zero-shot segmentation inference tasks to SLURM array job.

set -euo pipefail

ROOT=/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches
OUT=/midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_zeroshot/tasks.txt

python /home/ads4015/ssl_project/zeroshot_inference/segmentation/build_zeroshot_tasks.py \
  --root "$ROOT" \
  --out "$OUT"

N=$(wc -l < "$OUT")

if [[ "$N" -eq 0 ]]; then
  echo "[ERROR] No tasks generated"
  exit 1
fi

echo "[INFO] Submitting $N zeroshot tasks"

sbatch --array=0-$((N-1)) \
  /home/ads4015/ssl_project/zeroshot_inference/segmentation/zeroshot_segmentation_array_job.sh "$OUT"
