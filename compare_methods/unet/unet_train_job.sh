#!/bin/bash
#SBATCH --job-name=selma_unet
#SBATCH --partition=sablab-gpu
#SBATCH --account=sablab
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/compare_methods/unet/logs/slurm_%j.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/compare_methods/unet/logs/slurm_%j.err

set -euo pipefail

# ---- avoid NFS .nfsXXXX temp cleanup crashes ----
export TMPDIR="/tmp/${USER}/${SLURM_JOB_ID}"
mkdir -p "$TMPDIR"
chmod 700 "$TMPDIR"

# also helpful for torch dataloader stability on some clusters
export PYTHONWARNINGS="ignore"

# ---- paths ----
DATA_DIR="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches/cell_nucleus_patches"
OUT_ROOT="/midtier/paetzollab/scratch/ads4015/compare_methods/unet"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/preds" "${OUT_ROOT}/checkpoints" "${OUT_ROOT}/splits"

# ---- environment ----
# adjust these to your cluster conventions
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# (recommended) make W&B write inside OUT_ROOT/logs
export WANDB_DIR="${OUT_ROOT}/logs"
export WANDB_CACHE_DIR="${OUT_ROOT}/logs/.wandb_cache"
export WANDB_CONFIG_DIR="${OUT_ROOT}/logs/.wandb_config"

# Optional: if compute nodes have restricted internet, you can use offline mode:
# export WANDB_MODE=offline

python -u /home/ads4015/ssl_project/compare_methods/unet/unet_train.py \
  --data_dir "${DATA_DIR}" \
  --out_root "${OUT_ROOT}" \
  --epochs 100 \
  --batch_size 2 \
  --num_workers 4 \
  --lr 1e-3 \
  --weight_decay 1e-4 \
  --seed 100 \
  --roi_size "96,96,96" \
  --wandb_project "selma3d_unet" \
  --run_name "cell_nucleus_unet" \
  --early_stop_patience 3 \
