#!/bin/bash
#SBATCH --job-name=cellpose_cv
#SBATCH --output=/midtier/paetzollab/scratch/ads4015/compare_methods/cellpose/cross_val/logs/cellpose_cv_%A_%a.out
#SBATCH --error=/midtier/paetzollab/scratch/ads4015/compare_methods/cellpose/cross_val/logs/cellpose_cv_%A_%a.err
#SBATCH --partition=scu-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-65

# /home/ads4015/ssl_project/compare_methods/cellpose/cellpose_experiment_launcher.sh


# indicate starting
echo "Starting Cellpose training/evaluation for split ${SLURM_ARRAY_TASK_ID} at $(date)"


# load environment
module load anaconda3/2022.10-34zllqw
source activate cellpose-env2

SPLIT_DIR="/midtier/paetzollab/scratch/ads4015/compare_methods/cellpose/cross_val/splits" # path to splits
SPLIT_FILE=$(ls $SPLIT_DIR/split_*.json | sed -n "$((SLURM_ARRAY_TASK_ID+1))p") # get split file for this array task

# run training and evaluation for this split
python /home/ads4015/ssl_project/compare_methods/cellpose/cellpose_experiment_train_val_test.py --split-json "$SPLIT_FILE"


# indicate ending
echo "Finished Cellpose training/evaluation for split ${SLURM_ARRAY_TASK_ID} at $(date)"


