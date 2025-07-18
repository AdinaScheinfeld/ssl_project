#!/bin/bash
#SBATCH --job-name=extract_patches
#SBATCH --output=logs/extract_patches_%A_%a.out
#SBATCH --error=logs/extract_patches_%A_%a.err
#SBATCH --array=0-3
#SBATCH --partition=minilab-cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=48:00:00


# indicate starting
echo "Starting patch extraction..."


# activate conda environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# define the path to the config file
CONFIG_PATH="/home/ads4015/ssl_project/configs/selma3d_extract_finetune.yaml"

# Map array index to class
CLASS_NAMES=("brain_amyloid_plaque_patches" "brain_c_fos_positive_patches" "brain_cell_nucleus_patches" "brain_vessels_patches")
CLASS_NAME=${CLASS_NAMES[$SLURM_ARRAY_TASK_ID]}

echo "Processing class: $CLASS_NAME"
python /home/ads4015/ssl_project/src/selma3d_extract_finetune_patches.py --config $CONFIG_PATH --class_name $CLASS_NAME


# indicate completion
echo "Finishing patch extraction."

