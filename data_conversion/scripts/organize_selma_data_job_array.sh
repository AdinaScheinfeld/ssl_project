#!/bin/bash
#SBATCH --job-name=organize_selma
#SBATCH --output=logs/organize_selma_%A_%a.out
#SBATCH --error=logs/organize_selma_%A_%a.err
#SBATCH --partition=minilab-cpu
#SBATCH --array=0-34
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G


# indicate starting
echo "Beginning image organization script $SLURM_ARRAY_TASK_ID ..."

# load modules
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# run python script
python /home/ads4015/ssl_project/data_conversion/src/organize_selma_data.py --index $SLURM_ARRAY_TASK_ID

# indicate completion
echo "Image organization complete."





