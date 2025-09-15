#!/bin/bash
#SBATCH --job-name=mask_comparison
#SBATCH --output=logs/mask_comparison_%j.out
#SBATCH --error=logs/mask_comparison_%j.err
#SBATCH --partition=minilab-cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1


# mask_comparisons_job.sh - SLURM job script to compare segmentation masks


# indicate starting
echo "Starting the mask comparison script..."


# activate conda environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1


# run mask comparison
python /home/ads4015/ssl_project/src/mask_comparisons.py \
    --preds_dir /midtier/paetzollab/scratch/ads4015/cellseg3d_data/preds \
    --suffix_a pretrained-v24 \
    --suffix_b pretrained-v25 \
    --labels_dir /midtier/paetzollab/scratch/ads4015/cellseg3d_data/labels \
    --out_csv /midtier/paetzollab/scratch/ads4015/temp/mask_comparison_results.csv \
    --save_disagreements /midtier/paetzollab/scratch/ads4015/temp/ \
    --disagreement_ext nii.gz \
    --label_name_replacements "image->labels_new_label"


# indicate completion
echo "Mask comparison script completed successfully."






