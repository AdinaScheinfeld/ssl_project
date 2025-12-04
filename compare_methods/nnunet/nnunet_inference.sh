#!/bin/bash
#SBATCH --job-name=nnunet_infer
#SBATCH --output=logs/nnunet_infer_%j.out
#SBATCH --error=logs/nnunet_infer_%j.err
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=7-00:00:00

# /home/ads4015/ssl_project/compare_methods/nnunet/nnunet_inference.sh


# indicate starting
echo "Starting nnUNet inference at $(date)"


# set environment variables
export nnUNet_raw="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/nnUNet_preprocessed"
export nnUNet_results="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/nnUNet_results"


# load environment
module load anaconda3/2022.10-34zllqw
source activate nnunet2-env1

# run inference
OUTDIR=/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/test_predictions

nnUNetv2_predict \
    -d 1 \
    -i $nnUNet_raw/Dataset001_LSM/imagesTs \
    -o $OUTDIR \
    -c 3d_fullres \
    -f 0 1 2 3 4


# indicate ending
echo "Finished nnUNet inference at $(date)"

