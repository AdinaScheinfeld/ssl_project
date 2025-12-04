#!/bin/bash
#SBATCH --job-name=nnunet_infer
#SBATCH --output=logs/nnunet_infer_%A.out
#SBATCH --error=logs/nnunet_infer_%A.err
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=12:00:00

DTYPE=$1
POOL=$2

BASE=/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/cross_val/$DTYPE/pool_${POOL}
CV0=$BASE/cv0

export nnUNet_raw=$CV0/nnUNet_raw
export nnUNet_preprocessed=$CV0/nnUNet_preprocessed
export nnUNet_results=$CV0/nnUNet_results

OUTDIR=$BASE/preds
mkdir -p $OUTDIR

module load anaconda3/2022.10-34zllqw
source activate nnunet2-env1

nnUNetv2_predict \
    -d 1 \
    -i $nnUNet_raw/Dataset001_LSM/imagesTs \
    -o $OUTDIR \
    -c 3d_fullres \
    -f 0 1 2
