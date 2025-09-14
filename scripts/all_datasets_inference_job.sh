#!/bin/bash
#SBATCH --job-name=lsm_inference
#SBATCH --output=logs/lsm_inference_%j.out
#SBATCH --error=logs/lsm_inference_%j.err
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1


# all_datasets_inference_job.sh - SLURM job script to run inference on a specified dataset using a pre-trained model checkpoint.


# indicate starting
echo "Starting the inference script..."


# activate conda environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1


# # Run inference on file
# python /home/ads4015/ssl_project/src/all_datasets_inference.py \
#   --input_path /midtier/paetzollab/scratch/ads4015/allen_human/sub-MITU01_ses-20210521h17m17s06_sample-178_stain-NPY_run-1_chunk-1_SPIM.nii.gz \
#   --checkpoint /home/ads4015/ssl_project/checkpoints/finetune_pretrained-v24.ckpt \
#   --output_dir /midtier/paetzollab/scratch/ads4015/temp


# Run inference on file
python /home/ads4015/ssl_project/src/all_datasets_inference.py \
  --input_path /midtier/paetzollab/scratch/ads4015/cellseg3d_data/data/c1image.tif \
  --checkpoint /home/ads4015/ssl_project/checkpoints/finetune_pretrained-v24.ckpt \
  --output_dir /midtier/paetzollab/scratch/ads4015/temp





# indicate completion
echo "Inference script completed successfully."




