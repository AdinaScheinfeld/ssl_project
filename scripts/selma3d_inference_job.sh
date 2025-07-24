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


# indicate starting
echo "Starting the inference script..."


# activate conda environment
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# Run inference on folder
# python /home/ads4015/ssl_project/src/selma3d_inference.py \
#   --input_path /midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_unlabeled_val \
#   --checkpoint /home/ads4015/ssl_project/lsm_finetune/vtmxkihh/checkpoints/best.ckpt \
#   --output_dir /midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_unlabeled_val_preds

# python /home/ads4015/ssl_project/src/selma3d_inference.py \
#   --input_path /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/AA1-PO-C-R45/aa1-po-c-r45_p1.nii.gz \
#   --checkpoint /home/ads4015/ssl_project/lsm_finetune/vtmxkihh/checkpoints/best.ckpt \
#   --output_dir /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/AA1-PO-C-R45

# # 2
# python /home/ads4015/ssl_project/src/selma3d_inference.py \
#   --input_path /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/AE2-WF2a_A/ae2-wf2a_a.nii.gz \
#   --checkpoint /home/ads4015/ssl_project/lsm_finetune/vtmxkihh/checkpoints/best.ckpt \
#   --output_dir /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/AE2-WF2a_A/

# # 3
# python /home/ads4015/ssl_project/src/selma3d_inference.py \
#   --input_path /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/AJ12-LG1E-n_A/aj12-lg1e-n_a.nii.gz \
#   --checkpoint /home/ads4015/ssl_project/lsm_finetune/vtmxkihh/checkpoints/best.ckpt \
#   --output_dir /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/AJ12-LG1E-n_A/

# # 4
# python /home/ads4015/ssl_project/src/selma3d_inference.py \
#   --input_path /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/AZ10-SR3B-6_A/az10-sr3b-6_a_p1.nii.gz \
#   --checkpoint /home/ads4015/ssl_project/lsm_finetune/vtmxkihh/checkpoints/best.ckpt \
#   --output_dir /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/AZ10-SR3B-6_A/

# # 5
# python /home/ads4015/ssl_project/src/selma3d_inference.py \
#   --input_path /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/CH1-PCW1A_A/ch1-pcw1a_a.nii.gz \
#   --checkpoint /home/ads4015/ssl_project/lsm_finetune/vtmxkihh/checkpoints/best.ckpt \
#   --output_dir /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/CH1-PCW1A_A/



# all wu lab brains

# 1
python /home/ads4015/ssl_project/src/selma3d_inference.py \
  --input_path /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/01_AA1-PO-C-R45/ \
  --checkpoint /home/ads4015/ssl_project/lsm_finetune/vtmxkihh/checkpoints/best.ckpt \
  --output_dir /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/01_AA1-PO-C-R45/preds

# 2
python /home/ads4015/ssl_project/src/selma3d_inference.py \
  --input_path /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/02_AE2-WF2a_A/ \
  --checkpoint /home/ads4015/ssl_project/lsm_finetune/vtmxkihh/checkpoints/best.ckpt \
  --output_dir /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/02_AE2-WF2a_A/preds

# 3
python /home/ads4015/ssl_project/src/selma3d_inference.py \
  --input_path /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/03_AJ12-LG1E-n_A/ \
  --checkpoint /home/ads4015/ssl_project/lsm_finetune/vtmxkihh/checkpoints/best.ckpt \
  --output_dir /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/03_AJ12-LG1E-n_A/preds

# 4
python /home/ads4015/ssl_project/src/selma3d_inference.py \
  --input_path /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/04_AE2-WF2a_A/ \
  --checkpoint /home/ads4015/ssl_project/lsm_finetune/vtmxkihh/checkpoints/best.ckpt \
  --output_dir /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/04_AE2-WF2a_A/preds

# 5
python /home/ads4015/ssl_project/src/selma3d_inference.py \
  --input_path /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/05_CH1-PCW1A_A/ \
  --checkpoint /home/ads4015/ssl_project/lsm_finetune/vtmxkihh/checkpoints/best.ckpt \
  --output_dir /midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/05_CH1-PCW1A_A/preds


# indicate completion
echo "Inference script completed successfully."




