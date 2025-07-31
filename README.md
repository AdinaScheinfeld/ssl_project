# SSL Project

## Scripts

- **Preprocessing**
    - SELMA 3D
        - [selma3d_get_patches.py](/preprocess_patches/src/selma3d_get_patches.py) - Script to extract small patches of unannotated selma3d data for pretraining.
        - [selma3d_get_patches_job_array](/preprocess_patches/scripts/selma3d_get_patches_job_array.sh) - Slurm scrip to parallelize extraction of patches for pretraining.
        - [visualize_selma3d_train_transforms.ipynb](/preprocess_patches/notebooks/visualize_selma3d_train_transforms.ipynb) - Notebook to visualize transforms for unannotated selma3d pretraining data. Also downloads patches for pretraining.
    - Wu
        - [visualize_wu_train_transforms.ipynb](/preprocess_patches/notebooks/visualize_wu_train_transforms.ipynb) - Notebook to visualize transforms for Wu brain images. All visualizations in notebook. Does not download anything. 


- **Configs**
    - SELMA 3D
        - [selma3d_pretrain_config.yaml](/configs/selma3d_pretrain_config.yaml) - Pretrain config for Selma3D data.
    - Wu
        - [wu_pretrain_config.yaml](/configs/wu_pretrain_config.yaml) - Pretrain config for Wu data.
    - Wu (CLIP)
        - [wu_clip_pretrain_config.yaml](/configs/wu_clip_pretrain_config.yaml) - Pretrain config for Wu data using CLIP. 


- **Pretraining**
    - SELMA 3D
        - [selma3d_pretrain.py](/src/selma3d_pretrain.py) - Pretraining script using unannotated selma3d data. 
        - [selma3d_pretrain_job.sh](/scripts/selma3d_pretrain_job.sh) - Slurm script to run pretraining python script. 
        - [selma3d_visualization_functions.py](/preprocess_patches/src/selma3d_visualization_functions.py) - Functions for preprocessing and pretraining files and scripts.
    - Wu
        - [wu_pretrain.py](/src/wu_pretrain.py) - Pretraining script using unannotated Wu data.
        - [wu_pretrain_job.sh](/scripts/wu_pretrain_job.sh) - Slurm script to run pretraining python script.
        - [wu_visualization_functions.py](/preprocess_patches/src/wu_visualization_functions.py) - Functions for preprocessing and pretraining files and scripts.
        - [wu_transforms.py](/src/wu_transforms.py) - Transforms for pretraining and finetuning.
    - Wu (CLIP)
        - [wu_clip_pretrain.py](/src/wu_clip_pretrain.py) - Pretraining script for unannotated Wu data for CLIP. 
        - [wu_clip_pretrain_job.sh](/scripts/wu_clip_pretrain_job.sh) - Slurm script to run pretraining python script for Wu CLIP. 


- **Data Classes**
    - Wu (CLIP)
        - [nifti_text_patch_dataset.py](/data/nifti_text_patch_dataset.py) - Dataset for Nifti images with accompanying text for Wu CLIP pretraining. 
        - [wu_clip_data_module.py](/data/wu_clip_data_module.py) - Data module for Wu train/val data for CLIP pretraining. 


- **Models**
    - Wu
        - [ibot_pretrain_module.py](/models/ibot_pretrain_module.py) - IBOT pretraining model (used for Wu pretraining in wu_pretrain.py).
        - [binary_segmentation_module.py](/models/binary_segmentation_module.py) - Binary segmentation model (used to finetune Wu model using Selma3D data).
    - Wu (CLIP)
        - [ibot_clip_pretrain_module](/models/ibot_clip_pretrain_module.py) - IBOT + CLIP pretraining module for Wu data. 


- **Finetuning**
    - SELMA 3D
        - [selma3d_extract_finetune_patches.py](/src/selma3d_extract_finetune_patches.py) - Python script to extract patches of annotated selma3d data to finetune model
        - [selma3d_extract_finetune_patches_job.sh](/scripts/selma3d_extract_finetune_patches_job.sh) - Slurm script to extract annotated finetuning patches. 
        - [selma3d_extract_finetune.yaml](/configs/selma3d_extract_finetune.yaml) - Config file for finetuning patches extraction.
        - [selma3d_split_finetune_patches.py](/src/selma3d_split_finetune_patches.py) - Python script to split annotated finetuning patches into train/val sets
        - [selma3d_split_finetune_patches_job.sh](/scripts/selma3d_split_finetune_patches_job.sh) - Slurm script to split finetune patches. 
        - [selma3d_finetune.py](/src/selma3d_finetune.py) - Python script to finetune model.
        - [selma3d_finetune_job.sh](/scripts/selma3d_finetune_job.sh) - Job to run finetuning script.
        - [selma3d_finetune_config.yaml](/configs/selma3d_finetune_config.yaml) - Config file for finetuning.
    - Wu
        - [wu_transforms.py](/src/wu_transforms.py) - Transforms for pretraining and finetuning.
        - [wu_finetune.py](/src/wu_finetune.py) - Script to finetune Wu model using Selma3D data.
        - [wu_finetune_job.sh](/scripts/wu_finetune_job.sh) - Shell script to run finetuning for Wu model.
        - [wu_finetune_config.yaml](/configs/wu_finetune_config.yaml) - Config file for Wu finetuning.


- **Inference**
    - SELMA 3D
        - [selma3d_inference.py](/src/selma3d_inference.py) - Script to run inference on a single file or on all files in a folder.
        - [selma3d_inference_job.sh](/scripts/selma3d_inference_job.sh) - Job to run inference script. 


## Files
- [unannotated_ab_plaque](/midtier/paetzollab/scratch/ads4015/data_selma3d/unannotated_ab_plaque), [unannotated_cfos](/midtier/paetzollab/scratch/ads4015/data_selma3d/unannotated_cfos), [unannotated_nucleus](/midtier/paetzollab/scratch/ads4015/data_selma3d/unannotated_nucleus), [unannotated_vessel](/midtier/paetzollab/scratch/ads4015/data_selma3d/unannotated_vessel) - Unannoted selma3d data for pretraining
- [small_patches](/midtier/paetzollab/scratch/ads4015/data_selma3d/small_patches) - Small patches of unannotated selma3d data for pretraining
- [selma3d_unannotated_pretrain_patches](/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_unannotated_pretrain_patches) - Unannotated transformed train/val patches for pretraining
- [SELMA3D](/midtier/paetzollab/scratch/ads4015/data_selma3d/SELMA3D) - Annotated selma3d patches for finetuning
- [selma3d_finetune_patches](/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches) - Patches extracted from annoted selma3d volumes to finetune model
- [selma3d_finetune_patches_split](/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches_split) - Selma3d finetune patches split into tran/val sets
- [selma3d_unlabeled_val](/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_unlabeled_val) - Selma3d unlabeled validation data<br><br>
- [all_wu_brain_patches](/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches) - Folder containing subfolders with all brains from Wu lab.
    - [01_AA1-PO-C-R45](/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/01_AA1-PO-C-R45) - TH protein
    - [02_AZ10-SR3B-6_A](/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/02_AZ10-SR3B-6_A) - CTIP2 protein
    - [03_AJ12-LG1E-n_A](/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/03_AJ12-LG1E-n_A) - GFAP protein
    - [04_AE2-WF2a_A](/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/04_AE2-WF2a_A) - P75NTR
    - [05_CH1-PCW1A_A](/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/05_CH1-PCW1A_A) - DBH protein
