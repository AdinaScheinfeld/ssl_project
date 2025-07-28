# SSL Project

## Scripts

- Preprocessing
    - [selma3d_get_patches.py](/home/ads4015/ssl_project/preprocess_patches/src/selma3d_get_patches.py) - Script to extract small patches of unannotated selma3d data for pretraining.
    - [selma3d_get_patches_job_array](/home/ads4015/ssl_project/preprocess_patches/scripts/selma3d_get_patches_job_array.sh) - Slurm scrip to parallelize extraction of patches for pretraining.
    - [visualize_selma3d_train_transforms.ipynb](/home/ads4015/ssl_project/notebooks/visualize_selma3d_train_transforms.ipynb) - Notebook to visualize transforms for unannotated selma3d pretraining data. Also downloads patches for pretraining.

    - [visualize_wu_train_transforms.ipynb](/home/ads4015/ssl_project/preprocess_patches/notebooks/visualize_wu_train_transforms.ipynb) - Notebook to visualize transforms for Wu brain images. All visualizations in notebook. Does not download anything. 

- Pretraining
    - [selma3d_pretrain.py](/home/ads4015/ssl_project/src/selma3d_pretrain.py) - Pretraining script using unannotated selma3d data. 
    - [selma3d_pretrain_job.sh](/home/ads4015/ssl_project/scripts/selma3d_pretrain_job.sh) - Slurm script to run pretraining python script. 
    - [selma3d_pretrain_config.yaml](/home/ads4015/ssl_project/configs/selma3d_pretrain_config.yaml) - Pretrain config for Selma3D data.
    - [selma3d_visualization_functions.py](/home/ads4015/ssl_project/preprocess_patches/src/selma3d_visualization_functions.py) - Functions for preprocessing and pretraining files and scripts.

    - [wu_pretrain.py](/home/ads4015/ssl_project/src/wu_pretrain.py) - Pretraining script using unannotated Wu data.
    - [wu_pretrain_job.sh](/home/ads4015/ssl_project/scripts/wu_pretrain_job.sh) - Slurm script to run pretraining python script.
    - [wu_pretrain_config.yaml](/home/ads4015/ssl_project/configs/wu_pretrain_config.yaml) - Pretrain config for Wu data.
    - [wu_visualization_functions.py](/home/ads4015/ssl_project/preprocess_patches/src/wu_visualization_functions.py) - Functions for preprocessing and pretraining files and scripts.

- Models

    - [ibot_pretrain_module.py](/home/ads4015/ssl_project/models/ibot_pretrain_module.py) - IBOT pretraining model (used for Wu pretraining in wu_pretrain.py).

- Finetuning
    - [selma3d_extract_finetune_patches.py](/home/ads4015/ssl_project/src/selma3d_extract_finetune_patches.py) - Python script to extract patches of annotated selma3d data to finetune model
    - [selma3d_extract_finetune_patches_job.sh](/home/ads4015/ssl_project/scripts/selma3d_extract_finetune_patches_job.sh) - Slurm script to extract annotated finetuning patches. 
    - [selma3d_extract_finetune.yaml](/home/ads4015/ssl_project/configs/selma3d_extract_finetune.yaml) - Config file for finetuning patches extraction.
    - [selma3d_split_finetune_patches.py](/home/ads4015/ssl_project/src/selma3d_split_finetune_patches.py) - Python script to split annotated finetuning patches into train/val sets
    - [selma3d_split_finetune_patches_job.sh](/home/ads4015/ssl_project/scripts/selma3d_split_finetune_patches_job.sh) - Slurm script to split finetune patches. 
    - [selma3d_finetune.py](/home/ads4015/ssl_project/src/selma3d_finetune.py) - Python script to finetune model.
    - [selma3d_finetune_job.sh](/home/ads4015/ssl_project/scripts/selma3d_finetune_job.sh) - Job to run finetuning script.
    - [selma3d_finetune_config.yaml](/home/ads4015/ssl_project/configs/selma3d_finetune_config.yaml) - Config file for finetuning.

- Inference
    - [selma3d_inference.py](/home/ads4015/ssl_project/src/selma3d_inference.py) - Script to run inference on a single file or on all files in a folder.
    - [selma3d_inference_job.sh](/home/ads4015/ssl_project/scripts/selma3d_inference_job.sh) - Job to run inference script. 

## Files
- [unannotated_ab_plaque](/midtier/paetzollab/scratch/ads4015/data_selma3d/unannotated_ab_plaque), [unannotated_cfos](/midtier/paetzollab/scratch/ads4015/data_selma3d/unannotated_cfos), [unannotated_nucleus](/midtier/paetzollab/scratch/ads4015/data_selma3d/unannotated_nucleus), [unannotated_vessel](/midtier/paetzollab/scratch/ads4015/data_selma3d/unannotated_vessel) - Unannoted selma3d data for pretraining
- [small_patches](/midtier/paetzollab/scratch/ads4015/data_selma3d/small_patches) - Small patches of unannotated selma3d data for pretraining
- [selma3d_unannotated_pretrain_patches](/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_unannotated_pretrain_patches) - Unannotated transformed train/val patches for pretraining
- [SELMA3D](/midtier/paetzollab/scratch/ads4015/data_selma3d/SELMA3D) - Annotated selma3d patches for finetuning
- [selma3d_finetune_patches](/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches) - Patches extracted from annoted selma3d volumes to finetune model
- [selma3d_finetune_patches_split](/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches_split) - Selma3d finetune patches split into tran/val sets
-[selma3d_unlabeled_val](/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_unlabeled_val) - Selma3d unlabeled validation data

## To do:

- [ ] Read papers
- [ ] Wu data
- [ ] New data from Johannes