# Getting started with nn-Unet

1. Create directories. Ex: `/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/{nnUNet_raw,nnUNet_preprocessed,nnUNet_results}`

2. Set env variables. Ex: 
```
export nnUNet_raw="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/nnUNet_raw"
export nnUNet_preprocessed="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/nnUNet_preprocessed"
export nnUNet_results="/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/nnUNet_results"
```

3. Create dataset. From the command line, run: `python /home/ads4015/ssl_project/compare_methods/nnunet/nnunet_build_dataset.py`

4. Run preprocessing and planning. From the command line, run: `nnUNetv2_plan_and_preprocess -d 1 --verify_dataset_integrity`

5. Start training on server by submitting: `sbatch /home/ads4015/ssl_project/compare_methods/nnunet/nnunet_train_folds.sh`


