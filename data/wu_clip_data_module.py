# wu_clip_data_module.py - Data module for Wu data with CLIP

# --- Setup ---

# imports
import glob
import os
import random
import sys

import torch
from torch.utils.data import DataLoader

from pytorch_lightning import LightningDataModule

from monai.transforms import Compose as MonaiCompose

# get dataset class
sys.path.append('/home/ads4015/ssl_project/data/')
from nifti_text_patch_dataset import NiftiTextPatchDataset

# get functions from other files
sys.path.append('/home/ads4015/ssl_project/src/')
from wu_transforms import get_train_transforms, get_val_transforms, get_load_transforms


# --- Data Module ---

# datamodule class for wu data
class WuCLIPDataModule(LightningDataModule):

    # init
    def __init__(self, data_dir, batch_size, train_frac, seed, data_subset_frac, text_prompts, 
                 use_sub_patches=False, base_patch_size=96, sub_patch_size=64, downsample_to=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.seed = seed
        self.data_subset_frac = data_subset_frac
        self.text_prompts = text_prompts
        self.use_sub_patches = use_sub_patches
        self.base_patch_size = base_patch_size
        self.sub_patch_size = sub_patch_size
        self.downsample_to = downsample_to
        
        # warn if batch size is too small
        if self.use_sub_patches and self.batch_size < 16:
            print(f'[WARNING] Using sub-patches with small batch size ({self.batch_size}). Consider increasing batch size.', flush=True)

    # setup
    def setup(self, stage=None):

        # determine actual input size
        if self.use_sub_patches: # crop to specific size
            target_size = int(self.sub_patch_size)
            mode_msg = f'Using {target_size}^3 sub-patches (crops).'
        elif self.downsample_to is not None: # downsample to specific size
            target_size = int(self.downsample_to)
            mode_msg = f'Downsampling full {self.base_patch_size}^3 -> {target_size}^3 (no cropping).'
        else: # no cropping or downsampling
            target_size = int(self.base_patch_size)
            mode_msg = f'Using {target_size}^3 base patches.'
        print(f'[INFO] {mode_msg}', flush=True)

        # get volume directories
        volume_dirs = sorted(glob.glob(os.path.join(self.data_dir, '*/input')))
        if not volume_dirs:
            raise FileNotFoundError(f'No input folders found under {self.data_dir}')
        
        # get train/val split
        torch.manual_seed(self.seed)
        random.shuffle(volume_dirs)
        split_idx = int(self.train_frac * len(volume_dirs))
        train_dirs = volume_dirs[:split_idx]
        val_dirs = volume_dirs[split_idx:]

        # get list of train/val directories
        self.train_volume_names = [os.path.basename(os.path.dirname(p)) for p in train_dirs]
        self.val_volume_names = [os.path.basename(os.path.dirname(p)) for p in val_dirs]

        # function to collect all files in a list of directories
        def collect_files(dirs):
            files = []
            for d in dirs:
                volume_files = sorted(glob.glob(os.path.join(d, '*.nii.gz')))
                random.shuffle(volume_files)
                n_keep = int(len(volume_files) * self.data_subset_frac)
                files.extend(volume_files[:n_keep])
            return files
        
        # collect train/val files
        train_files = collect_files(train_dirs)
        val_files = collect_files(val_dirs)

        # print debugging and info
        print(f'[DEBUG] Found {len(train_files)} train and {len(val_files)} val patches from {len(train_dirs)} train and {len(val_dirs)} val volumes.', flush=True)
        print(f'[INFO] Train volumes: {self.train_volume_names}', flush=True)
        print(f'[INFO] Val volumes: {self.val_volume_names}', flush=True)
        print(f'[INFO] Subsample fraction: {self.data_subset_frac} => {len(train_files)} train and {len(val_files)} val files used.')

        # create train/val datasets
        load = get_load_transforms(target_size=target_size)
        self.train_ds = NiftiTextPatchDataset(train_files, 
                                              transforms=MonaiCompose([load, get_train_transforms()]),
                                              text_prompts=self.text_prompts,
                                              use_sub_patches=self.use_sub_patches,
                                              base_patch_size=self.base_patch_size,
                                              sub_patch_size=self.sub_patch_size)
        self.val_ds = NiftiTextPatchDataset(val_files, 
                                            transforms=MonaiCompose([load, get_val_transforms()]),
                                            text_prompts=self.text_prompts,
                                            use_sub_patches=self.use_sub_patches,
                                            base_patch_size=self.base_patch_size,
                                            sub_patch_size=self.sub_patch_size)

    # train dataloader
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    # val dataloader
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)






        