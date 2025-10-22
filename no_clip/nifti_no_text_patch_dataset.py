# nifti_no_text_patch_dataset.py - dataset class for loading nifti image patches without text captions

# --- Setup ---

# imports
import glob
import nibabel as nib
import os
import random
import torch
from torch.utils.data import Dataset

from monai.transforms import Compose as MonaiCompose, LoadImaged


# --- NiftiNoTextPatchDataset class ---
class NiftiNoTextPatchDataset(Dataset):

    # init
    def __init__(self, file_paths, transforms=None, use_sub_patches=False, base_patch_size=96, sub_patch_size=64):

        # ensure that file paths exist
        if not file_paths:
            raise ValueError("No file paths provided to NiftiNoTextPatchDataset.")
        
        self.file_paths = list(file_paths)
        self.transforms = transforms
        self.use_sub_patches = use_sub_patches
        self.base_patch_size = base_patch_size
        self.sub_patch_size = sub_patch_size

        # if using sub patches, ensure sub patch size is smaller than base patch size
        if self.use_sub_patches:
            self.sub_patches = []
            for p in self.file_paths:
                vol = nib.load(p).get_fdata()
                s = self.sub_patch_size

                # extract all possible sub patches from volume
                starts = [
                    (0,0,0), (s//2,0,0), (0,s//2,0), (0,0,s//2),
                    (s//2,s//2,0), (s//2,0,s//2), (0,s//2,s//2), (s//2,s//2,s//2)
                ]

                # add sub patches to list
                for (x,y,z) in random.sample(starts, 2):
                    sub = vol[z:z+s, y:y+s, x:x+s]
                    self.sub_patches.append((sub, p)) # store sub patch with original file path

    # len
    def __len__(self):
        return len(self.sub_patches) if self.use_sub_patches else len(self.file_paths)
    
    # get item
    def __getitem__(self, idx):
        if self.use_sub_patches:
            image_np, path = self.sub_patches[idx]
            data = {'image': image_np, 'path': path}
            return self.transforms(data) if self.transforms else data
        
        else:
            path = self.file_paths[idx]
            data = {'image': path, 'path': path}
            return self.transforms(data) if self.transforms else data








