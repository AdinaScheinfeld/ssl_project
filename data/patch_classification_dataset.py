# patch_classification_dataset.py - Dataset class for multiclass patch-based image classification tasks

# --- Setup ---

# imports
import nibabel as nib
import numpy as np
from pathlib import Path
import random

import torch
from torch.utils.data import Dataset


# --- Helper Functions ---

# function to load nifti and return array
def _safe_load_nii(path):
    img = nib.load(str(path))
    arr = img.get_fdata(dtype=np.float32)
    return arr, img

# function to min-max normalize image to [0, 1]
def _min_max_normalize(patch, eps=1e-6):
    arr_min = float(np.min(patch))
    arr_max = float(np.max(patch))
    if arr_max - arr_min < eps:
        return np.zeros_like(patch, dtype=np.float32)
    return ((patch - arr_min) / (arr_max - arr_min)).astype(np.float32)

# function to apply random augmentations
def _maybe_augment(x):

    # 3d flips
    if random.random() < 0.5:
        x = torch.flip(x, dims=[-1]) # W flip
    if random.random() < 0.5:
        x = torch.flip(x, dims=[-2]) # H flip
    if random.random() < 0.5:
        x = torch.flip(x, dims=[-3]) # D flip

    return x

# dataset class
class PatchClassificationDataset(Dataset):

    # init
    def __init__(self, samples, augment=False, channel_substr='ALL'):
        self.samples = list(samples)
        self.augment = augment

        # normalize channel filters
        self.substrings = None
        s = str(channel_substr).strip()
        if s and s.upper() != 'ALL':
            self.substrings = [sub.strip().lower() for sub in s.split(',') if sub.strip()]

        if self.substrings is not None:
            self.samples = [rec for rec in self.samples if any(tok in rec['path'].name.lower() for tok in self.substrings)]

    # length
    def __len__(self):
        return len(self.samples)
    
    # get item
    def __getitem__(self, idx):

        # get sample record
        rec = self.samples[idx]

        # get path and label index
        path = rec['path']
        y = int(rec['label_idx'])

        # load patch
        vol, _ = _safe_load_nii(path)
        vol = _min_max_normalize(vol) # normalize
        x = torch.from_numpy(vol).unsqueeze(0)

        # augment if specified
        if self.augment:
            x = _maybe_augment(x)

        return {
            'image': x, # tensor of shape (C=1, D, H, W)
            'label': torch.tensor(y, dtype=torch.long), # class id
            'filename': str(path.name), # original filename
            'abs_path': str(path.resolve()), # absolute path
            'label_name': rec['label_name'] # class name
        }















