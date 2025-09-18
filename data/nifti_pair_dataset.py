# nifti_pair_dataset.py - A dataset class for handling pairs of NIfTI images and labels

# imports
# imports
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import nibabel as nib
import numpy as np
import os
from pathlib import Path
import random
import sys
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger


# dataset (loads nifti volumes and returns tensors)
class NiftiPairDataset(Dataset):

    # init
    def __init__(self, pairs, augment=None):
        self.pairs = pairs
        self.augment = augment
    
    # length
    def __len__(self):
        return len(self.pairs)
    
    # get nifti
    @staticmethod
    def _load_nii(path):
        nii = nib.load(str(path))
        arr = nii.get_fdata(dtype=np.float32)
        return arr, nii.affine, nii.header
    
    # conver to tensor
    def _to_tensor(self, arr):

        # ensure (1, D, H, W) shape
        if arr.ndim == 3:
            vol = arr[None, ...]  # add channel dim
        elif arr.ndim == 4 and arr.shape[-1] == 1:
            vol = np.transpose(arr, (3, 0, 1, 2))  # move channel to front
        else:
            raise ValueError(f'Unsupported array shape: {arr.shape}')

        # normalize per volume
        lo, hi = np.percentile(vol, (1.0, 99.0))
        if hi > lo:  # avoid div by zero
            vol = np.clip((vol - lo) / (hi - lo), 0.0, 1.0)
        else:
            vol = np.zeros_like(vol)

        # to tensor
        return torch.from_numpy(vol.astype(np.float32))
    
    # get item
    def __getitem__(self, idx):

        # get image and label pair
        pair = self.pairs[idx]
        img_np, img_affine, img_header = self._load_nii(pair.image)
        lbl_np, lbl_affine, lbl_header = self._load_nii(pair.label)

        # ensure correct dimensions
        if lbl_np.ndim == 3:
            lbl_np = lbl_np[None, ...]  # add channel dim
        elif lbl_np.ndim == 4 and lbl_np.shape[-1] == 1:
            lbl_np = np.transpose(lbl_np, (3, 0, 1, 2))  # move channel to front
        lbl_np = (lbl_np > 0.5).astype(np.float32)  # binarize

        # return image and label tensors
        img_tensor = self._to_tensor(img_np)
        lbl_tensor = torch.from_numpy(lbl_np.astype(np.float32))

        # return dict
        return {
            'image': img_tensor,
            'label': lbl_tensor,
            'filename': str(pair.image),
            'affine': torch.from_numpy(img_affine.astype(np.float32))
        }
    






