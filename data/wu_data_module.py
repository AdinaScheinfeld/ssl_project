# Wu Data Module

# --- Setup ---

# imports
import glob
import os
import random
import sys

from monai.transforms import Compose as MonaiCompose

from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader

# get functions from other files
sys.path.append('/home/ads4015/ssl_project/src/')
from wu_transforms import get_train_transforms, get_val_transforms, get_load_transforms

# get dataset
sys.path.append('/home/ads4015/ssl_project/data')
from nifti_patch_dataset import NiftiPatchDataset

# --- DataModule Class ---

# datamodule class for wu data
class WuDataModule(LightningDataModule):

    # init
    def __init__(self, data_dir, batch_size, train_frac, seed):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_frac = train_frac
        self.seed = seed

    # setup
    def setup(self, stage=None):

        # get volume directories
        volume_dirs = sorted(glob.glob(os.path.join(self.data_dir, '*/input')))
        if not volume_dirs:
            raise FileNotFoundError(f'No output folders found under {self.data_dir}')
        
        # get train/val split
        random.seed(self.seed)
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
                files.extend(glob.glob(os.path.join(d, '*.nii.gz')))
            return sorted(files)
        
        # collect train/val files
        train_files = collect_files(train_dirs)
        val_files = collect_files(val_dirs)

        # print debugging and info
        print(f'[DEBUG] Found {len(train_files)} train and {len(val_files)} val patches from {len(train_dirs)} train and {len(val_dirs)} val volumes.')
        print(f'[INFO] Train volumes: {self.train_volume_names}')
        print(f'[INFO] Val volumes: {self.val_volume_names}')

        # create train/val datasets
        load = get_load_transforms()
        self.train_ds = NiftiPatchDataset(train_files, transforms=MonaiCompose([load, get_train_transforms()]))
        self.val_ds = NiftiPatchDataset(val_files, transforms=MonaiCompose([load, get_val_transforms()]))

    # train dataloader
    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=4)
    
    # val dataloader
    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

