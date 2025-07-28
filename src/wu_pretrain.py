# Wu Data Pretraining

# --- Setup ---

# imports
import argparse
import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
import wandb
import yaml

from monai.networks.nets import SwinUNETR
from monai.transforms import Compose as MonaiCompose

import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

# get functions from other files
sys.path.append('/home/ads4015/ssl_project/src/')
from wu_transforms import get_train_transforms, get_val_transforms, get_load_transforms

# get ibot pretraining module
sys.path.append('/home/ads4015/ssl_project/models')
from ibot_pretrain_module import IBOTPretrainModule


# set matmul precision for better performance on tensor core gpus
torch.set_float32_matmul_precision('medium')


# load config from yaml file
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    

# --- Dataset Class ---

# nifti patch dataset class
class NiftiPatchDataset(Dataset):

    # init
    def __init__(self, file_paths, transforms=None):
        self.file_paths = file_paths
        self.transforms = transforms

    # length
    def __len__(self):
        return len(self.file_paths)
    
    # getter
    def __getitem__(self, idx):
        data = {'image': self.file_paths[idx]}
        if self.transforms:
            data = self.transforms(data)
        return data
    

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


# --- Main Entry Point --- 

# main
if __name__ == '__main__':

    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml file')
    args = parser.parse_args()
    config = load_config(args.config)

    # set seed for reproducibility
    pl.seed_everything(config['training']['seed'])

    # wandb logger
    wandb_logger = WandbLogger(project=config['training']['project_name'])

    # initialize data module
    datamodule = WuDataModule(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        train_frac=config['data']['train_frac'],
        seed=config['training']['seed']
    )

    # initialize model
    model = IBOTPretrainModule(
        image_size=config['model']['image_size'],
        mask_ratio=config['model']['mask_ratio'],
        lr=config['model']['lr'],
        ema_decay=config['model']['ema_decay'],
        mask_patch_size=config['model']['mask_patch_size'],
        temp_student=config['model']['temp_student'],
        temp_teacher=config['model']['temp_teacher']
    )

    # callback for early stopping
    # early stopping is triggered when loss does not decrease for `patience` consecutive epochs
    callback_early_stopping = EarlyStopping(monitor='val_loss', patience=config['training']['patience'], mode='min')

    # callback for checkpointing
    callback_checkpoint = ModelCheckpoint(
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        filename='best-val-loss',
        verbose=True
    )

    # pytorch lightning trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        logger=wandb_logger,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=config['training']['log_every_n_steps'],
        callbacks=[callback_early_stopping, callback_checkpoint]
    )

    # start training
    trainer.fit(model, datamodule=datamodule)

        
    




















