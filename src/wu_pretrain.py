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

# get dataset
sys.path.append('/home/ads4015/ssl_project/data')
from nifti_patch_dataset import NiftiPatchDataset
from wu_data_module import WuDataModule


# set matmul precision for better performance on tensor core gpus
torch.set_float32_matmul_precision('medium')


# load config from yaml file
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    

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
        filename=config['model']['save_filename'],
        dirpath=config['model']['save_dirpath'],
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

        
    




















