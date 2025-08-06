# Wu Data Pretraining

# --- Setup ---

# imports
import argparse
# import glob
# import matplotlib.pyplot as plt
# import numpy as np
import os
# import random
import sys
import wandb
import yaml

# from monai.networks.nets import SwinUNETR
# from monai.transforms import Compose as MonaiCompose

import pytorch_lightning as pl
# from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import torch
# from torch import nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
# from torchvision.transforms import Compose

# # get functions from other files
# sys.path.append('/home/ads4015/ssl_project/src/')
# from wu_transforms import get_train_transforms, get_val_transforms, get_load_transforms

# get ibot pretraining module
sys.path.append('/home/ads4015/ssl_project/models')
from ibot_clip_pretrain_module import IBOTCLIPPretrainModule

# # get dataset class
# sys.path.append('/home/ads4015/ssl_project/data/')
# from nifti_text_patch_dataset import NiftiTextPatchDataset

# get data module
sys.path.append('/home/ads4015/ssl_project/data/')
from wu_clip_data_module import WuCLIPDataModule


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

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
    datamodule = WuCLIPDataModule(
        data_dir=config['data']['data_dir'],
        batch_size=config['data']['batch_size'],
        train_frac=config['data']['train_frac'],
        seed=config['training']['seed'],
        data_subset_frac=config['data']['data_subset_frac']
    )

    # initialize model
    model = IBOTCLIPPretrainModule(config)

    # callbacks
    callbacks = [
        # callback for early stopping
        # early stopping is triggered when loss does not decrease for `patience` consecutive epochs
        EarlyStopping(monitor='val_loss', patience=config['training']['patience'], mode='min'),

        # callback for checkpointing
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            filename=config['model']['save_filename'],
            dirpath=config['model']['save_dirpath'],
            verbose=True
        )
    ]

    # pytorch lightning trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        logger=wandb_logger,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=config['training']['log_every_n_steps'],
        callbacks=callbacks
    )

    # start training
    trainer.fit(model, datamodule=datamodule)

    # log best results
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_val_loss = trainer.checkpoint_callback.best_model_score
    wandb_logger.experiment.log({
        'best_val_loss': best_val_loss.item() if best_val_loss else None,
        # 'best_train_loss': best_train_loss,
        'best_model_path': best_model_path
    })

    print(f'[INFO] Best model saved to : {best_model_path}')
    print(f'[INFO] Best val loss: {best_val_loss}')


        
    




















