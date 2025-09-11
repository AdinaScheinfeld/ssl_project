# all_datasets_finetune.py - Script to use Selma data to finetune model created using all datasets

# --- Setup ---

# imports
import argparse
from datetime import datetime
import numpy as np
import os 
from pathlib import Path
import sys
import time
import yaml
import wandb

from monai.data.meta_tensor import MetaTensor

import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback, EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset, get_worker_info
from torch.serialization import add_safe_globals


# get functions from other files
sys.path.append('/home/ads4015/ssl_project/src/')
from all_datasets_transforms import get_finetune_train_transforms, get_finetune_val_transforms

# get binary segmentation module for finetuning
sys.path.append('/home/ads4015/ssl_project/models')
from binary_segmentation_module import BinarySegmentationModule


# set matmul precision for better performance on tensor core gpus
torch.set_float32_matmul_precision('medium')


# --- Dataset ---

# dataset for finetuning
class SegPatchDataset(Dataset):

    # init
    def __init__(self, root_dir, transforms):
        self.paths = list(Path(root_dir).rglob('*.pt'))
        self.transforms = transforms

    # get item
    def __getitem__(self, idx):

        # get image-label pair
        pt_path = self.paths[idx]
        item = torch.load(pt_path, map_location='cpu') # load to cpu first

        # separate image and label
        image = item['image'] # shape: (1 or 2, D, H, W)
        label = item['label'] # shape: (D, H, W)

        # print(f'[DEBUG] Loaded: {pt_path.name} | Image shape before: {image.shape}, label shape before: {label.shape}', flush=True)

        # ensure image has correct dimensions
        # vessel images have 2 channels, all other images have 1 channel
        # so need to pad all other images with a dummy channel to ensure consistency in training
        if image.ndim == 3:
            image = image.unsqueeze(0)  # (1, D, H, W)
        if image.shape[0] == 1:
            image = torch.cat([image, torch.zeros_like(image)], dim=0)  # pad to (2, D, H, W)

        # ensure that label has correct dimensions
        if label.ndim == 3:
            label = label.unsqueeze(0)  # add channel dimension (1, D, H, W)

        # ensure raw torch.Tensor not MetaTensor
        if isinstance(image, MetaTensor):
            image = image.as_tensor()
        if isinstance(label, MetaTensor):
            label = label.as_tensor()  

        # print(f'[DEBUG] Loaded: {pt_path.name} | Image shape after channels: {image.shape}, label shape after channels: {label.shape}', flush=True)

        # add image and label to dict
        data = {'image': image, 'label': label, 'filename': str(self.paths[idx])}

        # print(f"[DEBUG] {pt_path} | image shape: {image.shape}, label shape: {label.shape}", flush=True)

        # transform data
        data = self.transforms(data)

        # print(f"[DEBUG] Loaded: {pt_path.name} | Image shape after transforms: {data['image'].shape}, label shape after transforms: {data['label'].shape}", flush=True)
        # print(f"[DEBUG] Type after transform: image={type(data['image'])}, label={type(data['label'])}", flush=True)


        # return augmented data
        return data
    
    # length
    def __len__(self):
        return len(self.paths)

# function to seed dataloader workers to make them reproducible
def _seed_worker(worker_id):

    info = get_worker_info()
    if info is not None:
        base_seed = torch.initial_seed() % 2**31
        np.random.seed(base_seed + worker_id)
        random_state = np.random.RandomState(base_seed + worker_id)
    

# --- Main Entry Point ---

if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    # load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # start script timer
    _t0 = time.perf_counter()
    _start_local = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    print(f'[INFO] Script started at {_start_local}', flush=True)

    # set seed for reproducibility
    pl.seed_everything(config['seed'])

    # initialize logger
    wandb_logger = WandbLogger(project=config['wandb_project'])

    # create train/val datasets
    train_root = Path(config['patch_dir']) / 'train'
    val_root = Path(config['patch_dir']) / 'val'
    train_subdirs = [p for p in train_root.iterdir() if p.is_dir()]
    val_subdirs = [p for p in val_root.iterdir() if p.is_dir()]

    train_ds = torch.utils.data.ConcatDataset([
        SegPatchDataset(p, get_finetune_train_transforms()) for p in train_subdirs
    ])
    val_ds = torch.utils.data.ConcatDataset([
        SegPatchDataset(p, get_finetune_val_transforms()) for p in val_subdirs
    ])

    # load train/val datasets
    num_workers = min(4, os.cpu_count() or 4)

    # load train/val data
    train_loader = DataLoader(train_ds, 
                              batch_size=config['batch_size'], 
                              shuffle=True, 
                              num_workers=num_workers, 
                              pin_memory=True, 
                              persistent_workers=True, 
                              worker_init_fn=_seed_worker)
    val_loader = DataLoader(val_ds, 
                            batch_size=config['batch_size'], 
                            shuffle=False, 
                            num_workers=num_workers, 
                            pin_memory=True, 
                            persistent_workers=True, 
                            worker_init_fn=_seed_worker)

    # initialize model
    model = BinarySegmentationModule(
        pretrained_ckpt=config.get('pretrained_ckpt', None), 
        lr=config['lr'],
        feature_size=config['feature_size'],
        freeze_encoder_epochs=config.get('freeze_encoder_epochs', 0)
        )

    # setup callbacks
    ckpt_val = ModelCheckpoint(
        monitor='val_loss', 
        save_top_k=1, 
        mode='min', 
        filename=config['checkpoint_filename'],
        dirpath=config['checkpoint_dirpath']
    )


    early_stopping_callback = EarlyStopping(
        monitor='val_loss', 
        patience=config['early_stopping_patience'], 
        mode='min'
    )

    callbacks = [early_stopping_callback, ckpt_val]

    # train
    trainer = pl.Trainer(max_epochs=config['max_epochs'],
                         logger=wandb_logger,
                         callbacks=callbacks,
                         accelerator='gpu',
                         devices=1,
                         precision='bf16-mixed',
                         log_every_n_steps=5)
    
    # fit model
    trainer.fit(model, train_loader, val_loader)

    # end script timer
    _t1 = time.perf_counter()
    _elapsed = _t1 - _t0
    _end_local = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    _h = int(_elapsed // 3600); _m = int((_elapsed % 3600) // 60); _s = int(_elapsed % 60)
    print(f'[INFO] Script ended at {_end_local}', flush=True)
    print(f'[INFO] Script runtime: {_h:02d}:{_m:02d}:{_s:02d} ({_elapsed:.2f}s)', flush=True)















