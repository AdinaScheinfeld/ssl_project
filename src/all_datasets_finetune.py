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

# function to get desired classes for finetuning
def _wanted_dir(p, include=None, exclude=None):

    name = p.name.lower()

    # include only certain classes
    if include:
        if not any(tok.lower() in name for tok in include):
            return False
        
    # exclude certain classes
    if exclude:
        if any(tok.lower() in name for tok in exclude):
            return False
        
    return True


# function to list subdirs based on include/exclude classes
def _list_subdirs(root, include=None, exclude=None):
    return [p for p in root.iterdir() if p.is_dir() and _wanted_dir(p, include, exclude)]


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

        # ensure image has correct dimensions (1, D, H, W)
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.shape[0] != 1:
            raise ValueError(f'Expected image to have 1 channel, but got {image.shape[0]} channels in file {pt_path}')

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

    # add wandb summarization variables
    try:
        wandb_logger.experiment.define_metric("train_loss", summary="min")
        wandb_logger.experiment.define_metric("val_loss", summary="min")
        wandb_logger.experiment.define_metric("val_dice_best", summary="max")
    except Exception:
        pass

    # create train/val datasets
    train_root = Path(config['patch_dir']) / 'train'
    val_root = Path(config['patch_dir']) / 'val'
    include_classes = config.get('include_classes')
    exclude_classes = config.get('exclude_classes')
    train_subdirs = _list_subdirs(train_root, include=include_classes, exclude=exclude_classes)
    val_subdirs = _list_subdirs(val_root, include=include_classes, exclude=exclude_classes)

    # indicate which classes are being used
    print(f'[INFO] Finetuning using the following classes:', flush=True)
    if include_classes:
        print(f'       Included: {include_classes}', flush=True)
    if exclude_classes:
        print(f'       Excluded: {exclude_classes}', flush=True)
    print(f'[INFO] Found {len(train_subdirs)} training subdirectories and {len(val_subdirs)} validation subdirectories.', flush=True)
    for p in train_subdirs:
        print(f'       Train: {p.name}', flush=True)
    for p in val_subdirs:
        print(f'       Val:   {p.name}', flush=True)

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
        freeze_encoder_epochs=config.get('freeze_encoder_epochs', 0),
        encoder_lr_mult=config.get('encoder_lr_mult', 0.5),
        loss_name=config.get('loss_name', 'dicece')
        )

    # setup callbacks
    ckpt_val = ModelCheckpoint(
        monitor='val_dice_050', 
        save_top_k=1, 
        mode='max', 
        filename=config['checkpoint_filename'],
        dirpath=config['checkpoint_dirpath']
    )


    early_stopping_callback = EarlyStopping(
        monitor='val_dice_050', 
        patience=config['early_stopping_patience'], 
        mode='max'
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















