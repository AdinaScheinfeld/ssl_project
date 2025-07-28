# Script to finetune foundation model using annotated SELMA3D patches

# --- Setup ---

# imports
import argparse
import os 
from pathlib import Path
import yaml
import wandb

from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# set matmul precision for better performance on tensor core gpus
torch.set_float32_matmul_precision('medium')


# --- Dataset ---

# create custom dataset class
class SegPatchDataset(Dataset):

    # init
    def __init__(self, root_dir):
        self.paths = list(Path(root_dir).rglob('*.pt'))

    # get image-label pair
    def __getitem__(self, idx):

        # load item
        item = torch.load(self.paths[idx])
        image = item['image'] # shape: (1 or 2, D, H, W)
        label = item['label'] # shape: (D, H, W)

        # ensure correct dimensions
        if image.ndim == 3:
            image = image.unsqueeze(0)  # (1, D, H, W)
        if image.shape[0] == 1:
            image = torch.cat([image, torch.zeros_like(image)], dim=0)  # pad to (2, D, H, W)

        # return image and label as dict
        return {'image': image.float(), 'label': label.long()}
    
    # get length of dataset
    def __len__(self):
        return len(self.paths)
    

# --- Model ---

class BinarySegmentationModule(pl.LightningModule):

    # init
    def __init__(self, pretrained_ckpt=None, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        # define model
        self.model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            feature_size=48,
            use_checkpoint=True
        )
        self.lr = lr
        self.loss_fn = DiceCELoss(sigmoid=True)

        # load pretrained weights if provided
        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt)['state_dict']
            encoder_weights = {k.replace('encoder.', 'model.'): v for k, v in state_dict.items() if k.startswith('encoder.')}
            self.model.load_state_dict(encoder_weights, strict=False)

    # forward pass
    def forward(self, x):
        return self.model(x)
    
    # training step
    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['image'])
        train_loss = self.loss_fn(logits, batch['label'].unsqueeze(1).float()) # compute loss
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True)
        return train_loss
    
    # val step
    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch['image'])
        val_loss = self.loss_fn(logits, batch['label'].unsqueeze(1).float()) # compute loss
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        return val_loss
    
    # configure optimizers
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    

# --- Main entry point ---

if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config YAML file')
    args = parser.parse_args()

    # load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # set seed for reproducibility
    pl.seed_everything(config['seed'])

    # initialize logger
    wandb_logger = WandbLogger(project=config['wandb_project'])

    # load train/val data
    train_loader = DataLoader(SegPatchDataset(Path(config['patch_dir']) / 'train'), batch_size=config['batch_size'], shuffle=True, num_workers=4)
    val_loader = DataLoader(SegPatchDataset(Path(config['patch_dir']) / 'val'), batch_size=config['batch_size'], shuffle=False, num_workers=4)

    # initialize model
    model = BinarySegmentationModule(pretrained_ckpt=config.get('pretrained_ckpt', None), lr=config['lr'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config['early_stopping_patience'], mode='min'),
        ModelCheckpoint(monitor='val_loss', save_top_k=1, mode='min', filename='best2')
    ]

    # train
    trainer = pl.Trainer(max_epochs=config['max_epochs'],
                         logger=wandb_logger,
                         callbacks=callbacks,
                         accelerator='gpu',
                         devices=1,
                         log_every_n_steps=5)
    
    # fit model
    trainer.fit(model, train_loader, val_loader)












