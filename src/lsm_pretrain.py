# Script for self-supervised pretraining with iBOT masking and DINO distillation

# --- Setup ---

# imports
import argparse
import glob
import matplotlib.pyplot as plt
from monai.networks.nets import SwinUNETR
import numpy as np
import os
import random
import tifffile as tiff
import wandb
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose

# set matmul precision for better performance on tensor core gpus
torch.set_float32_matmul_precision('medium')

# load config from yaml file
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


# --- Load Images ---

# class for loading 3d tiff patches
class TiffPatchDataset(Dataset):

    # init
    def __init__(self, file_paths, patch_size=96):
        self.file_paths = file_paths
        self.patch_size = patch_size
        self.transforms = Compose([
            lambda x: np.expand_dims(tiff.imread(x), axis=0), # add channel dimension, (D, H, W) -> (1, D, H, W)
            lambda x: torch.tensor(x, dtype=torch.float32) # convert to float32 tensor
        ])

    # get number of files in dataset
    def __len__(self):
        return len(self.file_paths)
    
    # load and preprocess a .tif image
    def __getitem__(self, idx):
        img = self.transforms(self.file_paths[idx])
        return img
    

# --- Model ---

class IBOTPretrainModule(pl.LightningModule):

    # init
    def __init__(self, image_size=96, mask_ratio=0.6, lr=1e-4, ema_decay=0.996): ## ADD TO CONFIG - mask ratio, lr
        super().__init__()
        self.save_hyperparameters() # save all hyperparameters to self.hparams
        self.mask_ratio = mask_ratio # percentage of voxels to mask
        self.lr = lr # learning rate

        # create swinUNETR as student network
        self.encoder = nn.Sequential(
            SwinUNETR(
                img_size=(image_size, image_size, image_size),
                in_channels=1,
                out_channels=1, # dummy output for reconstruction
                feature_size=48, ## ADD TO CONFIG - feature_size
                use_checkpoint=True
            ),
            nn.Sigmoid() # constrain output to [0, 1]
        )

        # ema teacher network 
        # (same architecture as student, updated using running average of student model's parameters)
        self.teacher = nn.Sequential(
            SwinUNETR(
                img_size=(image_size, image_size, image_size),
                in_channels=1,
                out_channels=1, # dummy output for reconstruction
                feature_size=48, ## ADD TO CONFIG - feature_size
                use_checkpoint=False      
            ),
            nn.Sigmoid()      
        )
        for p in self.teacher.parameters():
            p.requires_grad = False # freeze teacher weights

        self.register_buffer('ema_decay', torch.tensor(ema_decay)) # ema decay factor (0.996 is iBOT default)

        # MSE loss (L2 loss): sensitive to large error, strongly penalizes outliers, 
        # encourages smoother reconstruction bec squaring diff emphasizes large deviations,
        # can lead to blurry outputs if used alone
        # MAE loss (L1 loss): robust to outliers,
        # helps preserve edges and fine contrast better than MSE alone
        # can introduce sparsity in output (good for patchy structures like vessels/plaques)
        self.loss_fn = lambda x, y: 0.8 * nn.MSELoss()(x, y) + 0.2 * nn.L1Loss()(x, y)

    # forward pass through student encoder
    def forward(self, x):
        return self.encoder(x)
    
    # training step
    def training_step(self, batch, batch_idx):

        # input volume, shape (B, 1, D, H, W)
        x = batch

        # generate random binary mask based on mask_ratio
        mask = torch.rand_like(x) < self.mask_ratio # create boolean mask (determined by whether random value is < mask_ratio)
        x_masked = x.clone()
        x_masked[mask] = 0

        # student sees masked input
        student_features = self.encoder(x_masked)

        # teacher sees full input (no gradients)
        with torch.no_grad():
            teacher_features = self.teacher(x)

        # self-distillation loss (match student output to teacher output)
        train_loss = self.loss_fn(student_features, teacher_features.detach()) # teacher model does not receive any gradient updates
        self.log('train_loss', train_loss, prog_bar=True, on_step=True, on_epoch=True)

        # log original, masked, and student prediction images to wanb every 100 steps
        if self.global_step % 100 == 0 and x.shape[0] > 0:

            # get center slices
            center_z = x.shape[2] // 2
            original_img = x[0, 0].cpu().numpy()[center_z]
            masked_img = x_masked[0, 0].cpu().numpy()[center_z]
            pred_img = student_features[0, 0].detach().cpu().numpy()[center_z]

            # create plot
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(original_img, cmap='gray')
            axs[0].set_title('Original')
            axs[1].imshow(masked_img, cmap='gray')
            axs[1].set_title('Masked Input')
            axs[2].imshow(pred_img, cmap='gray')
            axs[2].set_title('Student Output')
            
            # format plot
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()

            # log images to wandb
            self.logger.experiment.log({
                'Input vs. Masked vs. Output': wandb.Image(fig),
                'global_step': self.global_step
            })
            plt.close(fig)



        return train_loss
    
    # validation step
    def validation_step(self, batch, batch_idx):

        # input volume, shape (B, 1, D, H, W)
        x = batch

        # generate mask and apply it to create masked input for student
        mask = torch.rand_like(x) < self.mask_ratio
        x_masked = x.clone()
        x_masked[mask] = 0

        # get output from student and teacher
        student_features = self.encoder(x_masked)
        with torch.no_grad():
            teacher_features = self.teacher(x)

        # compute validation loss between student and teacher outputs
        val_loss = self.loss_fn(student_features, teacher_features.detach())
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)
        return val_loss
    
    # after backward
    def on_after_backward(self):

        # update ema teacher parameters from student
        for student_param, teacher_param in zip(self.encoder.parameters(), self.teacher.parameters()):
            teacher_param.data = self.ema_decay * teacher_param.data + (1 - self.ema_decay) * student_param.data

    # optimizer
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr) ## ADD TO CONFIG - torch.optim
    

# --- Data Loading ---

def get_data_loaders(train_dir, val_dir, batch_size, train_samp_size, val_samp_size, seed): ## ADD TO CONFIG - batch_size (depends on gpu memory and model complexity-feature_size)

    # set random seed for reproducibility
    random.seed(seed)

    # get lists of files
    train_files = sorted(glob.glob(os.path.join(train_dir, '*.tif')))
    val_files = sorted(glob.glob(os.path.join(val_dir, '*.tif')))

    # subsample a fixed number of patches
    train_files = random.sample(train_files, train_samp_size)
    val_files = random.sample(val_files, val_samp_size)
    print(f'Using {len(train_files)} train samples and {len(val_files)} val samples.')

    # create datasets
    train_ds = TiffPatchDataset(train_files)
    val_ds = TiffPatchDataset(val_files)

    # print dataset sizes and steps per epoch for debugging
    print(f'Num train samples: {len(train_ds)}')
    print(f'Num val samples: {len(val_ds)}')
    print(f'Train steps per epoch: {len(train_ds) // batch_size}')

    return DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4), \
           DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4)


# --- Main Entry Point ---

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

    # create dataloaders
    train_loader, val_loader = get_data_loaders(
        config['data']['train_dir'],
        config['data']['val_dir'],
        batch_size=config['data']['batch_size'],
        train_samp_size=config['data']['train_samp_size'],
        val_samp_size=config['data']['val_samp_size'],
        seed=config['training']['seed']
    )

    # initialize model
    model = IBOTPretrainModule(
        image_size=config['model']['image_size'],
        mask_ratio=config['model']['mask_ratio'],
        lr=config['model']['lr'],
        ema_decay=config['model']['ema_decay']
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

    # pytorch lightning trainer ## ADD TO CONFIG - max_epochs
    trainer = pl.Trainer(
        max_epochs=100,
        logger=wandb_logger,
        accelerator='gpu',
        devices=1,
        log_every_n_steps=5,
        callbacks=[callback_early_stopping, callback_checkpoint]
    )

    # start training
    trainer.fit(model, train_loader, val_loader)

        
    














