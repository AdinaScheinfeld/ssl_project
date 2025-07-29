# Binary segmentation module for finetuning Wu model using Selma data

# --- Setup ---

# imports
import os
import wandb

import pytorch_lightning as pl

from monai.data.meta_tensor import MetaTensor
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR

import torch
from torch.serialization import add_safe_globals


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
            add_safe_globals([MetaTensor])
            state_dict = torch.load(pretrained_ckpt, weights_only=False, map_location='cpu')['state_dict']
            encoder_weights = {k.replace('encoder.', 'model.'): v for k, v in state_dict.items() if k.startswith('encoder.')}
            self.model.load_state_dict(encoder_weights, strict=False)
            print(f'[INFO] Successfully loaded pretrained checkpoint from: {pretrained_ckpt}', flush=True)

        # create variables to save best train loss and best val loss
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')

    # forward pass
    def forward(self, x):
        return self.model(x)
    
    # training step
    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['image'])
        train_loss = self.loss_fn(logits, batch['label'].float()) # compute loss
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.best_train_loss = min(self.best_train_loss, train_loss.item())
        return train_loss
    
    # val step
    def validation_step(self, batch, batch_idx):

        # print(f'[DEBUG] batch["image"].shape: {batch["image"].shape}', flush=True)
        
        logits = self.forward(batch['image'])
        val_loss = self.loss_fn(logits, batch['label'].float()) # compute loss
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        self.best_val_loss = min(self.best_val_loss, val_loss.item())

        # log some images to wandb
        if self.logged_images < 5:
            preds = torch.sigmoid(logits) > 0.5
            images = batch['image'][:, 0:1] # use only first channel
            num_to_log = min(5 - self.logged_images, images.shape[0])

            for i in range(num_to_log):

                # log center slice
                img_np = images[i, 0].cpu().numpy()
                lbl_np = batch['label'][i].cpu().numpy().squeeze()
                pred_np = preds[i, 0].cpu().numpy()
                mid = img_np.shape[0] // 2

                # print(f'Image: {img_np.shape}, Label: {lbl_np.shape}, Pred: {pred_np.shape}', flush=True)

                self.val_table.add_data(
                    batch['filename'][i],
                    wandb.Image(img_np[mid]),
                    wandb.Image(lbl_np[mid]),
                    wandb.Image(pred_np[mid])
                )
                self.logged_images += 1

        return val_loss
    
    # on validation epoch start
    def on_validation_epoch_start(self):
        
        # create counter and wandb table to log images after validation epoch
        self.logged_images = 0
        self.val_table = wandb.Table(columns=['Filename', 'Image', 'Label', 'Prediction'])
    

    # on validation epoch end
    def on_validation_epoch_end(self):
        
        # log val image table to wandb
        if self.logged_images > 0:
            self.logger.experiment.log({f'val_examples_{self.current_epoch}': self.val_table})

    # configure optimizers
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    
    # after complete training
    def on_fit_end(self):

        # log best losses
        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.logger.experiment.log({
                'best_train_loss': self.best_train_loss,
                'best_val_loss': self.best_val_loss,
                'final_checkpoint_path': os.path.join(
                    self.trainer.checkpoint_callback.dirpath,
                    self.trainer.checkpoint_callback.filename
                )
            })
    










