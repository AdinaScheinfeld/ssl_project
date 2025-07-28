# Binary segmentation module for finetuning Wu model using Selma data

# --- Setup ---

# imports
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

    # forward pass
    def forward(self, x):
        return self.model(x)
    
    # training step
    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['image'])
        train_loss = self.loss_fn(logits, batch['label'].float()) # compute loss
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True)
        return train_loss
    
    # val step
    def validation_step(self, batch, batch_idx):

        print(f'[DEBUG] batch["image"].shape: {batch["image"].shape}', flush=True)
        
        logits = self.forward(batch['image'])
        val_loss = self.loss_fn(logits, batch['label'].float()) # compute loss
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True)
        return val_loss
    
    # configure optimizers
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    










