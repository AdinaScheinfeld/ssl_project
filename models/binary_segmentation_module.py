# binary_segmentation_module.py - Binary segmentation module for finetuning model using Selma data

# --- Setup ---

# imports
import os
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from monai.data.meta_tensor import MetaTensor
from monai.losses import DiceCELoss
from monai.networks.nets import SwinUNETR

import torch
from torch.serialization import add_safe_globals


# --- Model ---

class BinarySegmentationModule(pl.LightningModule):

    # init
    def __init__(self, pretrained_ckpt=None, lr=1e-4, feature_size=48, freeze_encoder_epochs=0):
        super().__init__()
        self.save_hyperparameters()

        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.encoder_frozen = freeze_encoder_epochs > 0

        # trackers/counters
        self.logged_images = 0 # counter for number of logged images in validation step
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        self.best_ckpt = None

        # define model
        self.model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            feature_size=feature_size,
            use_checkpoint=True
        )
        self.lr = lr
        self.loss_fn = DiceCELoss(sigmoid=True)

        # if within number of epochs that encoder is frozen, don't use gradients
        if self.encoder_frozen:
            for name, param in self.model.named_parameters():
                if name.startswith('swinViT'):
                    param.requires_grad = False

        # load pretrained weights if provided
        if pretrained_ckpt:
            add_safe_globals([MetaTensor])
            state_dict = torch.load(pretrained_ckpt, weights_only=False, map_location='cpu')['state_dict']

            # accept either student_encoder. or encoder. prefixes and map to SwinUNETR
            encoder_weights = {}
            for k, v in state_dict.items():
                if k.startswith('student_encoder.'):
                    encoder_weights[k[len('student_encoder.'):]] = v
                elif k.startswith('encoder.'):
                    encoder_weights[k[len('encoder.'):]] = v

            # map common swin keys to monai swinUNETR
            mapped = {}
            for k, v in encoder_weights.items():
                mapped[k] = v


            # handle 1/2 channel step (duplicating weights if needed)
            stem_key = 'swinViT.patch_embed.proj.weight'
            if stem_key in mapped:
                w = mapped[stem_key]
                if w.ndim == 5: # (out_channels, in_channels, kD, kH, kW)
                    out_c, in_c, kD, kH, kW = w.shape

                    # if model has 2 input channels but weights have 1, duplicate weights
                    if in_c == 1 and self.model.swinViT.patch_embed.proj.weight.shape[1] == 2:
                        w2 = torch.cat([w, w], dim=1) / 2.0
                        mapped[stem_key] = w2
                        print(f'[INFO] Duplicated stem weights from 1 to 2 channels for {stem_key}', flush=True)

            # filter before loading state dict
            model_sd = self.model.state_dict()
            safe_mapped = {}
            dropped = []

            for k, v in mapped.items():
                if k not in model_sd:
                    dropped.append((k, 'not in model'))
                    continue
                if v.shape == model_sd[k].shape:
                    safe_mapped[k] = v
                else:

                    # special case - expand from 1 -> 2 channels
                    if v.ndim == 5 and v.shape[1] == 1 and model_sd[k].shape[1] == 2:
                        v2 = torch.cat([v, v], dim=1) / 2.0
                        safe_mapped[k] = v2
                        print(f'[INFO] Duplicated stem weights from 1 to 2 channels for {k}', flush=True)
                    else:
                        dropped.append((k, f'ckpt {tuple(v.shape)} vs model {tuple(model_sd[k].shape)}'))

            incompatible = self.model.load_state_dict(safe_mapped, strict=False)

            print(f'[INFO] Loaded pretrained encoder with kept={len(safe_mapped)}, dropped={len(dropped)}, missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}', flush=True)

            if dropped:
                print('[WARN] Dropped due to shape mismatch:', flush=True)
                for k, msg in dropped[:10]:
                    print(' ', k, '->', msg, flush=True)
                if len(dropped) > 10:
                    print(f' ... (+{len(dropped)-10} more)', flush=True)

        # indicate if no pretrained checkpoint provided
        else:
            print(f'[WARN] No pretrained checkpoint provided, training from scratch.', flush=True)

    # forward pass
    def forward(self, x):
        return self.model(x)


    # on train epoch start
    def on_train_epoch_start(self):

        # unfreeze encoder after specified number of epochs
        if self.encoder_frozen and self.current_epoch >= self.freeze_encoder_epochs:
            print(f'[INFO] Unfreezing encoder at epoch {self.current_epoch}', flush=True)
            for name, param in self.model.named_parameters():
                param.requires_grad = True
            self.encoder_frozen = False

    
    # training step
    def training_step(self, batch, batch_idx):
        logits = self.forward(batch['image'])
        train_loss = self.loss_fn(logits, batch['label'].float()) # compute loss
        batch_size = batch['image'].shape[0]
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.best_train_loss = min(self.best_train_loss, float(train_loss.detach().cpu()))
        return train_loss
    
    # val step
    def validation_step(self, batch, batch_idx):
        
        logits = self.forward(batch['image'])
        val_loss = self.loss_fn(logits, batch['label'].float()) # compute loss
        batch_size = batch['image'].shape[0]
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.best_val_loss = min(self.best_val_loss, val_loss.item())

        # log some images to wandb
        if self.logged_images < 5:

            # prepare tensors for visualization (detach metatensor, move to cpu, cast to float)
            preds = torch.sigmoid(logits.detach().to(dtype=torch.float32)) > 0.5
            preds = preds.to(dtype=torch.float32)
            images = batch['image'][:, 0:1] # use only first channel for visualization
            labels = batch['label']
    
            # drop metatensor wrapper if present
            if isinstance(images, MetaTensor):
                images = images.as_tensor()
            if isinstance(labels, MetaTensor):
                labels = labels.as_tensor()
            if isinstance(preds, MetaTensor):
                preds = preds.as_tensor()

            # move to cpu float32 before numpy
            images = images.detach().to(device='cpu', dtype=torch.float32, copy=True)
            labels = labels.detach().to(device='cpu', dtype=torch.float32, copy=True)
            preds = preds.detach().to(device='cpu', dtype=torch.float32, copy=True)

            num_to_log = min(5 - self.logged_images, images.shape[0])

            for i in range(num_to_log):

                # log center slice
                img_np = images[i, 0].numpy()
                lbl_np = labels[i].numpy().squeeze()
                pred_np = preds[i, 0].numpy()
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

        # find best checkpoint from callbacks
        for cb in self.trainer.callbacks:
            if isinstance(cb, pl.callbacks.ModelCheckpoint):
                self.best_ckpt = getattr(cb, 'best_model_path', None)
                break

        payload = {
            'best_train_loss': self.best_train_loss,
            'best_val_loss': self.best_val_loss
        }
        if self.best_ckpt:
            payload['best_model_path'] = self.best_ckpt


        # print best metrics
        print(f'[INFO] Training complete:', flush=True)
        for k, v in payload.items():
            print(f'  {k}: {v}', flush=True)

        # log to wandb summary
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.summary.update(payload)
    










