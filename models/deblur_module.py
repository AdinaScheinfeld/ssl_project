# /home/ads4015/ssl_project/models/deblur_module.py - PyTorch module for image deblurring

# --- Setup ---

# imports
import wandb

from monai.data.meta_tensor import MetaTensor
from monai.losses import SSIMLoss
from monai.networks.nets import SwinUNETR

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import add_safe_globals


# --- DeblurModule class ---
class DeblurModule(pl.LightningModule):

    # init
    def __init__(
            self,
            pretrained_ckpt_path=None, # path to pretrained checkpoint (optional)
            lr=1e-4, # base learning rate
            feature_size=24, # feature size for Swin UNETR
            encoder_lr_mult=0.05, # learning rate multiplier for encoder
            freeze_encoder_epochs=0, # number of epochs to freeze encoder at start of training
            weight_decay=1e-5, # weight decay for optimizer
    ):
        
        super().__init__()
        self.save_hyperparameters()

        # hold wandb table of val samples for the current epoch
        self.val_examples_table = None

        # model
        self.model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=1,
            out_channels=1,
            feature_size=feature_size,
            use_checkpoint=True
        )

        # encoder freeze schedule
        self.freeze_encoder_epochs = int(freeze_encoder_epochs)
        self.encoder_frozen = self.freeze_encoder_epochs > 0

        # losses and lr
        self.l1_loss = nn.L1Loss(reduction='mean')
        self.ssim_loss = SSIMLoss(spatial_dims=3, data_range=1.0)
        self.edge_weight = 0.1 # weight for edge loss
        self.highfreq_weight = 0.1 # weight for high freq loss
        self.lr = float(lr)
        self.encoder_lr_mult = float(encoder_lr_mult)

        # load pretrained weights if provided
        if pretrained_ckpt_path:
            self._load_pretrained_weights(pretrained_ckpt_path)

    # function to load pretrained weights
    def _load_pretrained_weights(self, ckpt_path):

        # load checkpoint with safe globals
        add_safe_globals([MetaTensor])
        ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
        state_dict = ckpt.get('state_dict', ckpt)

        # function to strip prefixes
        def _strip_prefix(key):
            for p in ('model.', 'module.', 'student_model.', 'student.', 'teacher_model.', 'net.', 'encoder.', 'student_encoder.', 'backbone.'):
                if key.startswith(p):
                    return key[len(p):]
            return key
        
        # dict to hold new state dict
        new_state_dict = {}

        # process each key-value pair
        for key, value in state_dict.items():
            new_key = _strip_prefix(key)
            if new_key.startswith('swinViT.'):
                new_state_dict[new_key] = value
            elif 'swinViT.' in new_key:
                _, sub_key = new_key.split('swinViT.', 1)
                new_state_dict['swinViT.' + sub_key] = value

        # adjust patch embedding if necessary
        wkey = 'swinViT.patch_embed.proj.weight'
        if wkey in new_state_dict:
            w_ckpt = new_state_dict[wkey] # (out_channels, in_channels, D, H, W)
            w_target = self.model.swinViT.patch_embed.proj.weight
            if w_ckpt.shape != w_target.shape:
                print(f'[WARN] Deblur Module: incompatible shape for {wkey}: ckpt {w_ckpt.shape} vs target {w_target.shape}. Dropping...', flush=True)
                new_state_dict.pop(wkey)

        # bias sanity check
        bkey = 'swinViT.patch_embed.proj.bias'
        if bkey in new_state_dict:
            b_ckpt = new_state_dict[bkey]
            b_target = self.model.swinViT.patch_embed.proj.bias
            if b_ckpt.shape != b_target.shape:
                print(f'[WARN] Deblur Module: incompatible shape for {bkey}: ckpt {b_ckpt.shape} vs target {b_target.shape}. Dropping...', flush=True)
                new_state_dict.pop(bkey)

        # load state dict into model
        incompatible = self.model.load_state_dict(new_state_dict, strict=False)
        print(f'[INFO] Deblur Module: loaded {len(new_state_dict)} pretrained weights from {ckpt_path} with missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}', flush=True)

        # freeze encoder if specified
        if self.encoder_frozen:
            for name, param in self.model.named_parameters():
                if name.startswith('swinViT.'):
                    param.requires_grad = False
            print(f'[INFO] Deblur Module: encoder frozen for first {self.freeze_encoder_epochs} epochs', flush=True)

    # forward
    # x: blurred input volume, shape (B, 1, D, H, W) in [0, 1]
    # returns residual volume (of same shape) that is added to x to obtain the deblurred output
    def forward(self, x):
        residual = self.model(x)
        return residual
    
    # 3d gradient for edge loss
    @staticmethod
    def _grad3d(x):

        # d/dz
        dz_core = x[:, :, 1:, :, :] - x[:, :, :-1, :, :] # (B, C, D-1, H, W)
        dz = F.pad(dz_core, (0, 0, 0, 0, 0, 1)) # pad last dimension to match input shape

        # d/dy
        dy_core = x[:, :, :, 1:, :] - x[:, :, :, :-1, :] # (B, C, D, H-1, W)
        dy = F.pad(dy_core, (0, 0, 0, 1, 0, 0)) # pad last dimension to match input shape

        # d/dx
        dx_core = x[:, :, :, :, 1:] - x[:, :, :, :, :-1] # (B, C, D, H, W-1)
        dx = F.pad(dx_core, (0, 1, 0, 0, 0, 0)) # pad last dimension to match input shape

        return dz, dy, dx
    
    # 3d high pass filter (subtract local mean from image) to encourage model to match high freq structure (edges, fine details)
    @staticmethod
    def _highpass3d(x, kernel_size=3):
        B, C, D, H, W = x.shape

        # create averaging kernel
        weight = x.new_ones((C, 1, kernel_size, kernel_size, kernel_size)) / float(kernel_size ** 3) # (C_out=C, C_in=1, kD, kH, kW)

        # convolve and subtract
        smoothed = F.conv3d(x, weight, bias=None, stride=1, padding=kernel_size//2, groups=C)
        highfreq = x - smoothed
        return highfreq

    # on train epoch start
    def on_train_epoch_start(self):

        # unfreeze encoder after warmup epochs
        if self.encoder_frozen and self.current_epoch >= self.freeze_encoder_epochs:
            for name, param in self.model.named_parameters():
                param.requires_grad = True
            self.encoder_frozen = False
            print(f'[INFO] Deblur Module: encoder unfrozen at epoch {self.current_epoch}', flush=True)

    # training step
    def training_step(self, batch, batch_idx):

        # get data
        input_blurred = batch['input_vol'].to(self.device) # (B, 1, D, H, W)
        target_sharp = batch['target_vol'].to(self.device) # (B, 1, D, H, W)

        # forward
        residual = self.forward(input_blurred) # (B, 1, D, H, W), unconstrained logits
        # predict sharp output as blurred input + residual (clamped to [0, 1])
        output_deblurred_pred = torch.clamp(input_blurred + residual, 0.0, 1.0) # (B, 1, D, H, W)

        # l1 loss
        loss_l1 = self.l1_loss(output_deblurred_pred, target_sharp)

        # ssim loss (1 - ssim)
        loss_ssim = self.ssim_loss(output_deblurred_pred, target_sharp)

        # edge loss
        dz_pred, dy_pred, dx_pred = self._grad3d(output_deblurred_pred)
        dz_target, dy_target, dx_target = self._grad3d(target_sharp)
        loss_edge = (
            (dz_pred.abs() - dz_target.abs()).abs().mean() +
            (dy_pred.abs() - dy_target.abs()).abs().mean() +
            (dx_pred.abs() - dx_target.abs()).abs().mean()
        )

        # high frequency loss
        highfreq_pred = self._highpass3d(output_deblurred_pred)
        highfreq_target = self._highpass3d(target_sharp)
        loss_highfreq = self.l1_loss(highfreq_pred, highfreq_target)

        # total loss
        total_train_loss = (loss_l1 
                            + 0.2 * loss_ssim 
                            + self.edge_weight * loss_edge 
                            + self.highfreq_weight * loss_highfreq
                            )

        # psnr
        mse = torch.mean((output_deblurred_pred - target_sharp) ** 2) + 1e-8
        psnr = 10.0 * torch.log10(1.0 / mse)

        # log losses and metrics
        self.log('train_l1_loss', loss_l1, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_edge_loss', loss_edge, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_total_loss', total_train_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_highfreq_loss', loss_highfreq, on_step=False, on_epoch=True, prog_bar=False)
        self.log('train_ssim', 1.0-loss_ssim, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_psnr', psnr, on_step=True, on_epoch=True, prog_bar=True)

        return total_train_loss
    # val step
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        # get data
        input_blurred = batch['input_vol'].to(self.device) # (B, 1, D, H, W)
        target_sharp = batch['target_vol'].to(self.device) # (B, 1, D, H, W)

        # forward
        residual = self.forward(input_blurred) # (B, 1, D, H, W), unconstrained logits
        output_deblurred_pred = torch.clamp(input_blurred + residual, 0.0, 1.0) # (B, 1, D, H, W)

        # l1 loss
        loss_l1 = self.l1_loss(output_deblurred_pred, target_sharp)

        # ssim loss (1 - ssim)
        loss_ssim = self.ssim_loss(output_deblurred_pred, target_sharp)

        # edge loss
        dz_pred, dy_pred, dx_pred = self._grad3d(output_deblurred_pred)
        dz_target, dy_target, dx_target = self._grad3d(target_sharp)
        loss_edge = (
            (dz_pred.abs() - dz_target.abs()).abs().mean() +
            (dy_pred.abs() - dy_target.abs()).abs().mean() +
            (dx_pred.abs() - dx_target.abs()).abs().mean()
        )

        # high frequency loss
        highfreq_pred = self._highpass3d(output_deblurred_pred)
        highfreq_target = self._highpass3d(target_sharp)
        loss_highfreq = self.l1_loss(highfreq_pred, highfreq_target)

        # total loss
        total_val_loss = (loss_l1 
                          + 0.2 * loss_ssim 
                          + self.edge_weight * loss_edge 
                          + self.highfreq_weight * loss_highfreq
                          )

        # psnr
        mse = torch.mean((output_deblurred_pred - target_sharp) ** 2) + 1e-8
        psnr = 10.0 * torch.log10(1.0 / mse)

        # accumulate up to 5 val samples for wandb logging
        if batch_idx < 5 and isinstance(self.logger, WandbLogger) and hasattr(self.logger, 'experiment'):

            # create table if not exists
            if self.val_examples_table is None:
                self.val_examples_table = wandb.Table(columns=['Filename', 'Blurred (mid-z)', 'Deblurred (mid-z)', 'Sharp (mid-z)'])

            # filename (fallback to simple label if not provided)
            if isinstance(batch, dict) and 'filename' in batch:
                filename = batch['filename'][0]
            else:
                filename = f'val_sample_{batch_idx}'

            # get mid-z slices for visualization
            blurred_np = input_blurred[0, 0].detach().float().cpu().numpy()
            deblurred_np = output_deblurred_pred[0, 0].detach().float().cpu().numpy()
            sharp_np = target_sharp[0, 0].detach().float().cpu().numpy()

            mid_slice = blurred_np.shape[0] // 2
            blurred_slice = blurred_np[mid_slice, :, :]
            deblurred_slice = deblurred_np[mid_slice, :, :]
            sharp_slice = sharp_np[mid_slice, :, :]

            # add row to table
            self.val_examples_table.add_data(
                filename,
                wandb.Image(blurred_slice, caption='blurred'),
                wandb.Image(deblurred_slice, caption='deblurred'),
                wandb.Image(sharp_slice, caption='sharp')
            )

        # log losses and metrics
        self.log('val_l1_loss', loss_l1, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_edge_loss', loss_edge, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_total_loss', total_val_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_highfreq_loss', loss_highfreq, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_ssim', 1.0-loss_ssim, on_step=True, on_epoch=True, prog_bar=True)
        self.log('val_psnr', psnr, on_step=True, on_epoch=True, prog_bar=True)

        return {'total_val_loss': total_val_loss}

    # on val epoch end
    def on_validation_epoch_end(self):

        # log wandb table
        if self.val_examples_table is not None and isinstance(self.logger, WandbLogger) and hasattr(self.logger, 'experiment'):
            self.logger.experiment.log({f'val_examples_epoch_{self.current_epoch}': self.val_examples_table})

            # reset table for next epoch
            self.val_examples_table = None
    
    # configure optimizers
    def configure_optimizers(self):

        # separate encoder and decoder params to apply different lrs
        encoder_params, decoder_params = [], []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            (encoder_params if name.startswith('swinViT.') else decoder_params).append(param)

        # optimizers
        optimizer = torch.optim.AdamW(
            [
                {'params': encoder_params, 'lr': self.lr * self.encoder_lr_mult},
                {'params': decoder_params, 'lr': self.lr}
            ],
            weight_decay=float(self.hparams.weight_decay)
        )

        # cosine annealing lr scheduler
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(self.trainer.max_epochs),
            eta_min=self.lr * 0.1
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': lr_scheduler,
                'interval': 'epoch',
            }
        }

    





















