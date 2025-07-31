# Module for IBot pretraining

# --- Setup ---

# imports
from monai.networks.nets import SwinUNETR
import random
import sys

import pytorch_lightning as pl

import torch
from torch import nn
import torch.nn.functional as F


# get functions from other files
sys.path.append('/home/ads4015/ssl_project/preprocess_patches/src')
from wu_visualization_functions import log_images_to_wandb_table


# --- Module ---

class IBOTPretrainModule(pl.LightningModule):

    # init
    def __init__(self, image_size=96, mask_ratio=0.6, lr=1e-4, ema_decay=0.996, mask_patch_size=16, temp_student=0.1, temp_teacher=0.04): ## ADD TO CONFIG - mask ratio, lr, ema_decay, patch_size, temp_student, temp_teacher
        super().__init__()
        self.save_hyperparameters() # save all hyperparameters to self.hparams
        self.mask_ratio = mask_ratio # percentage of voxels to mask
        self.lr = lr # learning rate
        self.mask_patch_size = mask_patch_size # base patch size
        self.temp_student = temp_student # temperature for student softmax
        self.temp_teacher = temp_teacher # temperature for teacher softmax

        # create swinUNETR as student network
        # constrain output using sigmoid 
        self.encoder = nn.Sequential(
            SwinUNETR(
                img_size=(image_size, image_size, image_size),
                in_channels=1,
                out_channels=1, # dummy output for reconstruction
                feature_size=48, ## ADD TO CONFIG - feature_size
                use_checkpoint=True
            ),
            # nn.Sigmoid() # constrain output to [0, 1]
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
            # nn.Sigmoid()      
        )
        for p in self.teacher.parameters():
            p.requires_grad = False # freeze teacher weights

        self.register_buffer('ema_decay', torch.tensor(ema_decay)) # ema decay factor (0.996 is iBOT default)

    # warm start teacher
    def on_fit_start(self):
        for s, t in zip(self.encoder.parameters(), self.teacher.parameters()):
            t.data.copy_(s.data)

    # function to crate patch-level binary mask for each volume in batch
    def generate_patch_mask(self, x, base_mask_patch_size):

        # create mask
        B, C, D, H, W = x.shape
        device = x.device
        mask = torch.zeros((B, 1, D, H, W), dtype=torch.bool, device=device)

        # loop through batch
        for b in range(B):

            # randomly select patch size (base or double)
            mask_patch_size = random.choice([base_mask_patch_size, base_mask_patch_size*2])

            # determine how many nonoverlapping mask patches fit inside the volume
            d_mask_patches, h_mask_patches, w_mask_patches = D // mask_patch_size, H // mask_patch_size, W // mask_patch_size
            num_total_mask_patches = d_mask_patches * h_mask_patches * w_mask_patches # total number of patches that fit inside volume
            num_masked_mask_patches = int(self.mask_ratio * num_total_mask_patches) # number of patches that fit inside volume and should be masked

            # randomly sample patch indices to mask
            mask_patch_indices = torch.randperm(num_total_mask_patches, device=device)[:num_masked_mask_patches]

            # convert patch indices into volume coordinates and set mask
            for idx in mask_patch_indices:
                z = (idx // (h_mask_patches * w_mask_patches)) * mask_patch_size
                y = ((idx % (h_mask_patches * w_mask_patches)) // w_mask_patches) * mask_patch_size
                x_ = (idx % w_mask_patches) * mask_patch_size
                mask[b, :, z:z+mask_patch_size, y:y+mask_patch_size, x_:x_+mask_patch_size] = True

        return mask
        

        # MSE loss (L2 loss): sensitive to large error, strongly penalizes outliers, 
        # encourages smoother reconstruction bec squaring diff emphasizes large deviations,
        # can lead to blurry outputs if used alone
        # MAE loss (L1 loss): robust to outliers,
        # helps preserve edges and fine contrast better than MSE alone
        # can introduce sparsity in output (good for patchy structures like vessels/plaques)
        # self.loss_fn = lambda x, y: 0.8 * nn.MSELoss()(x, y) + 0.2 * nn.L1Loss()(x, y)

    # forward pass through student encoder
    def forward(self, x):
        return self.encoder(x)
    
    # function to compute combined loss, MSE for masked and L1 for unmasked
    def compute_loss(self, student_features, teacher_features, x, mask):

        # downsample mask to match feature resolution
        mask_ds = F.interpolate(mask.float(), size=student_features.shape[2:], mode='nearest').bool()

        # variable to hold masked loss (initialized to 0 for first 5 epochs)
        loss_masked = torch.tensor(0.0, device=x.device)

        # use kl divergence of student with teacher only after 5th epoch (otherwise only use L1 loss on unmasked voxels)
        if self.current_epoch > 5 and mask_ds.any():
            with torch.no_grad():
                teacher_soft = F.softmax(teacher_features / self.temp_teacher, dim=1)
            student_log = F.log_softmax(student_features / self.temp_student, dim=1)
            loss_masked = F.kl_div(
                student_log[mask_ds],
                teacher_soft[mask_ds],
                reduction='batchmean'
            )

        # L1 reconstruction on unmasked voxels 
        loss_unmasked = F.l1_loss(student_features[~mask_ds], x[~mask_ds])

        # return weighted sum of masked and unmasked loss
        return 0.8 * loss_masked + 0.2 * loss_unmasked




        # # flatten spatial dimensions to compute loss
        # student_flat = student_features.view(student_features.size(0), -1)
        # teacher_flat = teacher_features.view(teacher_features.size(0), -1)

        # # temperature scaling
        # temp_s, temp_t = self.temp_student, self.temp_teacher
        # teacher_probs = F.softmax(teacher_flat / temp_t, dim=1).detach()
        # student_logprobs = F.log_softmax(student_flat / temp_s, dim=1)

        # # get masked indices for computing distillation loss
        # mask_flat = mask.view(mask.size(0), -1)
        # masked_idx = mask_flat.bool()

        # # KL divergence loss on masked voxels only
        # if masked_idx.any():
        #     distill_loss = F.kl_div(
        #         student_logprobs[masked_idx],
        #         teacher_probs[masked_idx],
        #         reduction='batchmean'
        #     )
        # else:
        #     distill_loss = torch.tensor(0.0, device=self.device)

        # # L1 reconstruction loss on unmasked voxels
        # reconstruction_loss = F.l1_loss(student_features[~mask], x[~mask])

        # # return weighted sum of distillation loss and reconstruction loss
        # return 0.8 * distill_loss + 0.2 * reconstruction_loss

        # # distill_loss = nn.MSELoss()(student_features[mask], teacher_features[mask]) # match masked region to teacher
        # # recon_loss = nn.L1Loss()(student_features[~mask], x[~mask]) # match unmasked region to original
        # # return 0.8 * distill_loss + 0.2 * recon_loss
    
    # training step
    def training_step(self, batch, batch_idx):

        # input volume, shape (B, 1, D, H, W)
        x = batch['image']
        # print(f'[DEBUG] Training batch shape: {x.shape}')

        # generate patch level mask and apply to input
        mask = self.generate_patch_mask(x, self.mask_patch_size)
        x_masked = x.clone()
        x_masked[mask] = 0

        # student sees masked input
        student_features = self.encoder(x_masked)

        # teacher sees full input (no gradients)
        with torch.no_grad():
            teacher_features = self.teacher(x)

        # combined loss
        train_loss = self.compute_loss(student_features, teacher_features, x, mask)
        self.log('train_loss', train_loss, prog_bar=True, on_step=True, on_epoch=True)

        # cache last batch info for logging to wandb
        self.last_train_batch = batch
        self.last_train_mask = mask
        self.last_train_masked = x_masked
        self.last_train_output = student_features

        # debug output
        if self.current_epoch == 0 and batch_idx == 0:
            print(f'[DEBUG] Student output stats: min={student_features.min().item()}, max={student_features.max().item()}')
            print(f'[DEBUG] Mask coverage: {mask.float().mean().item() * 100:.2f}%')

        return train_loss
    
    # validation step (same as training but without backprop)
    def validation_step(self, batch, batch_idx):

        # input volume, shape (B, 1, D, H, W)
        x = batch['image']
        # print(f'[DEBUG] Validation batch shape: {x.shape}')

        # generate mask and apply it to create masked input for student
        mask = self.generate_patch_mask(x, self.mask_patch_size)
        x_masked = x.clone()
        x_masked[mask] = 0

        # get output from student and teacher
        student_features = self.encoder(x_masked)
        with torch.no_grad():
            teacher_features = self.teacher(x)

        # compute validation loss
        val_loss = self.compute_loss(student_features, teacher_features, x, mask)
        self.log('val_loss', val_loss, prog_bar=True, on_step=False, on_epoch=True)

        # cache last batch info for wandb logging
        self.last_val_batch = batch
        self.last_val_mask = mask
        self.last_val_masked = x_masked
        self.last_val_output = student_features

        # visualize 1 batch at the start of each val epoch
        if batch_idx == 0 and x.shape[0] > 0:

            # log images to wandb table
            log_images_to_wandb_table(self.logger, x, x_masked, student_features, 'Val', self.current_epoch, log_histograms=(self.current_epoch == 0))

        return val_loss
    
    # after train epoch
    def on_train_epoch_end(self):

        # log images to wandb
        if hasattr(self, 'last_train_batch'):
            log_images_to_wandb_table(logger=self.logger, 
                                      originals=self.last_train_batch['image'], 
                                      maskeds=self.last_train_masked, 
                                      student_preds=self.last_train_output, 
                                      prefix='Train', 
                                      step_or_epoch=self.current_epoch, 
                                      log_histograms=(self.current_epoch == 0))
            
    # after val epoch
    def on_validation_epoch_end(self):
        if hasattr(self, 'last_val_batch'):
            log_images_to_wandb_table(logger=self.logger,
                                      originals=self.last_val_batch['image'],
                                      maskeds=self.last_val_masked,
                                      student_preds=self.last_val_output,
                                      prefix='Val',
                                      step_or_epoch=self.current_epoch,
                                      log_histograms=(self.current_epoch == 0))

    # after backwards
    def on_after_backward(self):

        # update ema teacher parameters from student
        for student_param, teacher_param in zip(self.encoder.parameters(), self.teacher.parameters()):
            teacher_param.data = self.ema_decay * teacher_param.data + (1 - self.ema_decay) * student_param.data

    # optimizer
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr) ## ADD TO CONFIG - torch.optim