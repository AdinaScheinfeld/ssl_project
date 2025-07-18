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
import torch.nn.functional as F
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
            lambda x: tiff.imread(x),
            lambda x: torch.tensor(x, dtype=torch.float32) # convert to float32 tensor
        ])

    # get number of files in dataset
    def __len__(self):
        return len(self.file_paths)
    
    # load and preprocess a .tif image
    def __getitem__(self, idx):
        img = self.transforms(self.file_paths[idx])

        # print min/max intensity in image for debugging
        # print(f'[DEBUG] Image {idx} shape: {img.shape}, min: {img.min().item():.2f}, max: {img.max().item():.2f}')

        return {'image': img} # return dict
    

# --- Helper Functions ---

# function to normalize images for visualization
def normalize_for_visualization(img, low=1, high=99):

    # appply percentile-based contrast stretching to single 2d slice
    p_low, p_high = np.percentile(img, (low, high))
    if p_high - p_low < 1e-5:
        return np.zeros_like(img) # avoid division by 0
    img_norm = np.clip((img - p_low) / (p_high - p_low), 0, 1)
    return img_norm

# function to log image intensity histogram
def log_intensity_histogram(img_tensor, prefix, logger, step_or_epoch):

    # flatten image
    img_flat = img_tensor.detach().cpu().numpy().flatten()
    logger.experiment.log({
        f'{prefix} Intensity histogram (epoch {step_or_epoch})': wandb.Histogram(img_flat)
    })

# function to log images to wandb table
def log_images_to_wandb_table(logger, originals, maskeds, student_preds, prefix, step_or_epoch, max_rows=4, log_histograms=True):

    # # create table
    # table = wandb.Table(columns=['Original RAW', 'Masked RAW', 'Student RAW', 
    #                              'Original NORM', 'Masked NORM', 'Student NORM'])
    # create table
    table = wandb.Table(columns=['Original', 'Masked', 'Student'])

    # loop through number of entries to plot and get images for plotting
    for i in range(min(max_rows, originals.shape[0])):
        center_z = originals.shape[2] // 2
        original_img = originals[i, 0].cpu().numpy()[center_z]
        masked_img = maskeds[i, 0].cpu().numpy()[center_z]
        pred_img = student_preds[i, 0].detach().cpu().numpy()[center_z]

        # log intensity histogram for first sample in each batch
        if i == 0 and log_histograms:
            log_intensity_histogram(originals[i, 0], f'{prefix} Original', logger, step_or_epoch)
            log_intensity_histogram(maskeds[i, 0], f'{prefix} Masked', logger, step_or_epoch)
            log_intensity_histogram(student_preds[i, 0], f'{prefix} Student', logger, step_or_epoch)

        # create table
        table.add_data(
            wandb.Image(original_img, caption=f'Original {i}'),
            wandb.Image(masked_img, caption=f'Masked {i}'),
            wandb.Image(pred_img, caption=f'Predicted {i}')
            # wandb.Image(original_img, caption=f'Original RAW {i}'),
            # wandb.Image(masked_img, caption=f'Masked RAW {i}'),
            # wandb.Image(pred_img, caption=f'Predicted RAW {i}'),
            # wandb.Image(normalize_for_visualization(original_img), caption=f'Original NORM {i}'),
            # wandb.Image(normalize_for_visualization(masked_img), caption=f'Masked NORM {i}'),
            # wandb.Image(normalize_for_visualization(pred_img), caption=f'Predicted NORM {i}'),
        )
    
    # log table
    logger.experiment.log({f'{prefix} Samples (epoch {step_or_epoch})': table})


# --- Model ---

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

        # flatten spatial dimensions to compute loss
        student_flat = student_features.view(student_features.size(0), -1)
        teacher_flat = teacher_features.view(teacher_features.size(0), -1)

        # temperature scaling
        temp_s, temp_t = self.temp_student, self.temp_teacher
        teacher_probs = F.softmax(teacher_flat / temp_t, dim=1).detach()
        student_logprobs = F.log_softmax(student_flat / temp_s, dim=1)

        # get masked indices for computing distillation loss
        mask_flat = mask.view(mask.size(0), -1)
        masked_idx = mask_flat.bool()

        # KL divergence loss on masked voxels only
        if masked_idx.any():
            distill_loss = F.kl_div(
                student_logprobs[masked_idx],
                teacher_probs[masked_idx],
                reduction='batchmean'
            )
        else:
            distill_loss = torch.tensor(0.0, device=self.device)

        # L1 reconstruction loss on unmasked voxels
        reconstruction_loss = F.l1_loss(student_features[~mask], x[~mask])

        # return weighted sum of distillation loss and reconstruction loss
        return 0.8 * distill_loss + 0.2 * reconstruction_loss

        # distill_loss = nn.MSELoss()(student_features[mask], teacher_features[mask]) # match masked region to teacher
        # recon_loss = nn.L1Loss()(student_features[~mask], x[~mask]) # match unmasked region to original
        # return 0.8 * distill_loss + 0.2 * recon_loss
    
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
    

# --- Data Loading ---

def get_data_loaders(train_dir, val_dir, batch_size, train_samp_size, val_samp_size, seed): ## ADD TO CONFIG - batch_size (depends on gpu memory and model complexity-feature_size)

    # set random seed for reproducibility
    random.seed(seed)

    # get lists of files
    train_files = sorted(glob.glob(os.path.join(train_dir, '*.tiff')))
    val_files = sorted(glob.glob(os.path.join(val_dir, '*.tiff')))

    # print how many files were found
    print(f'[DEBUG] Found {len(train_files)} train files and {len(val_files)} val files.')

    # check if there are enough errors, or raise an informative error
    if train_samp_size > len(train_files):
        raise ValueError(f'Requested {train_samp_size} training samples, but only found {len(train_files)} in {train_dir}')
    if val_samp_size > len(val_files):
        raise ValueError(f'Requested {val_samp_size} validation samples, but only found {len(val_files)} in {val_dir}')

    # subsample a fixed number of patches
    train_files = random.sample(train_files, train_samp_size)
    val_files = random.sample(val_files, val_samp_size)
    print(f'[DEBUG] Using {len(train_files)} train samples and {len(val_files)} val samples.')

    # create datasets
    train_ds = TiffPatchDataset(train_files)
    val_ds = TiffPatchDataset(val_files)

    # print dataset sizes and steps per epoch for debugging
    # print(f'[DEBUG] Num train samples: {len(train_ds)}')
    # print(f'[DEBUG] Num val samples: {len(val_ds)}')
    print(f'[DEBUG] Train steps per epoch: {len(train_ds) // batch_size}')

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
        ema_decay=config['model']['ema_decay'],
        mask_patch_size=config['model']['mask_patch_size'],
        temp_student=config['model']['temp_student'],
        temp_teacher=config['model']['temp_teacher']
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

        
    














