# Module for IBot pretraining

# --- Setup ---

# imports
from monai.networks.nets import SwinUNETR
import os
import random
import sys
from transformers import AutoTokenizer, AutoModel

import pytorch_lightning as pl

import torch
from torch import nn
import torch.nn.functional as F

# get functions from other files
sys.path.append('/home/ads4015/ssl_project/preprocess_patches/src')
from wu_visualization_functions import log_images_to_wandb_table, log_images_batches_to_wandb_table

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# --- Module ---

class IBOTCLIPPretrainModule(pl.LightningModule):

    # init
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config) # save all hyperparameters to self.hparams

        self.image_size = config['model']['image_size']
        self.mask_ratio = config['model']['mask_ratio'] # percentage of voxels to mask
        self.lr = config['model']['lr'] # learning rate
        self.mask_patch_size = config['model']['mask_patch_size']
        self.temp_student = config['model']['temp_student'] # temperature for student softmax
        self.temp_teacher = config['model']['temp_teacher'] # temperature for teacher softmax
        self.text_model_name = config['model']['text_model_name']
        self.embed_dim = config['model']['embed_dim']
        self.clip_temperature = config['model']['clip_temperature']
        self.reconstruction_head = nn.Sequential(
            nn.Conv3d(self.embed_dim, 1, kernel_size=1),
            nn.Sigmoid()
        )

        # *** SwinUNETR encoders ***

        # create swinUNETR as student network
        # constrain output using sigmoid 
        self.student_encoder = SwinUNETR(
            img_size=(self.image_size,)*3,
            in_channels=1,
            out_channels=self.embed_dim,
            feature_size=config['model']['feature_size'],
            use_checkpoint=True
        )

        # ema teacher network 
        # (same architecture as student, updated using running average of student model's parameters)
        self.teacher_encoder = SwinUNETR(
            img_size=(self.image_size,)*3,
            in_channels=1,
            out_channels=self.embed_dim,
            feature_size=config['model']['feature_size'], 
            use_checkpoint=False        
        )
        for p in self.teacher_encoder.parameters():
            p.requires_grad = False # freeze teacher weights

        self.register_buffer('ema_decay', torch.tensor(config['model']['ema_decay'])) # ema decay factor (0.996 is iBOT default)

        # *** text encoder ***
        
        # load pretrained tokenizer from HuggingFace (converts input text strings -> numerical token IDs)
        self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)

        # load transformer (process token IDs to produce contextual embeddings for the input text)
        self.text_encoder = AutoModel.from_pretrained(self.text_model_name)

        # project high dimensional vectors down to a fixed dimensional space (embed_dim) to compare/align with image features (CLIP)
        self.text_proj = nn.Linear(self.text_encoder.config.hidden_size, self.embed_dim)


        # image projection head for CLIP loss
        self.image_proj = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(self.embed_dim, self.embed_dim)
        )

        # lists and count for logging to wandb
        self.train_batches_for_logging = []
        self.val_batches_for_logging = []
        self.train_log_count = 0
        self.val_log_count = 0
        self.max_log_images = config['model']['max_log_images']


    # function to create patch-level binary mask for each volume in batch
    def generate_patch_mask(self, x, base_mask_patch_size):

        # create mask
        B, C, D, H, W = x.shape
        mask = torch.zeros((B, 1, D, H, W), dtype=torch.bool, device=x.device)

        # loop through batch
        for b in range(B):

            # randomly select patch size (base or double)
            mask_patch_size = random.choice([base_mask_patch_size, base_mask_patch_size*2])

            # determine how many nonoverlapping mask patches fit inside the volume
            d_mask_patches, h_mask_patches, w_mask_patches = D // mask_patch_size, H // mask_patch_size, W // mask_patch_size
            num_total_mask_patches = d_mask_patches * h_mask_patches * w_mask_patches # total number of patches that fit inside volume
            num_masked_mask_patches = int(self.mask_ratio * num_total_mask_patches) # number of patches that fit inside volume and should be masked

            # randomly sample patch indices to mask
            mask_patch_indices = torch.randperm(num_total_mask_patches, device=x.device)[:num_masked_mask_patches]

            # convert patch indices into volume coordinates and set mask
            for idx in mask_patch_indices:
                z = (idx // (h_mask_patches * w_mask_patches)) * mask_patch_size
                y = ((idx % (h_mask_patches * w_mask_patches)) // w_mask_patches) * mask_patch_size
                x_ = (idx % w_mask_patches) * mask_patch_size
                mask[b, :, z:z+mask_patch_size, y:y+mask_patch_size, x_:x_+mask_patch_size] = True

        return mask
    

    # *** encodings ***

    # encode text
    def encode_text(self, texts):

        # tokenize list of input texts into tensors for model
        tokens = self.text_tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)

        # pass tokenized input into text encoder, returning a dictionary of outputs
        out = self.text_encoder(**tokens)

        # get pooled ouptut or use average pooling across token dimension to create single vector for entire sequence
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state.mean(dim=1)

        # pass pooled vector through learned linear projection to map into shared image-text embedding space (for CLIP)
        return self.text_proj(pooled)
    
    # encode image
    def encode_image(self, x, mask, network):
        x_masked = x.clone()
        x_masked[mask] = 0
        return network(x_masked) # network will be either the student or the teacher encoder


    # *** losses ***

    # clip loss
    # image_embeds = tensor of image embeddings from SwinUNETR encoder
    # text_embeds = tensor of text embeddings from BERT text encoder
    def compute_clip_loss(self, image_embeds, text_embeds):

        # normalize embeddings to unit length (important for cosine similarity and contrastive learning)
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)

        # compute similarity matrix between all image-text pairs using matrix multiplication
        # sharpen/soften softmax output using clip_temperature
        logits = image_embeds @ text_embeds.T / self.clip_temperature

        # create ground truth labels for contrastive learning
        targets = torch.arange(logits.shape[0], device=self.device)

        # compute symmetric cross-entropy contrastive loss (bidirectional, image -> text AND text -> image)
        return (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2


    # forward pass through student encoder
    def forward(self, x):
        return self.student_encoder(x)
    

    # function to compute combined loss, MSE for masked and L1 for unmasked
    # combines distillation loss (student mimics teacher on masked regions) 
    # and reconstruction loss (student reconstructs original image in unmasked regions)
    def compute_ibot_loss(self, student_features, teacher_features, x, mask):

        # compute shape of student features
        B, C, D, H, W = student_features.shape # C = embed_dim

        # downsample mask and x to match output feature map and output for reconstruction
        mask_downsampled = F.interpolate(mask.float(), size=(D, H, W), mode='nearest')#.bool() # shape: (B, 1, D, H, W)
        x_downsampled = F.interpolate(x, size=(D, H, W), mode='trilinear', align_corners=False) # shape: (B, 1, D, H, W)

        # expand mask along channel dimension to mask embed_dim
        mask_expand = mask_downsampled.expand(B, C, D, H, W) # shape: (B, C, D, H, W)

        # flatten spatial dimensions to compute loss
        student_flat = student_features.reshape(B, -1) # shape: (B, C*D*H*W)
        teacher_flat = teacher_features.reshape(B, -1) # shape: (B, C*D*H*W)
        mask_flat = mask_expand.reshape(B, -1) # shape (B, C*D*H*W)

        # temperature scaling
        teacher_probs = F.softmax(teacher_flat / self.temp_teacher, dim=1).detach()
        student_logprobs = F.log_softmax(student_flat / self.temp_student, dim=1)

        mask_flat = mask_flat.to(dtype=torch.bool)

        # KL divergence loss on masked voxels only
        if mask_flat.any():
            mask_flat = mask_flat.bool()
            distill_loss = F.kl_div(
                student_logprobs[mask_flat],
                teacher_probs[mask_flat],
                reduction='batchmean'
            )
        else:
            distill_loss = torch.tensor(0.0, device=self.device)

        # # L1 reconstruction loss on unmasked voxels
        # x_downsampled = x_downsampled.expand(B, C, D, H, W) # ensure matches channel dimension of student features
        # # reconstruction_loss = F.l1_loss(student_features[~mask_downsampled], x_downsampled[~mask_downsampled])

        # # compute per-voxel L1 loss
        # l1_per_voxel = F.l1_loss(student_features.float(), x_downsampled.float(), reduction='none')

        # # only retain unmasked regions
        # mask_inv = (~mask_downsampled).float()
        # masked_l1 = l1_per_voxel * mask_inv
        # reconstruction_loss = masked_l1.sum() / mask_inv.sum()

        student_recon = self.reconstruction_head(student_features)
        mask_inv = 1.0 - mask_downsampled
        recon_loss = F.l1_loss(student_recon * mask_inv, x * mask_inv, reduction='sum') / mask_inv.sum()

        # return weighted sum of distillation loss and reconstruction loss
        return 0.8 * distill_loss + 0.2 * recon_loss

    
    # training step
    def training_step(self, batch, batch_idx):

        # get input image-text pair
        # x shape: (B, 1, D, H, W); texts is a list of strings, 1 per image
        x = batch['image']
        texts = batch['text']

        # generate patch level mask that randomly zeros out a subset of 3d patches in input image
        mask = self.generate_patch_mask(x, self.mask_patch_size)

        # student sees masked input
        student_out = self.encode_image(x, mask, self.student_encoder)

        # teacher sees full input (no gradients)
        with torch.no_grad():
            teacher_out = self.encode_image(x, mask*0, self.teacher_encoder)

        # embed image and text using average pooling + linear projection for image and BERT + projection for text
        image_embed = self.image_proj(student_out)
        text_embed = self.encode_text(texts)

        # compute contrastive loss between image and text embeddings - corresponding aligned pairs should be close in embedding space
        clip_loss = self.compute_clip_loss(image_embed, text_embed)

        # compute iBOT loss (KL divergence between student and teacher on masked regions and L1 reconstruction on unmasked regions)
        ibot_loss = self.compute_ibot_loss(student_out, teacher_out, x, mask)

        # combine both losses into single scaler to minimize
        total_loss = ibot_loss + clip_loss

        # log losses
        self.log('train_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('clip_loss', clip_loss, on_step=True, on_epoch=True)
        self.log('ibot_loss', ibot_loss, on_step=True, on_epoch=True)

        if self.train_log_count < self.max_log_images:
            num_to_add = min(self.max_log_images - self.train_log_count, x.shape[0])

        # cache for wandb image logging
        self.last_train_batch = batch
        self.last_train_masked = x.clone()
        self.last_train_masked[mask] = 0

        if self.train_log_count < self.max_log_images:

            recon_image = self.reconstruction_head(student_out)
            recon_image = (recon_image - recon_image.min()) / (recon_image.max() - recon_image.min() + 1e-8)

            masked_input = x.clone()
            masked_input[mask] = 0


            self.train_batches_for_logging.append((x[:num_to_add].detach().cpu(), masked_input[:num_to_add].detach().cpu(), recon_image[:num_to_add].detach().cpu()))
            self.train_log_count += num_to_add

        # return total loss
        return total_loss
    

    # validation step (same as training but without backprop)
    def validation_step(self, batch, batch_idx):

        # get input image-text pair
        # x shape: (B, 1, D, H, W); texts is a list of strings, 1 per image
        x = batch['image']
        texts = batch['text']

        # generate patch level mask that randomly zeros out a subset of 3d patches in input image
        mask = self.generate_patch_mask(x, self.mask_patch_size)

        # student sees masked input
        student_out = self.encode_image(x, mask, self.student_encoder)

        # teacher sees full input (no gradients)
        with torch.no_grad():
            teacher_out = self.encode_image(x, mask*0, self.teacher_encoder)

        # embed image and text using average pooling + linear projection for image and BERT + projection for text
        image_embed = self.image_proj(student_out)
        text_embed = self.encode_text(texts)

        # compute contrastive loss between image and text embeddings - corresponding aligned pairs should be close in embedding space
        clip_loss = self.compute_clip_loss(image_embed, text_embed)

        # compute iBOT loss (KL divergence between student and teacher on masked regions and L1 reconstruction on unmasked regions)
        ibot_loss = self.compute_ibot_loss(student_out, teacher_out, x, mask)

        # combine both losses into single scaler to minimize
        total_loss = ibot_loss + clip_loss

        # log losses
        self.log('val_loss', total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('clip_loss', clip_loss, on_step=True, on_epoch=True)
        self.log('ibot_loss', ibot_loss, on_step=True, on_epoch=True)

        if self.val_log_count < self.max_log_images:

            num_to_add = min(self.max_log_images - self.val_log_count, x.shape[0])

            recon_image = self.reconstruction_head(student_out)
            recon_image = (recon_image - recon_image.min()) / (recon_image.max() - recon_image.min() + 1e-8)

            masked_input = x.clone()
            masked_input[mask] = 0

            self.val_batches_for_logging.append((x[:num_to_add].detach().cpu(), masked_input[:num_to_add].detach().cpu(), recon_image[:num_to_add].detach().cpu()))
            self.val_log_count += num_to_add

        # return total loss
        return total_loss


    # after train epoch
    def on_train_epoch_end(self):

        # log images to wandb
        if self.train_batches_for_logging:
            log_images_batches_to_wandb_table(
                logger=self.logger, 
                batches=self.train_batches_for_logging, 
                prefix='Train', 
                step_or_epoch=self.current_epoch, 
                max_rows=self.max_log_images,
                log_histograms=False
            )
            self.train_batches_for_logging.clear()
            self.train_log_count = 0
            
    # after val epoch
    def on_validation_epoch_end(self):
        if self.val_batches_for_logging:
            log_images_batches_to_wandb_table(
                logger=self.logger,
                batches=self.val_batches_for_logging,
                prefix='Val',
                step_or_epoch=self.current_epoch,
                max_rows=self.max_log_images,
                log_histograms=False
            )
            self.val_batches_for_logging.clear()
            self.val_log_count = 0


    # after backwards
    def on_after_backward(self):

        # update ema teacher parameters from student
        for student_param, teacher_param in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
            teacher_param.data = self.ema_decay * teacher_param.data + (1 - self.ema_decay) * student_param.data


    # optimizer
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)