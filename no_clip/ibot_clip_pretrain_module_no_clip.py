# ibot_clip_pretrain_module_no_clip.py - Module for IBot pretraining

# --- Setup ---

# imports
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os
import random
import seaborn as sns
from sklearn.preprocessing import normalize as sklearn_normalize
import sys
from transformers import AutoTokenizer, AutoModel
import umap
import wandb

from monai.networks.nets import SwinUNETR

import pytorch_lightning as pl

import torch
from torch import nn
import torch.nn.functional as F

# get functions from other files
sys.path.append('/home/ads4015/ssl_project/preprocess_patches/src')
from wu_visualization_functions import log_images_to_wandb_table, log_images_batches_to_wandb_table, log_embedding_umap_plot

os.environ['TOKENIZERS_PARALLELISM'] = 'false'


# --- Lightweight decoder ---

class LightDecoder(nn.Module):

    # init
    def __init__(self, embed_dim):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv3d(embed_dim, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(32, 1, kernel_size=1)
        )

    # forward
    def forward(self, x):
        return self.decoder(x)


# --- Module ---

class IBOTCLIPPretrainModuleNoClip(pl.LightningModule):

    # init
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config) # save all hyperparameters to self.hparams
        self.use_text = config['model'].get('use_text', True) # whether to use text encoder and text-image losses
        self.base_mask_ratio = config['model']['mask_ratio'] # base mask ratio to use after warmup

        # parse downsampling config
        data_cfg = config['data']
        down_cfg = data_cfg.get('downsample', {})
        if data_cfg.get('use_sub_patches', False): # use crops
            self.image_size = int(data_cfg.get('sub_patch_size', 64))
        elif bool(down_cfg.get('enabled', False)): # downsample
            self.image_size = int(down_cfg.get('target_size', data_cfg.get('base_patch_size', 96)))
        else: # use images without cropping/downsampling
            self.image_size = int(data_cfg.get('base_patch_size', 96))

        print(f'[INFO] Using {self.image_size}^3 size patches.', flush=True)

        self.mask_ratio = config['model']['mask_ratio'] # percentage of voxels to mask
        self.mask_ratio_warmup = config['model']['mask_ratio_warmup']
        self.warmup_epochs = config['model']['warmup_epochs']
        self.lr = config['model']['lr'] # learning rate
        self.mask_patch_size = config['model']['mask_patch_size']
        self.temp_student = config['model']['temp_student'] # temperature for student softmax
        self.temp_teacher = config['model']['temp_teacher'] # temperature for teacher softmax
        self.embed_dim = config['model']['embed_dim']
        self.reconstruction_head = LightDecoder(self.embed_dim)
        self.finetune_text = bool(config['model'].get('finetune_text', True)) if self.use_text else False
        self.text_top_k_layers = int(config['model'].get('text_top_k_layers', 4)) if self.use_text else 0
        self.text_finetune_start_epoch = int(config['model'].get('text_finetune_start_epoch', self.warmup_epochs)) if self.use_text else 10**9

        # if using text encoder and text-image losses
        if self.use_text:
            self.text_model_name = config['model']['text_model_name']
            self.clip_temperature = config['model']['clip_temperature']
            init_temp = float(config['model'].get('clip_temperature', 0.07))
            self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / init_temp), dtype=torch.float)) # learnable logit scale for clip loss (instead of fixed temperature)

        # indicate what size image patches are being used
        print(f'[INFO] Using {self.image_size}^3 size patches.', flush=True)

        # track best metrics across the run
        self.best_metrics = {
            'train_loss': float('inf'),
            'val_loss': float('inf'),
            'train_clip_loss': float('inf'),
            'val_clip_loss': float('inf'),
            'train_ibot_loss': float('inf'),
            'val_ibot_loss': float('inf')
        }

        # *** SwinUNETR encoders ***

        # create swinUNETR as student network
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

        # initialize teacher with student weights
        self.teacher_encoder.load_state_dict(self.student_encoder.state_dict())

        for p in self.teacher_encoder.parameters():
            p.requires_grad = False # freeze teacher weights
        self.teacher_encoder.eval() # set teacher to eval mode

        self.register_buffer('ema_decay', torch.tensor(config['model']['ema_decay'])) # ema decay factor

        # *** text encoder ***
        
        # if using text encoder and text-image losses
        if self.use_text:

            # load pretrained tokenizer from HuggingFace (converts input text strings -> numerical token IDs)
            self.text_tokenizer = AutoTokenizer.from_pretrained(self.text_model_name)

            # load transformer (process token IDs to produce contextual embeddings for the input text)
            self.text_encoder = AutoModel.from_pretrained(self.text_model_name)

            # keep checkpointing while frozen (saves memory), but will disable when unfreezing
            self.text_encoder.gradient_checkpointing_enable()

            # projection head (always trainable)
            # project high dimensional vectors down to a fixed dimensional space (embed_dim) to compare/align with image features (CLIP)
            # 2 layer MLP with ReLU and layer norm for stability
            self.text_proj = nn.Sequential(
                nn.Linear(self.text_encoder.config.hidden_size, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim))
            
            # start fully frozen
            self._freeze_all_text()
            self.text_encoder.eval()

            # image projection head for CLIP loss
            # 2 layer MLP with ReLU and layer norm for stability
            self.image_proj = nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Flatten(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.ReLU(),
                nn.Linear(self.embed_dim, self.embed_dim),
                nn.LayerNorm(self.embed_dim)
            )

        # else, set text-related modules to None
        else:
            self.text_tokenizer = None
            self.text_encoder = None
            self.text_proj = None
            self.image_proj = None

        # lists and count for logging to wandb
        self.train_batches_for_logging = []
        self.val_batches_for_logging = []
        self.train_log_count = 0
        self.val_log_count = 0
        self.max_log_images = config['model']['max_log_images']

        # loss weights
        self.distill_weight = config['loss_weights']['distill_weight']
        self.reconstruction_weight = config['loss_weights']['reconstruction_weight']
        self.align_weight = float(config['loss_weights'].get('align_weight', 0)) if self.use_text else 0.0
        self.clip_weight = float(config['loss_weights'].get('clip_weight', 0)) if self.use_text else 0.0


    # function to freeze all text encoder layers
    def _freeze_all_text(self):

        # freeze all text encoder parameters
        if not self.use_text or self.text_encoder is None:
            return
        
        # freeze all text encoder parameters
        for p in self.text_encoder.parameters():
            p.requires_grad = False

    # function to unfreeze top k layers of text encoder
    def _unfreeze_top_k_text_layers(self, k):

        # if not using text encoder, return
        if not self.use_text or self.text_encoder is None:
            return

        # freeze everything first, for safety
        self._freeze_all_text()

        num_layers = self.text_encoder.config.num_hidden_layers
        start = max(0, num_layers - int(k)) # starting layer index to unfreeze

        # unfreeze top k transformer layers
        for name, p in self.text_encoder.named_parameters():
            if any(name.startswith(f'encoder.layer.{i}.') for i in range(start, num_layers)):
                p.requires_grad = True

            # unfreeze embeddings and layer norms too
            if name.startswith(('embeddings.',)) or '.LayerNorm.' in name:
                p.requires_grad = True

        if hasattr(self.text_encoder, 'pooler'):
            for p in self.text_encoder.pooler.parameters():
                p.requires_grad = True


    # function to create patch-level binary mask for each volume in batch
    def generate_patch_mask(self, x, base_mask_patch_size):

        # create mask
        # print(f'[DEBUG] x.shape in generate_patch_mask: {x.shape}', flush=True)
        B, C, D, H, W = x.shape
        mask = torch.zeros((B, 1, D, H, W), dtype=torch.bool, device=x.device)

        # loop through batch
        for b in range(B):

            # randomly select patch size (base or double)
            if torch.rand((), device=x.device) < 0.5:
                mask_patch_size = base_mask_patch_size
            else:
                mask_patch_size = base_mask_patch_size * 2

            # determine how many nonoverlapping mask patches fit inside the volume
            d_mask_patches, h_mask_patches, w_mask_patches = D // mask_patch_size, H // mask_patch_size, W // mask_patch_size
            num_total_mask_patches = d_mask_patches * h_mask_patches * w_mask_patches # total number of patches that fit inside volume
            num_masked_mask_patches = max(1, int(self.mask_ratio * num_total_mask_patches)) # number of patches that fit inside volume and should be masked

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

        # if not using text encoder, return None
        if not self.use_text:
            raise RuntimeError('encode_text called but use_text is False.')

        # tokenize list of input texts into tensors for model
        tokens = self.text_tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)

        # only build grads for text when it's trainable
        with torch.set_grad_enabled(self.text_encoder.training):

            # pass tokenized input into text encoder, returning a dictionary of outputs
            out = self.text_encoder(**tokens)

        # get pooled ouptut or use average pooling across token dimension to create single vector for entire sequence
        pooled = out.last_hidden_state.mean(dim=1)

        # pass pooled vector through learned linear projection to map into shared image-text embedding space (for CLIP)
        proj = self.text_proj(pooled)

        # normalize before return
        return F.normalize(proj, dim=-1)
    
    # encode image
    def encode_image(self, x, mask, network):
        x_masked = x.clone()
        x_masked[mask] = 0
        features = network(x_masked) # network will be either the student or the teacher encoder

        # normalize before return
        features_norm = F.normalize(features, dim=1)

        return features, features_norm


    # *** losses ***

    # clip loss
    # image_embeds = tensor of image embeddings from SwinUNETR encoder
    # text_embeds = tensor of text embeddings from BERT text encoder
    def compute_clip_loss(self, image_embeds, text_embeds):

        # if not using text encoder, return zero loss
        if not self.use_text:
            return torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)

         # logit scale is exp of the parameter, clamped to avoid extreme values
        logit_scale = self.logit_scale.exp().clamp(1e-3, 100.0)

        # compute similarity matrix between all image-text pairs using matrix multiplication
        # sharpen/soften softmax output using clip_temperature
        logits = (image_embeds @ text_embeds.T) * logit_scale

        # create ground truth labels for contrastive learning
        targets = torch.arange(logits.shape[0], device=self.device)

        # compute symmetric cross-entropy contrastive loss (bidirectional, image -> text AND text -> image)
        loss = (F.cross_entropy(logits, targets) + F.cross_entropy(logits.T, targets)) / 2

        return loss, logit_scale


    # forward pass through student encoder
    def forward(self, x):
        return self.student_encoder(x)
    

    # function to compute combined loss:
    # - distillation loss (student mimics teacher on masked regions) 
    # - L1 reconstruction loss (student reconstructs masked regions)
    def compute_ibot_loss(self, student_out_norm, teacher_out_norm, x, mask, student_feats_raw, student_image_embed=None, text_embed=None):

        # compute shape of student features
        B, C, D, H, W = student_out_norm.shape # C = embed_dim

        # downsample mask and x to match output feature map and output for reconstruction
        mask_downsampled = F.interpolate(mask.float(), size=(D, H, W), mode='nearest') # shape: (B, 1, D, H, W)
        x_downsampled = F.interpolate(x, size=(D, H, W), mode='trilinear', align_corners=False) # shape: (B, 1, D, H, W)

        # reshape to (B, C, N) with N = D*H*W and build a boolean mask over N
        N = D * H * W
        stu_tok = student_out_norm.view(B, C, N)
        tea_tok = teacher_out_norm.view(B, C, N).detach()
        mask_bool = mask_downsampled.view(B, 1, N).to(torch.bool).squeeze(1) # shape: (B, N), boolean

        # select only masked tokens along the token (N) dimension
        # shapes after selection: (?, C) where ? = total number of masked tokens in batch
        if mask_bool.any():
            student_selection = stu_tok.transpose(1, 2)[mask_bool].float() # shape: (?, C)
            teacher_selection = tea_tok.transpose(1, 2)[mask_bool].float() # shape: (?, C)

            ## UP TO HERE - add teacher centering and momentum update for stability later

            # temperature scaled per token distribution over channels
            student_logprobs = F.log_softmax(student_selection / self.temp_student, dim=-1)
            teacher_probs = F.softmax(teacher_selection / self.temp_teacher, dim=-1)

            # KL divergence loss on masked tokens only
            distill_loss = F.kl_div(student_logprobs, teacher_probs, reduction='batchmean')

        else:
            distill_loss = torch.tensor(0.0, device=self.device)

        student_recon = self.reconstruction_head(student_feats_raw) # for reconstruction, decode from the raw (unnormalized) features

        # pixel reconstruction loss (L1) on masked regions only
        recon_loss_l1 = F.l1_loss(student_recon * mask_downsampled, 
                                  x_downsampled * mask_downsampled, 
                                  reduction='sum') / (mask_downsampled.sum() + 1e-8)
        
        # alignment loss
        alignment_loss = self.compute_alignment_loss(student_image_embed, text_embed) if self.use_text else torch.tensor(0.0, device=self.device)

        # compute weighted sum of distillation loss and reconstruction loss and alignment loss
        total_ibot_loss = self.distill_weight * distill_loss + self.reconstruction_weight * recon_loss_l1 + self.align_weight * alignment_loss
        
        # composite reconstruction for logging (unmasked regions from input, masked regions from student reconstruction)
        recon_composite = x_downsampled * (1.0 - mask_downsampled) + student_recon * mask_downsampled
        
        # return total loss and student reconstruction
        return total_ibot_loss, recon_composite, distill_loss, recon_loss_l1, alignment_loss


    # alignment loss (encourage image-text embeddings to become more similar in cosine space)
    # detach text embeddings so this term does not backprop through the text encoder
    def compute_alignment_loss(self, image_embeds, text_embeds, detach_text=True):

        # if not using text encoder, return zero loss
        if not self.use_text:
            return torch.tensor(0.0, device=self.device)

         # detach text embeddings if specified
        if detach_text:
            text_embeds = text_embeds.detach()

        # cosine similarity loss (1 - cosine sim) averaged over batch
        image_embeds = F.normalize(image_embeds, dim=-1)
        text_embeds = F.normalize(text_embeds, dim=-1)
        return 1 - F.cosine_similarity(image_embeds, text_embeds, dim=-1).mean()

    
    # function to get values for logging image and embeddings
    def log_image_and_embeddings(self, x, mask, student_out, texts, is_train=True, student_image_embed=None, text_embed=None, student_recon=None):

        # reconstruct image
        if student_recon is None:
            student_recon = self.reconstruction_head(student_out)
        recon_image = student_recon.clamp(0, 1)
        # recon_image = (student_recon - student_recon.min()) / (student_recon.max() - student_recon.min() + 1e-8)

        # get masked input
        masked_input = x.clone()
        masked_input[mask] = 0

        # count for how many images already logged
        log_count = self.train_log_count if is_train else self.val_log_count
        batches_for_logging = self.train_batches_for_logging if is_train else self.val_batches_for_logging

        # if counted images less than max, add more
        if log_count < self.max_log_images:
            num_to_add = min(self.max_log_images - log_count, x.shape[0])
            batches_for_logging.append((x[:num_to_add].detach().to(torch.float32).cpu(), 
                                        masked_input[:num_to_add].detach().to(torch.float32).cpu(), 
                                        recon_image[:num_to_add].detach().to(torch.float32).cpu()))
            if is_train:
                self.train_log_count += num_to_add
            else:
                self.val_log_count += num_to_add

        # only store embeddings if text encoder is used
        if self.use_text:
            embed_dict_attr = 'train_embeddings' if is_train else 'val_embeddings'
            embed_dict = getattr(self, embed_dict_attr, None)
            if embed_dict is None:
                embed_dict = {'image': [], 'text': [], 'label': []}
                setattr(self, embed_dict_attr, embed_dict)

        # if using text encoder, store both image and text embeddings
        if self.use_text:

            # store image embeddings
            if student_image_embed is None:
                student_image_embed = F.normalize(self.image_proj(student_out), dim=-1).detach().cpu()
            else:
                student_image_embed = student_image_embed.detach().cpu()

            # store text embeddings
            if text_embed is None:
                text_embed = self.encode_text(texts).detach().cpu()
            else:
                text_embed = text_embed.detach().cpu()

            # append to dict
            embed_dict['image'].append(student_image_embed)
            embed_dict['text'].append(text_embed)
            embed_dict['label'].extend(texts)

    
    # on train start
    def on_train_start(self):
        print(f'[DEBUG] RANK: {self.global_rank}, world_size={self.trainer.world_size}', flush=True)


    # on train epoch start
    def on_train_epoch_start(self):

        # check if still in first few epochs
        if self.current_epoch < self.warmup_epochs:

            # reduce masking for first few epochs
            self.mask_ratio = self.mask_ratio_warmup

            # if using text encoder, keep text fully frozen for stability
            if self.use_text:
                self._freeze_all_text()
                self.text_encoder.eval()

        else:
            self.mask_ratio = self.base_mask_ratio

            # unfreeze top k layers of text encoder if finetuning is enabled
            if self.use_text and self.finetune_text and self.current_epoch >= self.text_finetune_start_epoch:
            
                # train only top k text layers
                self._unfreeze_top_k_text_layers(self.text_top_k_layers)
                self.text_encoder.train()

                # disable gradient checkpointing
                try:
                    self.text_encoder.gradient_checkpointing_disable()
                except AttributeError:
                    pass

            # otherwise keep text encoder fully frozen
            elif self.use_text:
                self._freeze_all_text()
                self.text_encoder.eval()


    # shared step for train/val to keep code DRY
    # KL: uses normalized features for distillation loss at masked locations
    # L1: uses raw features for reconstruction loss at masked locations
    # logging: uses composite reconstruction (input unmasked + student masked regions)
    def shared_step(self, batch, batch_idx, is_train=True):

        # get input image-text pair
        # x shape: (B, 1, D, H, W); texts is a list of strings, 1 per image
        x = batch['image']
        texts = batch.get('text', None) # image-only dataset will not have text

        # squeeze stray trailing singleton dimension if present (B, C, D, H, W, 1) -> (B, C, D, H, W)
        if x.ndim == 6 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        # ensure channel first (B, 1, D, H, W), and if given (B, D, H, W, 1) then permute
        if x.ndim == 5 and x.shape[1] != 1 and x.shape[-1] == 1:
            x = x.permute(0, 4, 1, 2, 3).contiguous()

        # add channel dimension if missing
        if x.ndim == 4:
            x = x.unsqueeze(1) # (B, 1, D, H, W)

        # generate patch level mask that randomly zeros out a subset of 3d patches in input image
        mask = self.generate_patch_mask(x, self.mask_patch_size)

        # student (masked) and teacher (full) encodings
        student_feats_raw, student_out_norm = self.encode_image(x, mask, self.student_encoder)

        # teacher sees full input (no gradients)
        with torch.no_grad():
            _, teacher_out_norm = self.encode_image(x, 
                                                    torch.zeros_like(mask, dtype=torch.bool), 
                                                    self.teacher_encoder) # teacher does not use a mask

        # if using text encoder
        if self.use_text:

            # embed image and text
            student_image_embed = F.normalize(self.image_proj(student_out_norm), dim=-1)
            text_embed = self.encode_text(texts)

            # compute contrastive loss between image and text embeddings - corresponding aligned pairs should be close in embedding space
            clip_loss, logit_scale = self.compute_clip_loss(student_image_embed, text_embed)

        # else, set clip loss and logit scale to zero tensors
        else:
            student_image_embed, text_embed = None, None
            clip_loss, logit_scale = torch.tensor(0.0, device=self.device), torch.tensor(1.0, device=self.device)

        # compute iBOT loss (KL divergence between student and teacher on masked regions and L1 reconstruction on masked regions)
        ibot_loss, recon_composite, distill_loss, recon_loss_l1, align_loss = \
            self.compute_ibot_loss(student_out_norm, teacher_out_norm, x, mask, student_feats_raw, student_image_embed, text_embed)

        # combine both losses into single scaler to minimize
        # ibot loss already includes distill loss (on masked voxels), reconstruction loss (on masked voxels), and alignment loss (cosine alignment between image-text pairs)
        # clip loss is image-text contrastive loss
        total_loss = ibot_loss + self.clip_weight * clip_loss

        # log losses
        log_prefix = 'train' if is_train else 'val'
        self.log(f'{log_prefix}_loss', total_loss, on_step=is_train, on_epoch=True, prog_bar=True, batch_size=x.shape[0], sync_dist=True)
        self.log(f'{log_prefix}_clip_loss', clip_loss, on_step=True, on_epoch=True, batch_size=x.shape[0], sync_dist=True)
        self.log(f'{log_prefix}_ibot_loss', ibot_loss, on_step=True, on_epoch=True, batch_size=x.shape[0], sync_dist=True)
        self.log(f'{log_prefix}_distill_loss', distill_loss, on_step=True, on_epoch=True, batch_size=x.shape[0], sync_dist=True)
        self.log(f'{log_prefix}_recon_loss_l1', recon_loss_l1, on_step=True, on_epoch=True, batch_size=x.shape[0], sync_dist=True)
        self.log(f'{log_prefix}_align_loss', align_loss, on_step=True, on_epoch=True, batch_size=x.shape[0], sync_dist=True)
        self.log(f'{log_prefix}_logit_scale', logit_scale, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

        # use composited reconstruction for logging (output is identical to input on unmasked regions)
        recon_for_log = F.interpolate(recon_composite, size=x.shape[2:], mode='trilinear', align_corners=False) # resize to input size for logging
        self.log_image_and_embeddings(x, mask, student_out_norm, texts, 
                                      is_train=is_train, 
                                      student_image_embed=student_image_embed, 
                                      text_embed=text_embed, 
                                      student_recon=recon_for_log)

        # return total loss
        return total_loss

    
    # training step
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, is_train=True)
    

    # validation step (same as training but without backprop)
    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx, is_train=False)


    # after train epoch
    def on_train_epoch_end(self):

        # only log artifacts from rank 0 process in distributed training
        if not self.trainer.is_global_zero:
            return

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

        # log umap to wandb
        if hasattr(self, 'train_embeddings') and self.train_embeddings:

            # only run every 5 epochs and epoch 0
            if self.current_epoch % 5 == 0 or self.current_epoch == 0:

                log_embedding_umap_plot(
                    logger=self.logger,
                    embeddings_dict=self.train_embeddings,
                    epoch=self.current_epoch,
                    global_step=self.global_step,
                    tag='Train'
                )

            # clear cache after each epoch
            del self.train_embeddings
            

    # after val epoch
    def on_validation_epoch_end(self):

        # only log artifacts from rank 0 process in distributed training
        if not self.trainer.is_global_zero:
            return

        # log images to wandb
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

        # log umap to wandb
        if hasattr(self, 'val_embeddings'):

            # only run every 5 epochs and epoch 0
            if self.current_epoch % 5 == 0 or self.current_epoch == 0:

                log_embedding_umap_plot(
                    logger=self.logger,
                    embeddings_dict=self.val_embeddings,
                    epoch=self.current_epoch,
                    global_step=self.global_step,
                    tag='Val'
                )

            # clear cache after each epoch
            del self.val_embeddings

        # update best metrics
        c = self.trainer.callback_metrics

        # train metrics
        self.update_best_metrics('train_loss', c.get('train_loss'))
        self.update_best_metrics('train_ibot_loss', c.get('train_ibot_loss'))

        # skip text specific metrics if not using text encoder
        if self.use_text:
            self.update_best_metrics('train_clip_loss', c.get('train_clip_loss'))

        # val metrics
        self.update_best_metrics('val_loss', c.get('val_loss'))
        self.update_best_metrics('val_ibot_loss', c.get('val_ibot_loss'))

        # skip text specific metrics if not using text encoder
        if self.use_text:
            self.update_best_metrics('val_clip_loss', c.get('val_clip_loss'))


    # on train batch end
    def on_train_batch_end(self, outputs, batch, batch_idx):

        # update ema teacher parameters from student (no grad)
        with torch.no_grad():
            m = float(self.ema_decay.item() if isinstance(self.ema_decay, torch.Tensor) else self.ema_decay)
            for student_param, teacher_param in zip(self.student_encoder.parameters(), self.teacher_encoder.parameters()):
                teacher_param.mul_(m).add_(student_param, alpha=1 - m)

    
    # after fit 
    def on_fit_end(self):

        # only rank 0 should log best metrics
        if not self.trainer.is_global_zero:
            return

        # mirror into wandb summary
        if hasattr(self, 'logger') and hasattr(self.logger, 'experiment'):
            exp = self.logger.experiment

            # create table for best metrics
            cols = ['best_train_loss', 'best_val_loss', 'best_train_ibot_loss', 'best_val_ibot_loss']
            if self.use_text:
                cols[2:2] = ['best_train_clip_loss', 'best_val_clip_loss'] # insert clip loss columns if using text encoder
            table = wandb.Table(columns=cols)

            # add data to table
            row = [
                self.best_metrics['train_loss'],
                self.best_metrics['val_loss'],
            ]
            if self.use_text:
                row += [
                    self.best_metrics['train_clip_loss'],
                    self.best_metrics['val_clip_loss'],
                ]
            row += [
                self.best_metrics['train_ibot_loss'],
                self.best_metrics['val_ibot_loss']
            ]
            table.add_data(*row)

            # log table to wandb
            exp.log({'best_metrics_table': table})

            # add metrics to log summary too
            for k, v in self.best_metrics.items():
                exp.summary[f'best_{k}'] = v


    # optimizer
    def configure_optimizers(self):
        return torch.optim.AdamW((p for p in self.parameters() if p.requires_grad), lr=self.lr) # skip frozen teacher params

    # function to update best metrics
    def update_best_metrics(self, name, value):
        if value is None:
            return
        try:
            v = float(value.detach().cpu().item() if hasattr(value, 'detach') else value)
        except Exception:
            return
        if v < self.best_metrics[name]:
            self.best_metrics[name] = v








