# inpaint_module.py - Text conditioned 3D image inpainiting Lightning Module

# --- Setup ---

# imports
import wandb

from monai.data.meta_tensor import MetaTensor
from monai.losses import SSIMLoss
from monai.networks.nets import SwinUNETR

import pytorch_lightning as pl

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.serialization import add_safe_globals

# get build_components from your pretrain utils
from my_pretrain_utils import build_components


# --- Text Conditioning ---

# FiLM-style adaptor (maps text embedding to per-channel gamma/beta vectors that can modulate features)
class TextConditioner(nn.Module):

    # init
    def __init__(self, text_dim=512, feature_dim=768):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(text_dim, feature_dim), # project text embedding to feature dim
            nn.GELU(), # non-linearity
            nn.Linear(feature_dim, feature_dim * 2) # output gamma and beta vectors
        )

    # forward
    def forward(self, text_emb):

        # get gamma and beta from MLP
        gb = self.mlp(text_emb) # (batch, feature_dim * 2)
        half = gb.shape[1] // 2 # gb is concatenated gamma and beta
        return gb[:, :half], gb[:, half:] # return gamma and beta separately
    

# dependency-free text encocder for quick testing
class DummyTextEncoder(nn.Module):

    # init
    def __init__(self, dim=512, vocab_size=2048):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    # method to hash text to token ids
    @staticmethod
    def _hash_chars(s, vocab_size):

        # get unicode code point for each character of fist 256, multiply by large constant to spread out and get valid embedding index
        idxs = [(ord(c) * 1315423911) % vocab_size for c in s[:256]] or [0]
        return torch.tensor(idxs, dtype=torch.long) # return as long tensor
    
    # forward
    def forward(self, texts):

        # list for batches
        batch_embs = []

        # process each text individually
        for s in texts:
            ids = self._hash_chars(s or "", self.vocab_size).to(self.embedding.weight.device) # get token ids
            emb = self.embedding(ids) # (seq_len, dim)
            batch_embs.append(emb.mean(0, keepdim=True)) # mean pool to get single embedding per text
        return torch.cat(batch_embs, dim=0) # (batch, dim)
    

# CLIP text encoder wrapper
# expects helper: `your_pretrain_utils.build_components(ckpt_path, strict=False)` that returns (img_encode_fn, txt_encode_fn, device_str)
class CLIPTextEncoderWrapper(nn.Module):

    # init
    def __init__(self, ckpt_path):
        super().__init__()

        # use build_components to get text encoder
        _, txt_encode_fn, _ = build_components(ckpt_path=ckpt_path, strict=False)
        if txt_encode_fn is None:
            raise RuntimeError('build_components did not return a text encoder function')
        self._encode = txt_encode_fn

    # forward
    def forward(self, texts):

        # get text embeddings from CLIP text encoder
        embs = self._encode(texts) # (batch, dim)
        if not isinstance(embs, torch.Tensor):
            embs = torch.tensor(embs)

        return embs
    

# --- Inpaint Module ---
class InpaintModule(pl.LightningModule):

    # init
    def __init__(
            self,
            pretrained_ckpt_path=None, # path to pretrained checkpoint for text encoder
            lr=1e-4, # learning rate
            feature_size=24, # swinUNETR width
            encoder_lr_mult=0.05, # learning rate multiplier for encoder (smaller to avoid catastrophic forgetting)
            freeze_encoder_epochs=0, # number of epochs to freeze encoder (stabilize training)
            l1_weight_masked=1.0, # masked L1 loss weight
            l1_weight_global=0.1, # global L1 loss weight (small for stability)
            text_cond=True, # whether to use text conditioning
            text_dim=512, # text embedding dimension (not used, inferred from encoder)
            text_backend='clip', # text encoder backend: 'clip' or 'dummy'
            clip_ckpt=None, # path to CLIP checkpoint (if text_backend is 'clip')
            weight_decay=1e-5 # optimizer weight decay
    ):
        
        super().__init__()
        self.save_hyperparameters()

        # build model for 2 channel input (masked image, mask)
        self.model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            feature_size=feature_size,
            use_checkpoint=True
        )

        # optionally freeze encoder for some epochs
        self.freeze_encoder_epochs = int(freeze_encoder_epochs)
        self.encoder_frozen = self.freeze_encoder_epochs > 0

        # build text encoder
        self.text_cond = bool(text_cond)
        self._text_backend = text_backend
        if self.text_cond:
            if text_backend == 'clip':
                if not clip_ckpt:
                    raise ValueError('clip_ckpt must be provided if text_backend is "clip"')
                self.text_encoder = CLIPTextEncoderWrapper(ckpt_path=clip_ckpt)
                with torch.no_grad():
                    d = int(self.text_encoder(['dummy']).shape[1])
                text_dim = d # infer text dim
            else: # dummy
                self.text_encoder = DummyTextEncoder(dim=text_dim)

            bottleneck_dim = 768 if feature_size <=24 else 1024 # swinUNETR bottleneck dim
            self.text_adaptor = TextConditioner(text_dim=text_dim, feature_dim=bottleneck_dim) # FiLM adaptor

            # learnable gate for text conditioning (start with 0 so text does not collapse training)
            self.text_gate = nn.Parameter(torch.tensor(0.0))

        # losses and lr
        self.l1_masked = nn.L1Loss(reduction='mean')
        self.l1_global = nn.L1Loss(reduction='mean')
        self.ssim_masked = SSIMLoss(spatial_dims=3, data_range=1.0)
        self.edge_weight = 0.1 # small weight for global loss to stabilize training
        self.lr = float(lr)
        self.encoder_lr_mult = float(encoder_lr_mult)

        self.logged_images = 0 # counter for wandb logged images

        # load encoder weights from pretrained ckpt
        if pretrained_ckpt_path:
            add_safe_globals([MetaTensor]) # allow MetaTensor in checkpoint
            ckpt = torch.load(pretrained_ckpt_path, map_location='cpu', weights_only=False)
            state_dict = ckpt.get('state_dict', ckpt)

            # function to strip common Lightning/wrapper prefixes
            def _strip_prefix(key):
                for p in ('model.', 'module.', 'student_model.', 'student.', 'teacher_model.',
                          'net.', 'encoder.', 'student_encoder.', 'backbone.'):
                    if key.startswith(p):
                        return key[len(p):]
                return key
            
            # build new state dict with stripped keys
            # remap any keys with 'swinViT.' to monai model's 'swinViT.' subtree
            mapped = {}
            for k, v in state_dict.items():
                stripped_k = _strip_prefix(k)
                if stripped_k.startswith('swinViT.'):
                    mapped[stripped_k] = v
                elif 'swinViT.' in stripped_k:
                    _, tail = stripped_k.split('swinViT.', 1)
                    mapped['swinViT.' + tail] = v

            # fix in_channels mismatch (2 vs 1) for first conv layer
            wkey = 'swinViT.patch_embed.proj.weight'
            if wkey in mapped:
                w_ckpt = mapped[wkey] # (out_c, in_c, kD, kH, kW)
                w_tgt_shape = self.model.swinViT.patch_embed.proj.weight.shape # (out_c, in_c, kD, kH, kW)
                if tuple(w_ckpt.shape) != tuple(w_tgt_shape):

                    # if checkpoint was trained with in_channels=1, repeat weights for 2 channels
                    if w_ckpt.shape[1] == 1 and w_tgt_shape[1] == 2:
                        w_new = torch.zeros(w_tgt_shape, dtype=w_ckpt.dtype)

                        # copy image kernel only to channel 0 (image), keep mask channel 1 initialized to zero
                        # to prevent first layer from hard-coding box edges
                        w_new[:, 0] = w_ckpt[:, 0]
                        mapped[wkey] = w_new
                        print(f'[INFO] Adjusted {wkey} from {w_ckpt.shape} to {w_new.shape} by copying image kernel to channel 0, zero init channel 1')
                    else:
                        print(f'[WARN] Shape incompatible, dropping tensor and init from scratch: {wkey} ckpt shape {w_ckpt.shape}, target shape {w_tgt_shape}')
                        mapped.pop(wkey)

            # sanity check bias shape
            bkey = 'swinViT.patch_embed.proj.bias'
            if bkey in mapped:
                b_ckpt = mapped[bkey]
                b_tgt_shape = self.model.swinViT.patch_embed.proj.bias.shape
                if tuple(b_ckpt.shape) != tuple(b_tgt_shape):
                    print(f'[WARN] Shape incompatible, dropping tensor and init from scratch: {bkey} ckpt shape {b_ckpt.shape}, target shape {b_tgt_shape}')
                    mapped.pop(bkey)

            # load state dict
            incompatible = self.model.load_state_dict(mapped, strict=False)
            print(f'[INFO] Encoder load: missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}')

            # freeze encoder if needed
            if self.encoder_frozen:
                for name, p in self.model.named_parameters():
                    if name.startswith('swinViT'):
                        p.requires_grad = False
                print(f'[INFO] Encoder frozen for first {self.freeze_encoder_epochs} epochs')

    # forward pass
    # x: (B, 1, D, H, W) input masked volume
    # mask: (B, 1, D, H, W) binary mask volume
    # t_emb: (B, text_dim) text embedding tensor (optional)
    def forward(self, x, mask, t_emb=None):

        input = torch.cat([x, mask], dim=1) # (B, 2, D, H, W) concatenate masked image and mask
        output = self.model(input) # (B, 1, D, H, W) raw model output

        # only allow a masked region offset and learn its strength from data
        if self.text_cond and t_emb is not None:

            # robust 0-mean scalar from text
            t_scalar = torch.tanh(t_emb.mean(dim=-1, keepdim=True)).view(-1, 1, 1, 1, 1) # (B, 1, 1, 1, 1)
            output = output + self.text_gate * t_scalar * mask # modulate only in masked region
        return output

    # *** Training ***

    # helper to erode loss/metric mask (to reduce hard edges)
    @staticmethod
    def _erode_mask(mask, k):
        m = mask
        for _ in range(k):
            m = 1.0 - torch.nn.functional.max_pool3d(
                1.0 - m, kernel_size=3, stride=1, padding=1
            )
        return m

    # finite difference 3D gradient
    @staticmethod
    def _grad3d(x):

        # d/dz (depth axis = 2)
        dz_core = x[:, :, 1:, :, :] - x[:, :, :-1, :, :] # central differences
        dz = F.pad(dz_core, (0, 0, 0, 0, 0, 1)) # pad last slice

        # d/dy (height axis = 3)
        dy_core = x[:, :, :, 1:, :] - x[:, :, :, :-1, :] # central differences
        dy = F.pad(dy_core, (0, 0, 0, 1, 0, 0)) # pad last row

        # d/dx (width axis = 4)
        dx_core = x[:, :, :, :, 1:] - x[:, :, :, :, :-1] # central differences
        dx = F.pad(dx_core, (0, 1, 0, 0, 0, 0)) # pad last column

        return dz, dy, dx

    # on train epoch start
    def on_train_epoch_start(self):

        # unfreeze encoder after warmup period
        if self.encoder_frozen and self.current_epoch >= self.freeze_encoder_epochs:
            for name, p in self.model.named_parameters():
                p.requires_grad = True
            self.encoder_frozen = False
            print(f'[INFO] Encoder unfrozen at epoch {self.current_epoch}')

    # training step
    def training_step(self, batch, _):

        # get batch data
        masked_vol = batch['masked_vol'].to(self.device) # (B,1,D,H,W)
        mask = batch['mask'].to(self.device) # (B,1,D,H,W)
        target_vol = batch['target_vol'].to(self.device) # (B,1,D,H,W)
        
        # get text embeddings if needed
        t_emb = None
        if self.text_cond and 'text' in batch:
            t_emb = self.text_encoder(batch['text']).to(self.device) # (B, text_dim)

        # forward pass
        pred_logits = self(masked_vol, mask, t_emb) # raw
        pred = torch.sigmoid(pred_logits) # (B,1,D,H,W) sigmoid output to [0, 1] for losses/metrics

        # composite output (keep unmasked regions from input, inpaint only masked regions)
        composite = masked_vol * (1.0 - mask) + pred * mask # (B,1,D,H,W)

        # erode mask to reduce boundary bias in loss
        mask_eroded = self._erode_mask(mask, k=1)

        # compute losses (strong loss inside masked region, weak loss globally)
        loss_masked = self.l1_masked(pred * mask_eroded, target_vol * mask_eroded)
        loss_ssim = self.ssim_masked(pred * mask_eroded, target_vol * mask_eroded)

        # edge/gradient l1 only where masked (to encourage structure rather than smooth fill)
        pdz, pdy, pdx = self._grad3d(pred * mask_eroded)
        tdz, tdy, tdx = self._grad3d(target_vol * mask_eroded)
        loss_edge = (pdz.abs() - tdz.abs()).abs().mean() + \
                    (pdy.abs() - tdy.abs()).abs().mean() + \
                    (pdx.abs() - tdx.abs()).abs().mean()
        loss_global = self.l1_global(composite, target_vol)
        loss = (
            self.hparams.l1_weight_masked * loss_masked
            + self.hparams.l1_weight_global * loss_global
            + self.edge_weight * loss_edge
            + 0.2 * loss_ssim
        )

        # log losses
        self.log('train_l1_loss_masked', loss_masked, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_l1_loss_global', loss_global, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_l1_loss_edge', loss_edge, on_step=True, on_epoch=True, prog_bar=False)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_ssim_loss', 1.0-loss_ssim, on_step=False, on_epoch=True, prog_bar=False)
        return loss
    
    # *** Validation ***

    # on validation start
    def on_validation_epoch_start(self):
        self.logged_images = 0 # reset logged images counter
        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.val_table = wandb.Table(columns=['Filename', 'Masked Input (mid-z)', 'Mask', 'Composite pred (mid-z)', 'Target (mid-z)'])

    # validation step
    @torch.no_grad()
    def validation_step(self, batch, _):

        # get batch data
        masked_vol = batch['masked_vol'].to(self.device) # (B,1,D,H,W)
        mask = batch['mask'].to(self.device) # (B,1,D,H,W)
        target_vol = batch['target_vol'].to(self.device) # (B,1,D,H,W)

        # get text embeddings if needed
        t_emb = None
        if self.text_cond and 'text' in batch:
            t_emb = self.text_encoder(batch['text']).to(self.device) # (B, text_dim)

        # forward pass
        pred_logits = self(masked_vol, mask, t_emb) # raw
        pred = torch.sigmoid(pred_logits) # (B,1,D,H,W) sigmoid output to [0, 1] for losses/metrics

        # composite output (keep unmasked regions from input, inpaint only masked regions)
        composite = masked_vol * (1.0 - mask) + pred * mask # (B,1,D,H,W)

        # erode mask to reduce boundary bias in loss
        mask_eroded = self._erode_mask(mask, k=1)

        # validation metrics (psnr in masked region)
        loss_masked = self.l1_masked(pred * mask_eroded, target_vol * mask_eroded)
        loss_ssim = self.ssim_masked(pred * mask_eroded, target_vol * mask_eroded)
        pdz, pdy, pdx = self._grad3d(pred * mask_eroded)
        tdz, tdy, tdx = self._grad3d(target_vol * mask_eroded)
        loss_edge = (pdz.abs() - tdz.abs()).abs().mean() + \
                    (pdy.abs() - tdy.abs()).abs().mean() + \
                    (pdx.abs() - tdx.abs()).abs().mean()
        loss_global = self.l1_global(composite, target_vol)
        loss = (
            self.hparams.l1_weight_masked * loss_masked
            + self.hparams.l1_weight_global * loss_global
            + self.edge_weight * loss_edge
            + 0.2 * loss_ssim
        )

        # compute PSNR in masked region (peak signal to noise ratio)
        mse_masked = torch.mean(((pred - target_vol) * mask_eroded) ** 2) + 1e-8
        psnr_masked = 10.0 * torch.log10(1.0 / mse_masked)

        # log metrics
        self.log('val_l1_loss_masked', loss_masked, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_l1_loss_global', loss_global, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_l1_loss_edge', loss_edge, on_step=False, on_epoch=True, prog_bar=False)
        self.log('val_loss', loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_psnr_masked', psnr_masked, on_step=False, on_epoch=True, prog_bar=True)
        self.log('val_ssim_loss', 1.0-loss_ssim, on_step=False, on_epoch=True, prog_bar=False)

        # log images to wandb (only first few batches)
        if isinstance(self.logger, pl.loggers.WandbLogger) and self.logged_images < 5:
            B, _, D, H, W = target_vol.shape
            num_log = min(5 - self.logged_images, B)
            for i in range(num_log):
                z = D // 2 # middle slice index
                m_img = masked_vol[i, 0, z].detach().cpu().numpy()
                m_msk = mask[i, 0, z].detach().cpu().numpy()
                cp_img = composite[i, 0, z].detach().cpu().numpy()
                t_img = target_vol[i, 0, z].detach().cpu().numpy()
                self.val_table.add_data(
                    batch['filename'][i],
                    wandb.Image(m_img),
                    wandb.Image(m_msk),
                    wandb.Image(cp_img),
                    wandb.Image(t_img)
                )
                self.logged_images += 1
        return {'val_loss': loss}
    
    # on validation epoch end
    def on_validation_epoch_end(self):
        if isinstance(self.logger, pl.loggers.WandbLogger) and self.logged_images > 0:
            self.logger.experiment.log({f'val_examples_epoch_{self.current_epoch}': self.val_table})

# *** Optimizer ***

    # configure optimizers
    def configure_optimizers(self):

        # lists for parameter groups
        encoder, decoder = [], []

        # separate encoder and decoder parameters
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            (encoder if name.startswith('swinViT') else decoder).append(p)

        # build optimizer with different lr for encoder and decoder
        optimizer = torch.optim.AdamW(
            [
                {'params': encoder, 'lr': self.lr * self.encoder_lr_mult},
                {'params': decoder, 'lr': self.lr},
            ],
            weight_decay=float(self.hparams.weight_decay)
        )

        # per epoch lr scheduler
        tmax = getattr(self.trainer, 'max_epochs', None) or 500
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=int(tmax), eta_min=self.lr * 0.1)

        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': lr_scheduler, 'interval': 'epoch'}}




















