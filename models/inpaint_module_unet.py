# /home/ads4015/ssl_project/models/inpaint_module_unet.py
# Text-conditioned 3D inpainting Lightning Module with MONAI UNet backbone

import wandb
import numpy as np

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F

from monai.losses import SSIMLoss
from monai.networks.nets import Unet

from my_pretrain_utils import build_components

# -------------------------
# Text conditioning helpers (same style as your Swin module)
# -------------------------

class DummyTextEncoder(nn.Module):
    def __init__(self, dim=512, vocab_size=2048):
        super().__init__()
        self.dim = dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, dim)
        nn.init.normal_(self.embedding.weight, std=0.02)

    @staticmethod
    def _hash_chars(s, vocab_size):
        idxs = [(ord(c) * 1315423911) % vocab_size for c in s[:256]] or [0]
        return torch.tensor(idxs, dtype=torch.long)

    def forward(self, texts):
        embs = []
        for s in texts:
            ids = self._hash_chars(s or "", self.vocab_size).to(self.embedding.weight.device)
            e = self.embedding(ids)             # (seq, dim)
            embs.append(e.mean(0, keepdim=True)) # (1, dim)
        return torch.cat(embs, dim=0)           # (B, dim)
    
# -------------------------
# CLIP text encoder wrapper (mirrors inpaint_module.py)
# -------------------------

class CLIPTextEncoderWrapper(nn.Module):
    """
    Uses your pretrain helper:
      build_components(ckpt_path, strict=False) -> (img_encode_fn, txt_encode_fn, device_str)
    We only keep txt_encode_fn.
    """
    def __init__(self, ckpt_path: str):
        super().__init__()
        _, txt_encode_fn, _ = build_components(ckpt_path=ckpt_path, strict=False)
        if txt_encode_fn is None:
            raise RuntimeError("build_components did not return a text encoder function")
        self._encode = txt_encode_fn

    def forward(self, texts):
        embs = self._encode(texts)  # (B, dim)
        if not isinstance(embs, torch.Tensor):
            embs = torch.tensor(embs)
        return embs


# -------------------------
# Inpaint UNet Module
# -------------------------

class InpaintModuleUNet(pl.LightningModule):
    """
    Inputs:
      x:    (B,1,D,H,W) masked/noised input volume
      mask: (B,1,D,H,W) binary mask (1=hole)
      text: optional list[str] for conditioning

    Model input is 2 channels: [x, mask]
    Output is 1 channel logits -> sigmoid -> [0,1]
    """

    def __init__(
        self,
        pretrained_ckpt_path=None,  # your UNet pretrain ckpt
        lr=1e-4,
        encoder_lr_mult=0.05,
        freeze_encoder_epochs=5,
        l1_weight_masked=1.0,
        l1_weight_global=0.1,
        weight_decay=1e-5,
        # text conditioning (kept to match Swin experiment API)
        text_cond=True,
        text_dim=512,
        text_backend="dummy",
        clip_ckpt=None,
        # --- UNet architecture (MUST be part of hparams so load_from_checkpoint rebuilds correctly) ---
        unet_channels: str = "32,64,128,256,512",
        unet_strides: str = "2,2,2,1",
        unet_num_res_units: int = 2,
        unet_norm: str = "BATCH",
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---- UNet backbone for 2-channel input (masked_vol, mask) ----
        # NOTE: this MUST be constructed from __init__ args so that
        # InpaintModuleUNet.load_from_checkpoint(best_ckpt) recreates the same architecture.
        channels = tuple(int(x.strip()) for x in str(unet_channels).split(",") if x.strip())
        strides  = tuple(int(x.strip()) for x in str(unet_strides).split(",") if x.strip())
        if len(strides) != (len(channels) - 1):
            raise ValueError(
                f"UNet config invalid: need len(strides)==len(channels)-1, "
                f"got channels={channels} (len={len(channels)}), strides={strides} (len={len(strides)})"
            )

        self.model = Unet(
            spatial_dims=3,
            in_channels=2,
            out_channels=1,
            channels=channels,
            strides=strides,
            num_res_units=int(unet_num_res_units),
            norm=str(unet_norm),
        )

        # bookkeeping for which params are "encoder" (loaded from pretrain)
        self._encoder_param_names = set()

        # freeze encoder warmup
        self.freeze_encoder_epochs = int(freeze_encoder_epochs)
        self.encoder_frozen = self.freeze_encoder_epochs > 0

        # text conditioning (simple scalar gate like your Swin module)
        self.text_cond = bool(text_cond)
        self._text_backend = str(text_backend)

        if self.text_cond:
            if self._text_backend == "dummy":
                self.text_encoder = DummyTextEncoder(dim=int(text_dim))
                inferred_dim = int(text_dim)

            elif self._text_backend == "clip":
                if not clip_ckpt:
                    raise ValueError('clip_ckpt must be provided if text_backend="clip"')
                self.text_encoder = CLIPTextEncoderWrapper(ckpt_path=clip_ckpt)
                # infer dim like Swin module
                with torch.no_grad():
                    inferred_dim = int(self.text_encoder(["dummy"]).shape[1])

            else:
                raise ValueError(f"Unsupported text_backend: {self._text_backend} (expected 'dummy' or 'clip')")

            # learnable gate; starts at 0 so text can't destabilize training
            self.text_gate = nn.Parameter(torch.tensor(0.0))
            self._text_dim = inferred_dim

        # losses (match Swin)
        self.l1_masked = nn.L1Loss(reduction="mean")
        self.l1_global = nn.L1Loss(reduction="mean")
        self.ssim_masked = SSIMLoss(spatial_dims=3, data_range=1.0)
        self.edge_weight = 0.1

        # wandb logging for val samples
        self.logged_images = 0

        # load pretrained encoder weights
        if pretrained_ckpt_path:
            self._load_pretrained_unet_encoder(pretrained_ckpt_path)

    # -------------------------
    # Pretrained loader (robust, shape-safe, fixes in_channels=1 -> 2 for first conv)
    # -------------------------

    def _load_pretrained_unet_encoder(self, ckpt_path: str):
        print(f"[INFO] Loading pretrained UNet checkpoint: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)

        # your UNet pretrain stored encoder under student_encoder.*
        mapped = {}
        for k, v in sd.items():
            if k.startswith("student_encoder."):
                rest = k[len("student_encoder."):]   # e.g. "model.0.conv..."
                mapped_key = rest                    # we will match directly to model state_dict keys
                mapped[mapped_key] = v

        model_sd = self.model.state_dict()

        safe = {}
        dropped = []
        for k, v in mapped.items():
            if k not in model_sd:
                dropped.append((k, "not_in_model"))
                continue
            if tuple(v.shape) != tuple(model_sd[k].shape):
                dropped.append((k, f"shape {tuple(v.shape)} vs {tuple(model_sd[k].shape)}"))
                continue
            safe[k] = v

        # ---- fix 1ch -> 2ch for the *first* conv weight if present in checkpoint ----
        # We detect any conv kernel with shape (out, in, kD, kH, kW) where in==1 in ckpt and in==2 in model.
        # We copy ckpt channel 0 -> model channel 0, and zero-init channel 1 (mask channel).
        conv5_keys = [k for k, v in model_sd.items() if isinstance(v, torch.Tensor) and v.ndim == 5]
        for k in conv5_keys:
            tgt = model_sd[k]
            if tgt.shape[1] == 2:
                # candidate ckpt key with same out/kernel dims but in==1
                if k in mapped:
                    src = mapped[k]
                    if src.ndim == 5 and src.shape[1] == 1 and src.shape[0] == tgt.shape[0] and src.shape[2:] == tgt.shape[2:]:
                        w_new = torch.zeros_like(tgt)
                        w_new[:, 0] = src[:, 0]
                        # channel 1 stays zeros (mask channel)
                        safe[k] = w_new
                        print(f"[INFO] Adjusted first-conv {k}: {tuple(src.shape)} -> {tuple(w_new.shape)} (mask channel zero-init)", flush=True)
                        break

        # load
        incompat = self.model.load_state_dict(safe, strict=False)

        # record which params are encoder-loaded (for LR mult + freezing)
        for k in safe.keys():
            # state_dict keys map to parameter/buffer names; for params, use .named_parameters
            # We'll approximate by matching prefix up to last dot.
            pass

        # build param-name set by matching named_parameters
        safe_param_keys = set([k for k in safe.keys() if k in dict(self.model.named_parameters())])
        self._encoder_param_names = safe_param_keys

        print(
            f"[INFO] Pretrained load: kept={len(safe)} dropped={len(dropped)} "
            f"missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}",
            flush=True,
        )
        if self.encoder_frozen:
            self._set_encoder_requires_grad(False)
            print(f"[INFO] Encoder frozen for first {self.freeze_encoder_epochs} epochs", flush=True)

    def _set_encoder_requires_grad(self, flag: bool):
        # If we captured encoder param keys, freeze only those; else freeze everything except the last block.
        if self._encoder_param_names:
            for name, p in self.model.named_parameters():
                if name in self._encoder_param_names:
                    p.requires_grad = flag
        else:
            for _, p in self.model.named_parameters():
                p.requires_grad = flag

    # -------------------------
    # Forward
    # -------------------------

    def forward(self, x, mask, t_emb=None):
        inp = torch.cat([x, mask], dim=1)      # (B,2,D,H,W)
        out = self.model(inp)                  # (B,1,D,H,W) logits

        if self.text_cond and (t_emb is not None):
            # simple robust scalar from text; modulate only in masked region
            t_scalar = torch.tanh(t_emb.mean(dim=-1, keepdim=True)).view(-1, 1, 1, 1, 1)
            out = out + self.text_gate * t_scalar * mask

        return out

    # -------------------------
    # Training utilities (match Swin behavior)
    # -------------------------

    @staticmethod
    def _erode_mask(mask, k):
        m = mask
        for _ in range(k):
            m = 1.0 - F.max_pool3d(1.0 - m, kernel_size=3, stride=1, padding=1)
        return m

    @staticmethod
    def _grad3d(x):
        dz = F.pad(x[:, :, 1:, :, :] - x[:, :, :-1, :, :], (0, 0, 0, 0, 0, 1))
        dy = F.pad(x[:, :, :, 1:, :] - x[:, :, :, :-1, :], (0, 0, 0, 1, 0, 0))
        dx = F.pad(x[:, :, :, :, 1:] - x[:, :, :, :, :-1], (0, 1, 0, 0, 0, 0))
        return dz, dy, dx

    def on_train_epoch_start(self):
        if self.encoder_frozen and self.current_epoch >= self.freeze_encoder_epochs:
            self._set_encoder_requires_grad(True)
            self.encoder_frozen = False
            print(f"[INFO] Encoder unfrozen at epoch {self.current_epoch}", flush=True)

    def training_step(self, batch, _):
        masked_vol = batch["masked_vol"].to(self.device)
        mask = batch["mask"].to(self.device)
        target_vol = batch["target_vol"].to(self.device)

        t_emb = None
        if self.text_cond and ("text" in batch):
            t_emb = self.text_encoder(batch["text"]).to(self.device)

        pred_logits = self(masked_vol, mask, t_emb)
        pred = torch.sigmoid(pred_logits)

        composite = masked_vol * (1.0 - mask) + pred * mask
        mask_eroded = self._erode_mask(mask, k=1)

        loss_masked = self.l1_masked(pred * mask_eroded, target_vol * mask_eroded)
        loss_ssim = self.ssim_masked(pred * mask_eroded, target_vol * mask_eroded)

        pdz, pdy, pdx = self._grad3d(pred * mask_eroded)
        tdz, tdy, tdx = self._grad3d(target_vol * mask_eroded)
        loss_edge = (pdz.abs() - tdz.abs()).abs().mean() + (pdy.abs() - tdy.abs()).abs().mean() + (pdx.abs() - tdx.abs()).abs().mean()

        loss_global = self.l1_global(composite, target_vol)

        loss = (
            self.hparams.l1_weight_masked * loss_masked
            + self.hparams.l1_weight_global * loss_global
            + self.edge_weight * loss_edge
            + 0.2 * loss_ssim
        )

        self.log("train_l1_loss_masked", loss_masked, on_step=True, on_epoch=True)
        self.log("train_l1_loss_global", loss_global, on_step=True, on_epoch=True)
        self.log("train_l1_loss_edge", loss_edge, on_step=True, on_epoch=True)
        self.log("train_ssim", 1.0 - loss_ssim, on_step=False, on_epoch=True)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        return loss

    def on_validation_epoch_start(self):
        self.logged_images = 0
        if isinstance(self.logger, pl.loggers.WandbLogger):
            self.val_table = wandb.Table(
                columns=["Filename", "z_index", "Masked Input", "Mask", "Composite pred", "Target"]
            )

    @torch.no_grad()
    def validation_step(self, batch, _):
        masked_vol = batch["masked_vol"].to(self.device)
        mask = batch["mask"].to(self.device)
        target_vol = batch["target_vol"].to(self.device)

        t_emb = None
        if self.text_cond and ("text" in batch):
            t_emb = self.text_encoder(batch["text"]).to(self.device)

        pred_logits = self(masked_vol, mask, t_emb)
        pred = torch.sigmoid(pred_logits)

        composite = masked_vol * (1.0 - mask) + pred * mask
        mask_eroded = self._erode_mask(mask, k=1)

        loss_masked = self.l1_masked(pred * mask_eroded, target_vol * mask_eroded)
        loss_ssim = self.ssim_masked(pred * mask_eroded, target_vol * mask_eroded)

        pdz, pdy, pdx = self._grad3d(pred * mask_eroded)
        tdz, tdy, tdx = self._grad3d(target_vol * mask_eroded)
        loss_edge = (pdz.abs() - tdz.abs()).abs().mean() + (pdy.abs() - tdy.abs()).abs().mean() + (pdx.abs() - tdx.abs()).abs().mean()

        loss_global = self.l1_global(composite, target_vol)

        loss = (
            self.hparams.l1_weight_masked * loss_masked
            + self.hparams.l1_weight_global * loss_global
            + self.edge_weight * loss_edge
            + 0.2 * loss_ssim
        )

        mse_masked = torch.mean(((pred - target_vol) * mask_eroded) ** 2) + 1e-8
        psnr_masked = 10.0 * torch.log10(1.0 / mse_masked)

        self.log("val_l1_loss_masked", loss_masked, on_step=False, on_epoch=True)
        self.log("val_l1_loss_global", loss_global, on_step=False, on_epoch=True)
        self.log("val_l1_loss_edge", loss_edge, on_step=False, on_epoch=True)
        self.log("val_ssim", 1.0 - loss_ssim, on_step=False, on_epoch=True)
        self.log("val_psnr_masked", psnr_masked, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        # log a few examples
        if isinstance(self.logger, pl.loggers.WandbLogger) and self.logged_images < 5:
            B, _, D, H, W = target_vol.shape
            num_log = min(5 - self.logged_images, B)
            for i in range(num_log):
                mask_i = mask[i, 0]
                mask_per_z = mask_i.view(D, -1).sum(dim=1) > 0
                masked_z_idxs = torch.nonzero(mask_per_z, as_tuple=False).view(-1)
                if masked_z_idxs.numel() == 0:
                    z = D // 2
                else:
                    z = int(masked_z_idxs[len(masked_z_idxs) // 2].item())

                m_img = masked_vol[i, 0, z].detach().cpu().numpy()
                m_msk = mask[i, 0, z].detach().cpu().numpy()
                cp_img = composite[i, 0, z].detach().cpu().numpy()
                t_img = target_vol[i, 0, z].detach().cpu().numpy()

                self.val_table.add_data(
                    batch["filename"][i] if isinstance(batch["filename"], (list, tuple)) else batch["filename"],
                    z,
                    wandb.Image(m_img),
                    wandb.Image(m_msk),
                    wandb.Image(cp_img),
                    wandb.Image(t_img),
                )
                self.logged_images += 1

        return {"val_loss": loss}

    def on_validation_epoch_end(self):
        if isinstance(self.logger, pl.loggers.WandbLogger) and getattr(self, "logged_images", 0) > 0:
            self.logger.experiment.log({f"val_examples_epoch_{self.current_epoch}": self.val_table})

    def configure_optimizers(self):
        # encoder group = params that were loaded from ckpt (if we captured them), else fallback heuristic
        enc, dec = [], []
        enc_names = self._encoder_param_names

        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if enc_names and (name in enc_names):
                enc.append(p)
            else:
                dec.append(p)

        opt = torch.optim.AdamW(
            [
                {"params": enc, "lr": float(self.hparams.lr) * float(self.hparams.encoder_lr_mult)},
                {"params": dec, "lr": float(self.hparams.lr)},
            ],
            weight_decay=float(self.hparams.weight_decay),
        )

        tmax = getattr(self.trainer, "max_epochs", None) or 500
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=int(tmax), eta_min=float(self.hparams.lr) * 0.1)

        return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}
