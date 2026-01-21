# /home/ads4015/ssl_project/models/deblur_module_unet.py
# PyTorch Lightning module for 3D deblurring with MONAI UNet (residual prediction)

import wandb

from monai.losses import SSIMLoss
from monai.networks.nets import Unet

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeblurModuleUNet(pl.LightningModule):
    def __init__(
        self,
        pretrained_ckpt_path: str = None,
        lr: float = 1e-4,
        encoder_lr_mult: float = 0.1,
        weight_decay: float = 1e-5,
        freeze_encoder_epochs: int = 0,
        freeze_bn_stats: int = 0,
        # UNet arch
        channels=(32, 64, 128, 256),
        strides=(2, 2, 2, 1),
        num_res_units: int = 2,
        norm: str = "BATCH",
        # loss weights
        edge_weight: float = 0.1,
        highfreq_weight: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.val_examples_table = None

        # 3D UNet that outputs residual (same shape as input)
        self.model = Unet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=tuple(int(x) for x in channels),
            strides=tuple(int(x) for x in strides),
            num_res_units=int(num_res_units),
            norm=str(norm),
        )

        # encoder freeze schedule (mirrors your UNet segmentation heuristic)
        self.freeze_encoder_epochs = int(freeze_encoder_epochs)
        self.encoder_frozen = self.freeze_encoder_epochs > 0
        self.freeze_bn_stats = bool(int(freeze_bn_stats))

        # losses
        self.l1_loss = nn.L1Loss(reduction="mean")
        self.ssim_loss = SSIMLoss(spatial_dims=3, data_range=1.0)
        self.edge_weight = float(edge_weight)
        self.highfreq_weight = float(highfreq_weight)

        # optim hparams
        self.lr = float(lr)
        self.encoder_lr_mult = float(encoder_lr_mult)

        # load pretrained backbone weights if provided
        if pretrained_ckpt_path:
            self._load_pretrained_weights(pretrained_ckpt_path)

    def _load_pretrained_weights(self, ckpt_path: str):
        print(f"[INFO] DeblurUNetModule: loading pretrained checkpoint: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        # Map: student_encoder.<rest>  ->  model.<rest>
        mapped = {}
        for k, v in state_dict.items():
            if k.startswith("student_encoder."):
                rest = k[len("student_encoder.") :]
                mapped["model." + rest] = v

        # keep only compatible tensors
        model_sd = self.state_dict()
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

        incompat = self.load_state_dict(safe, strict=False)
        print(
            f"[INFO] DeblurUNetModule: pretrained load kept={len(safe)} dropped={len(dropped)} "
            f"missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}",
            flush=True,
        )

        # optionally freeze encoder params initially
        if self.encoder_frozen:
            for name, p in self.model.named_parameters():
                # same heuristic you used before: treat model.2.* as "head"
                if not name.startswith("model.2."):
                    p.requires_grad = False
            print(f"[INFO] DeblurUNetModule: encoder frozen for first {self.freeze_encoder_epochs} epochs", flush=True)

    def forward(self, x):
        # residual prediction
        return self.model(x)

    @staticmethod
    def _grad3d(x):
        dz_core = x[:, :, 1:, :, :] - x[:, :, :-1, :, :]
        dz = F.pad(dz_core, (0, 0, 0, 0, 0, 1))

        dy_core = x[:, :, :, 1:, :] - x[:, :, :, :-1, :]
        dy = F.pad(dy_core, (0, 0, 0, 1, 0, 0))

        dx_core = x[:, :, :, :, 1:] - x[:, :, :, :, :-1]
        dx = F.pad(dx_core, (0, 1, 0, 0, 0, 0))
        return dz, dy, dx

    @staticmethod
    def _highpass3d(x, kernel_size=3):
        B, C, D, H, W = x.shape
        weight = x.new_ones((C, 1, kernel_size, kernel_size, kernel_size)) / float(kernel_size**3)
        smoothed = F.conv3d(x, weight, bias=None, stride=1, padding=kernel_size // 2, groups=C)
        return x - smoothed

    def on_train_start(self):
        if self.freeze_bn_stats:
            bn = 0
            for m in self.model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
                    bn += 1
            print(f"[INFO] DeblurUNetModule: froze BatchNorm layers: {bn}", flush=True)

    def on_train_epoch_start(self):
        # unfreeze after warmup
        if self.encoder_frozen and self.current_epoch < self.freeze_encoder_epochs:
            for name, p in self.model.named_parameters():
                if not name.startswith("model.2."):
                    p.requires_grad = False
        elif self.encoder_frozen and self.current_epoch >= self.freeze_encoder_epochs:
            print(f"[INFO] DeblurUNetModule: unfreezing encoder at epoch={self.current_epoch}", flush=True)
            for _, p in self.model.named_parameters():
                p.requires_grad = True
            self.encoder_frozen = False

    def _compute_losses_and_metrics(self, blurred, sharp):
        residual = self(blurred)
        pred = torch.clamp(blurred + residual, 0.0, 1.0)

        loss_l1 = self.l1_loss(pred, sharp)
        loss_ssim = self.ssim_loss(pred, sharp)  # SSIMLoss is (1 - ssim)

        dz_p, dy_p, dx_p = self._grad3d(pred)
        dz_t, dy_t, dx_t = self._grad3d(sharp)
        loss_edge = (
            (dz_p.abs() - dz_t.abs()).abs().mean()
            + (dy_p.abs() - dy_t.abs()).abs().mean()
            + (dx_p.abs() - dx_t.abs()).abs().mean()
        )

        hf_p = self._highpass3d(pred)
        hf_t = self._highpass3d(sharp)
        loss_hf = self.l1_loss(hf_p, hf_t)

        total = loss_l1 + 0.2 * loss_ssim + self.edge_weight * loss_edge + self.highfreq_weight * loss_hf

        mse = torch.mean((pred - sharp) ** 2) + 1e-8
        psnr = 10.0 * torch.log10(1.0 / mse)

        return pred, total, loss_l1, loss_ssim, loss_edge, loss_hf, psnr

    def training_step(self, batch, batch_idx):
        blurred = batch["input_vol"].to(self.device)   # (B,1,D,H,W)
        sharp = batch["target_vol"].to(self.device)    # (B,1,D,H,W)

        pred, total, l1, ssim_l, edge, hf, psnr = self._compute_losses_and_metrics(blurred, sharp)

        self.log("train_l1_loss", l1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_edge_loss", edge, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_highfreq_loss", hf, on_step=False, on_epoch=True, prog_bar=False)
        self.log("train_total_loss", total, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_ssim", 1.0 - ssim_l, on_step=True, on_epoch=True, prog_bar=False)
        self.log("train_psnr", psnr, on_step=True, on_epoch=True, prog_bar=True)
        return total

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        blurred = batch["input_vol"].to(self.device)
        sharp = batch["target_vol"].to(self.device)

        pred, total, l1, ssim_l, edge, hf, psnr = self._compute_losses_and_metrics(blurred, sharp)

        # W&B table: up to 5 samples
        if batch_idx < 5 and isinstance(self.logger, WandbLogger) and hasattr(self.logger, "experiment"):
            if self.val_examples_table is None:
                self.val_examples_table = wandb.Table(
                    columns=["Filename", "Blurred (mid-z)", "Deblurred (mid-z)", "Sharp (mid-z)"]
                )

            filename = batch["filename"][0] if isinstance(batch, dict) and "filename" in batch else f"val_{batch_idx}"

            b_np = blurred[0, 0].detach().float().cpu().numpy()
            p_np = pred[0, 0].detach().float().cpu().numpy()
            s_np = sharp[0, 0].detach().float().cpu().numpy()

            mid = b_np.shape[0] // 2
            self.val_examples_table.add_data(
                filename,
                wandb.Image(b_np[mid], caption="blurred"),
                wandb.Image(p_np[mid], caption="deblurred"),
                wandb.Image(s_np[mid], caption="sharp"),
            )

        self.log("val_l1_loss", l1, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_edge_loss", edge, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_highfreq_loss", hf, on_step=False, on_epoch=True, prog_bar=False)
        self.log("val_total_loss", total, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_ssim", 1.0 - ssim_l, on_step=True, on_epoch=True, prog_bar=True)
        self.log("val_psnr", psnr, on_step=True, on_epoch=True, prog_bar=True)

        return {"val_total_loss": total}

    def on_validation_epoch_end(self):
        if self.val_examples_table is not None and isinstance(self.logger, WandbLogger) and hasattr(self.logger, "experiment"):
            self.logger.experiment.log({f"val_examples_epoch_{self.current_epoch}": self.val_examples_table})
        self.val_examples_table = None

    def configure_optimizers(self):
        base_lr = float(self.hparams.lr)
        wd = float(self.hparams.weight_decay)
        enc_mult = float(self.hparams.encoder_lr_mult)

        backbone, head = [], []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if name.startswith("model.2."):
                head.append(p)
            else:
                backbone.append(p)

        opt = torch.optim.AdamW(
            [{"params": backbone, "lr": base_lr * enc_mult},
             {"params": head, "lr": base_lr}],
            weight_decay=wd,
            betas=(0.9, 0.999),
        )

        sched = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt,
            T_max=int(self.trainer.max_epochs),
            eta_min=base_lr * 0.1,
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sched, "interval": "epoch"},
        }
