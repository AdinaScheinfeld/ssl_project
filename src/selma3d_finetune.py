# finetune_lsm_from_split.py

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import argparse
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

import monai
from monai.networks.nets import SwinUNETR
from monai.losses import DiceCELoss

# ---- Dataset ---- #

class SegPatchDataset(Dataset):
    def __init__(self, root_dir):
        self.paths = list(Path(root_dir).rglob("*.pt"))

    def __getitem__(self, idx):
        item = torch.load(self.paths[idx])
        image = item["image"]       # shape: (1 or 2, D, H, W)
        label = item["label"]       # shape: (D, H, W)

        if image.ndim == 3:
            image = image.unsqueeze(0)  # (1, D, H, W)
        if image.shape[0] == 1:
            image = torch.cat([image, torch.zeros_like(image)], dim=0)  # pad to (2, D, H, W)

        return {"image": image.float(), "label": label.long()}

    def __len__(self):
        return len(self.paths)

# ---- Model ---- #

class BinarySegmentationModule(pl.LightningModule):
    def __init__(self, pretrained_ckpt=None, lr=1e-4):
        super().__init__()
        self.save_hyperparameters()

        self.model = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=2,
            out_channels=1,
            feature_size=48,
            use_checkpoint=True
        )
        self.lr = lr
        self.loss_fn = DiceCELoss(sigmoid=True)

        if pretrained_ckpt:
            state_dict = torch.load(pretrained_ckpt)['state_dict']
            encoder_weights = {k.replace("encoder.", "model."): v for k, v in state_dict.items() if k.startswith("encoder.")}
            self.model.load_state_dict(encoder_weights, strict=False)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["image"])
        loss = self.loss_fn(logits, batch["label"].unsqueeze(1).float())
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch["image"])
        loss = self.loss_fn(logits, batch["label"].unsqueeze(1).float())
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

# ---- Main ---- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_root", required=True, help="Root directory with train/ and val/ folders")
    parser.add_argument("--pretrained_ckpt", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--project_name", type=str, default="lsm_finetune_from_split")
    args = parser.parse_args()

    pl.seed_everything(42)
    wandb_logger = WandbLogger(project=args.project_name)

    train_loader = DataLoader(SegPatchDataset(Path(args.split_root) / "train"), batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(SegPatchDataset(Path(args.split_root) / "val"), batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = BinarySegmentationModule(pretrained_ckpt=args.pretrained_ckpt, lr=args.lr)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=10, mode="min"),
        ModelCheckpoint(monitor="val_loss", save_top_k=1, mode="min", filename="best")
    ]

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        logger=wandb_logger,
        callbacks=callbacks,
        accelerator="gpu",
        devices=1,
        log_every_n_steps=5
    )

    trainer.fit(model, train_loader, val_loader)
