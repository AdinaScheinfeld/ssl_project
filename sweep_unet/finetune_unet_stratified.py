#!/usr/bin/env python3
"""
finetune_unet_stratified.py

Finetunes a pretrained MONAI UNet checkpoint (student_encoder.*) for binary 3D segmentation.
- Data: /.../selma3d_finetune_patches/<datatype>/*.nii.gz and *_label.nii.gz
- Stratified split across datatypes (each datatype contributes proportionally to train/val/test)
- Uses finetune transforms from /home/ads4015/ssl_project/src/all_datasets_transforms.py
- Logs to W&B and supports sweeps
"""

import argparse
import os
from pathlib import Path
import random
import json
import numpy as np
import wandb

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from monai.networks.nets import Unet
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric

# import your transforms
import sys
sys.path.append("/home/ads4015/ssl_project/src")
from all_datasets_transforms import get_finetune_train_transforms, get_finetune_val_transforms

torch.set_float32_matmul_precision("medium")

# -------------------------
# Utils: split + dataset
# -------------------------

DATATYPES_DEFAULT = [
    "amyloid_plaque_patches",
    "c_fos_positive_patches",
    "cell_nucleus_patches",
    "vessels_patches",
]

def seed_all(seed: int):
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def discover_pairs_by_datatype(root: Path, datatypes):
    """
    Returns:
      pairs: list of dicts {image: path, label: path, dtype: datatype}
    """
    pairs = []
    for dtype in datatypes:
        ddir = root / dtype
        if not ddir.exists():
            continue

        for img_path in sorted(ddir.glob("*.nii*")):
            name = img_path.name.lower()
            if name.endswith("_label.nii") or name.endswith("_label.nii.gz"):
                continue

            suffix = "".join(img_path.suffixes)  # .nii or .nii.gz
            base = img_path.name[:-len(suffix)]
            lbl_path = img_path.with_name(f"{base}_label{suffix}")
            if lbl_path.exists():
                pairs.append({"image": str(img_path), "label": str(lbl_path), "dtype": dtype})

    return pairs

def stratified_split(pairs, seed, train_frac=0.7, val_frac=0.15, test_frac=0.15):
    """
    Stratify by datatype: split *within each dtype* then concatenate.
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6

    rng = random.Random(seed)
    by_dtype = {}
    for p in pairs:
        by_dtype.setdefault(p["dtype"], []).append(p)

    train, val, test = [], [], []
    for dtype, lst in by_dtype.items():
        lst = list(lst)
        rng.shuffle(lst)
        n = len(lst)
        n_train = int(round(n * train_frac))
        n_val = int(round(n * val_frac))
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        n_test = n - n_train - n_val

        train += lst[:n_train]
        val += lst[n_train:n_train + n_val]
        test += lst[n_train + n_val:]

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)
    return train, val, test

class NiftiPairDictDataset(Dataset):
    """
    MONAI dict transforms expect dicts: {"image": <path or tensor>, "label": <path or tensor>}
    We'll use nibabel inside transforms? Your finetune transforms currently assume image/label are arrays/tensors,
    BUT they include ScaleIntensityRangePercentilesd which expects ndarray/torch tensors.

    Since your finetune transforms as pasted do NOT include LoadImaged, we will load here with nibabel.
    """
    def __init__(self, items, transform=None):
        import nibabel as nib
        self.nib = nib
        self.items = items
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        it = self.items[idx]

        img = self.nib.load(it["image"]).get_fdata().astype(np.float32)
        lbl = self.nib.load(it["label"]).get_fdata().astype(np.float32)

        # ensure shapes are (D,H,W) and label is binary {0,1}
        if img.ndim == 4 and img.shape[-1] == 1:
            img = img[..., 0]
        if lbl.ndim == 4 and lbl.shape[-1] == 1:
            lbl = lbl[..., 0]

        # add channel dim -> (1,D,H,W)
        img = img[None, ...]
        lbl = lbl[None, ...]

        # binarize label safely
        lbl = (lbl > 0.5).astype(np.float32)

        d = {"image": img, "label": lbl, "dtype": it["dtype"], "image_path": it["image"]}
        if self.transform is not None:
            d = self.transform(d)
        return d

# -------------------------
# Lightning module
# -------------------------

class FinetuneUNet(pl.LightningModule):
    def __init__(
        self,
        pretrained_ckpt: str,
        lr: float = 1e-4,
        weight_decay: float = 1e-4,
        loss_name: str = "dicece",
        encoder_lr_mult: float = 0.1,
        freeze_encoder_epochs: int = 0,
        channels=(32,64,128,256,512),
        strides=(2,2,2,1),
        num_res_units=2,
        norm="BATCH",
        freeze_bn_stats: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model = Unet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=tuple(channels),
            strides=tuple(strides),
            num_res_units=int(num_res_units),
            norm=str(norm),
        )

        # loss
        loss_name = str(loss_name).lower()
        if loss_name == "dicefocal":
            self.loss_fn = DiceFocalLoss(sigmoid=True, lambda_dice=0.5, lambda_focal=0.5, alpha=0.25, gamma=2.0)
        else:
            self.loss_fn = DiceCELoss(sigmoid=True)

        self.val_dice = DiceMetric(include_background=False, reduction="mean")
        self.test_dice = DiceMetric(include_background=False, reduction="mean")

        self._freeze_bn_stats = bool(freeze_bn_stats)
        self._encoder_frozen = freeze_encoder_epochs > 0
        self._freeze_encoder_epochs = int(freeze_encoder_epochs)

        # load pretrained
        if pretrained_ckpt:
            self._load_pretrained_backbone(pretrained_ckpt)

    def _load_pretrained_backbone(self, ckpt_path: str):
        print(f"[INFO] Loading pretrained checkpoint: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)

        # map student_encoder.* -> model.<rest>
        mapped = {}
        for k, v in sd.items():
            if k.startswith("student_encoder."):
                rest = k[len("student_encoder."):]  # usually "model...."
                mapped_key = "model." + rest        # because our module stores Unet under self.model
                mapped[mapped_key] = v

        model_sd = self.state_dict()
        safe, dropped = {}, []
        for k, v in mapped.items():
            if k not in model_sd:
                dropped.append((k, "not_in_model"))
                continue
            if tuple(v.shape) != tuple(model_sd[k].shape):
                dropped.append((k, f"shape {tuple(v.shape)} vs {tuple(model_sd[k].shape)}"))
                continue
            safe[k] = v

        incompat = self.load_state_dict(safe, strict=False)
        print(f"[INFO] Pretrained load: kept={len(safe)} dropped={len(dropped)} missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}", flush=True)
        if dropped:
            print("[INFO] Example dropped keys:", flush=True)
            for kk, why in dropped[:10]:
                print(f"  {kk} -> {why}", flush=True)

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        # Freeze BN running stats for tiny-batch finetuning (usually helps a lot)
        if self._freeze_bn_stats:
            bn = 0
            for m in self.model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
                    bn += 1
            print(f"[INFO] Froze BatchNorm layers: {bn}", flush=True)

    def on_train_epoch_start(self):
        if self._encoder_frozen and self.current_epoch < self._freeze_encoder_epochs:
            # freeze everything except final conv-ish head (MONAI UNet last block is under model.2.* often)
            for name, p in self.model.named_parameters():
                if not name.startswith("model.2."):
                    p.requires_grad = False
        elif self._encoder_frozen and self.current_epoch >= self._freeze_encoder_epochs:
            print(f"[INFO] Unfreezing encoder at epoch={self.current_epoch}", flush=True)
            for _, p in self.model.named_parameters():
                p.requires_grad = True
            self._encoder_frozen = False

    def training_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"].float()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=x.shape[0])
        return loss

    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"].float()
        logits = self(x)
        loss = self.loss_fn(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=x.shape[0])

        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()
        self.val_dice(pred, y)
        return loss

    def on_validation_epoch_end(self):
        d = self.val_dice.aggregate().item()
        self.val_dice.reset()
        self.log("val_dice_050", float(d), prog_bar=True)

    def test_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"].float()
        logits = self(x)
        probs = torch.sigmoid(logits)
        pred = (probs > 0.5).float()
        self.test_dice(pred, y)

    def on_test_epoch_end(self):
        d = self.test_dice.aggregate().item()
        self.test_dice.reset()
        self.log("test_dice_050", float(d), prog_bar=True)

    def configure_optimizers(self):
        base_lr = float(self.hparams.lr)
        wd = float(self.hparams.weight_decay)
        enc_mult = float(self.hparams.encoder_lr_mult)

        backbone, head = [], []
        for name, p in self.model.named_parameters():
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

        # simple cosine schedule (good sweep default)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, self.trainer.max_epochs))
        return {"optimizer": opt, "lr_scheduler": sched}

# -------------------------
# Main
# -------------------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", type=str, required=True)
    ap.add_argument("--pretrained_ckpt", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--datatypes", type=str, default=",".join(DATATYPES_DEFAULT))
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--train_frac", type=float, default=0.7)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--test_frac", type=float, default=0.15)

    # training
    ap.add_argument("--max_epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--loss_name", type=str, choices=["dicece", "dicefocal"], default="dicece")
    ap.add_argument("--encoder_lr_mult", type=float, default=0.1)
    ap.add_argument("--freeze_encoder_epochs", type=int, default=0)
    ap.add_argument("--freeze_bn_stats", type=int, default=1)
    ap.add_argument("--early_stopping_patience", type=int, default=30)
    ap.add_argument("--min_epochs", type=int, default=0)  # optional, useful if you sweep patience


    # model (must match pretrain)
    ap.add_argument("--unet_channels", type=str, default="32,64,128,256,512")
    ap.add_argument("--unet_strides", type=str, default="2,2,2,1")
    ap.add_argument("--unet_num_res_units", type=int, default=2)
    ap.add_argument("--unet_norm", type=str, default="BATCH")

    # wandb
    ap.add_argument("--wandb_project", type=str, default="selma3d_finetune_unet")
    ap.add_argument("--wandb_run_name", type=str, default=None)
    ap.add_argument("--wandb_tags", type=str, default="")

    return ap.parse_args()

def main():
    args = parse_args()
    seed_all(args.seed)

    data_root = Path(args.data_root)
    base_out_dir = Path(args.out_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    datatypes = [x.strip() for x in args.datatypes.split(",") if x.strip()]
    pairs = discover_pairs_by_datatype(data_root, datatypes)

    if len(pairs) == 0:
        raise RuntimeError(f"No (image,label) pairs found under {data_root}")

    train_items, val_items, test_items = stratified_split(
        pairs, seed=args.seed, train_frac=args.train_frac, val_frac=args.val_frac, test_frac=args.test_frac
    )

    print(f"[INFO] Total pairs={len(pairs)} | train={len(train_items)} val={len(val_items)} test={len(test_items)}", flush=True)
    counts = {}
    for split_name, items in [("train", train_items), ("val", val_items), ("test", test_items)]:
        c = {}
        for it in items: c[it["dtype"]] = c.get(it["dtype"], 0) + 1
        counts[split_name] = c
    print("[INFO] Split counts:", json.dumps(counts, indent=2), flush=True)

    # datasets + loaders
    train_tf = get_finetune_train_transforms()
    val_tf = get_finetune_val_transforms()

    train_ds = NiftiPairDictDataset(train_items, transform=train_tf)
    val_ds   = NiftiPairDictDataset(val_items, transform=val_tf)
    test_ds  = NiftiPairDictDataset(test_items, transform=val_tf)

    loader_kw = dict(
        num_workers=min(args.num_workers, os.cpu_count() or args.num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader   = DataLoader(val_ds, batch_size=1, shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds, batch_size=1, shuffle=False, **loader_kw)

    # wandb logger
    tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
    wandb_logger = WandbLogger(project=args.wandb_project, name=args.wandb_run_name, tags=tags)

    # ---- IMPORTANT: make run-specific output directory ----
    # In sweeps, WANDB_RUN_ID env var substitution is unreliable. Instead, use the actual run id
    # created by the WandbLogger, then write outputs to: <base_out_dir>/<run_id>/
    run_id = getattr(wandb_logger.experiment, "id", None) or (wandb.run.id if wandb.run is not None else None)
    if run_id is None:
        # very defensive fallback (should not happen)
        run_id = "no_wandb_run_id"
    out_dir = base_out_dir / str(run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] W&B run_id={run_id}", flush=True)
    print(f"[INFO] Writing outputs to {out_dir}", flush=True)

    # model
    channels = tuple(int(x) for x in args.unet_channels.split(","))
    strides  = tuple(int(x) for x in args.unet_strides.split(","))

    model = FinetuneUNet(
        pretrained_ckpt=args.pretrained_ckpt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_name=args.loss_name,
        encoder_lr_mult=args.encoder_lr_mult,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        channels=channels,
        strides=strides,
        num_res_units=args.unet_num_res_units,
        norm=args.unet_norm,
        freeze_bn_stats=bool(args.freeze_bn_stats),
    )

    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            dirpath=str(ckpt_dir),
            filename="best",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
        ),
        EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=int(args.early_stopping_patience),
        ),
    ]


    trainer = pl.Trainer(
        max_epochs=int(args.max_epochs),
        min_epochs=int(args.min_epochs),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
        deterministic=True,
)


    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, dataloaders=test_loader, ckpt_path="best")

    print(f"[INFO] Done. Outputs in {out_dir}", flush=True)

if __name__ == "__main__":
    main()
