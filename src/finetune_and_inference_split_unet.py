#!/usr/bin/env python3
"""
finetune_and_inference_split_unet.py

Run finetuning + inference for ONE subtype (or multiple) with:
- Optional CV folds JSON (train/eval lists produced by get_selma_cross_val_folds.py)
- Train-pool size K comes from folds JSON; then split train-pool into finetune train/val
- Train MONAI UNet, early stopping on val_loss (min)
- Run inference on held-out eval/test set, save preds + metrics

Designed to match your old finetune_and_inference_split.py structure.
"""

import argparse
import csv
import gc
from dataclasses import dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import random
import time
import wandb

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# your finetune transforms
import sys
sys.path.append("/home/ads4015/ssl_project/src")
from all_datasets_transforms import get_finetune_train_transforms, get_finetune_val_transforms

from monai.networks.nets import Unet
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric

torch.set_float32_matmul_precision("medium")

PRETTY_SUBTYPE_MAP = {
    "amyloid_plaque_patches": "amyloid_plaque",
    "c_fos_positive_patches": "c_fos_positive",
    "cell_nucleus_patches": "cell_nucleus",
    "vessels_patches": "vessels",
}


# -------------------------
# utils
# -------------------------

def _seed_everything(seed: int):
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _seed_worker(_):
    info = get_worker_info()
    if info is not None:
        base_seed = torch.initial_seed() % 2**31
        random.seed(base_seed + info.id)
        np.random.seed(base_seed + info.id)

def _format_hms(seconds):
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}s"

def list_available_subtypes(root: Path):
    return sorted([d.name for d in root.iterdir() if d.is_dir()])

@dataclass
class Pair:
    image: Path
    label: Path

def discover_pairs(class_dir: Path, channel_substr="ch0"):
    # normalize channel filters
    substrings = None
    if channel_substr:
        s = str(channel_substr).strip()
        if s and s.upper() != "ALL":
            substrings = [t.strip().lower() for t in s.split(",") if t.strip()]

    pairs = []
    for p in sorted(class_dir.glob("*.nii*")):
        lower = p.name.lower()
        if lower.endswith("_label.nii") or lower.endswith("_label.nii.gz"):
            continue
        if substrings is not None and not any(sub in lower for sub in substrings):
            continue

        suffix = "".join(p.suffixes)
        base = p.name[:-len(suffix)]
        label = p.with_name(f"{base}_label{suffix}")
        if label.exists():
            pairs.append(Pair(image=p, label=label))
    return pairs

def split_train_val(pairs, val_percent=0.2, val_count=None, seed=100, min_train=1, min_val=1):
    pairs = list(pairs)
    n = len(pairs)
    if n < (min_train + min_val):
        return [], []

    rng = random.Random(seed + 1)
    rng.shuffle(pairs)

    if val_count is not None:
        n_val = int(val_count)
    else:
        val_percent = min(max(float(val_percent), 0.0), 1.0)
        n_val = int(round(n * val_percent))

    n_val = max(min_val, min(n_val, n - min_train))
    n_train = n - n_val
    if n_train < min_train or n_val < min_val:
        return [], []

    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    return train_pairs, val_pairs

@torch.no_grad()
def predict_logits(model, x):
    model.eval()
    device = next(model.parameters()).device
    return model(x.to(device))

def dice_at_threshold_from_logits(logits, target, threshold=0.5, eps=1e-8):
    # logits/target shapes: (1,1,D,H,W)
    if target.device != logits.device:
        target = target.to(logits.device)
    target = target.to(dtype=logits.dtype)

    probs = torch.sigmoid(logits)
    pred = (probs >= threshold).to(dtype=logits.dtype)
    inter = (pred * target).sum()
    denom = pred.sum() + target.sum() + eps
    return float((2.0 * inter / denom).item())

def save_pred_nii(mask_bin, like_path, out_path):
    vol = mask_bin.squeeze().detach().cpu().numpy().astype(np.uint8)
    try:
        like = nib.load(str(like_path))
        affine, header = like.affine, like.header
    except Exception:
        affine, header = np.eye(4), nib.Nifti1Header()
    nib.save(nib.Nifti1Image(vol, affine, header), str(out_path))

def save_prob_nii(probs, like_path, out_path):
    # probs: expected shape (1, 1, D, H, W) or broadcastable
    vol = probs.squeeze().detach().cpu().numpy().astype(np.float32)
    try:
        like = nib.load(str(like_path))
        affine, header = like.affine, like.header
    except Exception:
        affine, header = np.eye(4), nib.Nifti1Header()
    nib.save(nib.Nifti1Image(vol, affine, header), str(out_path))


# -------------------------
# dataset
# -------------------------

class NiftiPairDictDataset(Dataset):
    """
    Loads nifti volumes to numpy, returns dict for MONAI dict transforms.
    Output includes:
      - image: (1,D,H,W) float32
      - label: (1,D,H,W) float32 binary
      - filename: str (image path)
    """
    def __init__(self, pairs, transform=None):
        self.pairs = list(pairs)
        self.transform = transform

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        p = self.pairs[idx]
        img = nib.load(str(p.image)).get_fdata().astype(np.float32)
        lbl = nib.load(str(p.label)).get_fdata().astype(np.float32)

        if img.ndim == 4 and img.shape[-1] == 1:
            img = img[..., 0]
        if lbl.ndim == 4 and lbl.shape[-1] == 1:
            lbl = lbl[..., 0]

        img = img[None, ...]
        lbl = (lbl[None, ...] > 0.5).astype(np.float32)

        d = {"image": img, "label": lbl, "filename": str(p.image)}
        if self.transform is not None:
            d = self.transform(d)
        return d


# -------------------------
# Lightning module (UNet)
# -------------------------

class FinetuneUNetModule(pl.LightningModule):
    def __init__(
        self,
        pretrained_ckpt: str,
        lr: float,
        weight_decay: float,
        loss_name: str,
        encoder_lr_mult: float,
        freeze_encoder_epochs: int,
        freeze_bn_stats: int,
        channels,
        strides,
        num_res_units: int,
        norm: str,
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

        ln = str(loss_name).lower()
        if ln == "dicefocal":
            self.loss_fn = DiceFocalLoss(
                sigmoid=True, lambda_dice=0.5, lambda_focal=0.5, alpha=0.25, gamma=2.0
            )
        else:
            self.loss_fn = DiceCELoss(sigmoid=True)

        self._freeze_encoder_epochs = int(freeze_encoder_epochs)
        self._encoder_frozen = self._freeze_encoder_epochs > 0
        self._freeze_bn_stats = bool(int(freeze_bn_stats))

        self.val_dice = DiceMetric(include_background=False, reduction="mean")

        # ---- W&B per-epoch val sample table state ----
        self._val_table = None
        self._val_logged = 0
        self._val_max_log = 5  # up to 5 val samples per epoch

        # load pretrained weights (student_encoder.* -> model.<rest>)
        if pretrained_ckpt is not None and str(pretrained_ckpt).strip() != "":
            self._load_pretrained(pretrained_ckpt)
        else:
            print("[INFO] Initializing UNet from scratch (no pretrained checkpoint).", flush=True)

    def _load_pretrained(self, ckpt_path: str):
        print(f"[INFO] Loading pretrained checkpoint: {ckpt_path}", flush=True)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("state_dict", ckpt)

        mapped = {}
        for k, v in sd.items():
            if k.startswith("student_encoder."):
                rest = k[len("student_encoder."):]
                mapped_key = "model." + rest
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
        print(
            f"[INFO] Pretrained load: kept={len(safe)} dropped={len(dropped)} "
            f"missing={len(incompat.missing_keys)} unexpected={len(incompat.unexpected_keys)}",
            flush=True,
        )

    def forward(self, x):
        return self.model(x)

    def on_train_start(self):
        if self._freeze_bn_stats:
            bn = 0
            for m in self.model.modules():
                if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                    m.eval()
                    bn += 1
            print(f"[INFO] Froze BatchNorm layers: {bn}", flush=True)

    def on_validation_epoch_start(self):
        # fresh table each epoch so we can log unique keys: val_samples_epoch_{epoch}
        self._val_logged = 0
        if isinstance(getattr(self, "logger", None), WandbLogger):
            self._val_table = wandb.Table(columns=["filename", "image_mid", "label_mid", "pred_mid"])
        else:
            self._val_table = None

    def on_train_epoch_start(self):
        # same heuristic you used: treat model.2.* as "head"
        if self._encoder_frozen and self.current_epoch < self._freeze_encoder_epochs:
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

        # Dice@0.5 on thresholded prediction
        probs = torch.sigmoid(logits)
        pred = (probs >= 0.5).float()
        self.val_dice(pred, y)


        # ---- log up to 5 val samples per epoch (thresholded pred) ----
        if self._val_table is not None and self._val_logged < self._val_max_log:
            # batch_size is 1 for your val_loader, but keep robust anyway
            img = x[0, 0].detach().float().cpu().numpy()     # (D,H,W)
            lbl = y[0, 0].detach().float().cpu().numpy()     # (D,H,W)
            # thresholded prediction (NOT probabilities)
            probs = torch.sigmoid(logits[0, 0]).detach().float().cpu().numpy()
            pred_bin = (probs > 0.5).astype(np.float32)

            mid = img.shape[0] // 2
            img2d = img[mid]
            lbl2d = lbl[mid]
            pred2d = pred_bin[mid]

            # filename column
            fname = ""
            if isinstance(batch.get("filename", None), (list, tuple)) and len(batch["filename"]) > 0:
                fname = str(batch["filename"][0])
            elif isinstance(batch.get("filename", None), str):
                fname = batch["filename"]
            self._val_table.add_data(
                fname,
                wandb.Image(img2d),
                wandb.Image(lbl2d),
                wandb.Image(pred2d),
            )
            self._val_logged += 1

        return loss
    
    def on_validation_epoch_end(self):

        # log epoch-level val dice
        d = self.val_dice.aggregate().item()
        self.val_dice.reset()
        self.log("val_dice_050", float(d), on_step=False, on_epoch=True, prog_bar=True)

        # log the table under a unique key per epoch
        if (
            self._val_table is not None
            and self._val_logged > 0
            and isinstance(getattr(self, "logger", None), WandbLogger)
        ):
            key = f"val_samples_epoch_{self.current_epoch}"
            self.logger.experiment.log({key: self._val_table}, commit=False)
        self._val_table = None

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
        return opt


# -------------------------
# run per subtype
# -------------------------

@dataclass
class RunOutputs:
    best_ckpt: str
    metrics_csv: Path
    preds_dir: Path

def run_for_subtype(subtype_dir: Path, args, device) -> RunOutputs:
    subtype = subtype_dir.name

    all_pairs = discover_pairs(subtype_dir, channel_substr=args.channel_substr)

    # folds mode: use fold train/eval lists
    use_folds = (args.folds_json is not None and args.fold_id is not None)
    if use_folds:
        with open(args.folds_json, "r") as f:
            folds = json.load(f)
        entry = folds.get(subtype, {})
        fold_list = entry.get("folds", [])
        if not fold_list or args.fold_id < 0 or args.fold_id >= len(fold_list):
            raise ValueError(f"Invalid fold_id {args.fold_id} for subtype {subtype}")

        fold = fold_list[args.fold_id]
        train_set = set(map(str, fold.get("train", [])))
        eval_set = set(map(str, fold.get("eval", [])))
        _map = {str(p.image): p for p in all_pairs}
        train_pairs = [_map[s] for s in train_set if s in _map]
        eval_pairs = [_map[s] for s in eval_set if s in _map]

        # fold generator already capped to K; still honor optional train_limit defensively
        if args.train_limit is not None and int(args.train_limit) >= 0:
            train_pairs = train_pairs[: min(len(train_pairs), int(args.train_limit))]
    else:
        raise ValueError("This script is intended for the per-K folds_json workflow. Provide --folds_json and --fold_id.")

    print(f"[INFO] {subtype}: Found {len(all_pairs)} pairs -> pool={len(train_pairs)} train_pool, test={len(eval_pairs)}", flush=True)
    if len(train_pairs) == 0 or len(eval_pairs) == 0:
        print(f"[WARN] {subtype}: skipping (empty train_pool or test)", flush=True)
        return RunOutputs(best_ckpt="", metrics_csv=Path(""), preds_dir=Path(""))

    # split train_pool into finetune train/val
    train_core, val_pairs = split_train_val(
        train_pairs,
        val_percent=args.val_percent,
        val_count=args.val_count,
        seed=args.seed,
        min_train=args.min_finetune_train,
        min_val=args.min_finetune_val,
    )
    if len(train_core) < args.min_finetune_train or len(val_pairs) < args.min_finetune_val:
        print(f"[WARN] {subtype}: skipping (after split train={len(train_core)} val={len(val_pairs)})", flush=True)
        return RunOutputs(best_ckpt="", metrics_csv=Path(""), preds_dir=Path(""))

    # datasets/loaders
    train_tf = get_finetune_train_transforms()
    val_tf = get_finetune_val_transforms()

    num_workers = min(args.num_workers, os.cpu_count() or args.num_workers)
    loader_kw = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        worker_init_fn=_seed_worker,
    )

    train_loader = DataLoader(NiftiPairDictDataset(train_core, transform=train_tf),
                              batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(NiftiPairDictDataset(val_pairs, transform=val_tf),
                            batch_size=1, shuffle=False, **loader_kw)
    test_loader = DataLoader(NiftiPairDictDataset(eval_pairs, transform=val_tf),
                             batch_size=1, shuffle=False, **loader_kw)

    # --- pretty subtype + tag naming (match your convention) ---
    pretty = PRETTY_SUBTYPE_MAP.get(subtype, subtype)
    ntr = len(train_pairs)          # train_pool size (train+val pool)
    nev = len(eval_pairs)           # held-out test size
    fttr = len(train_core)          # finetune train count
    ftval = len(val_pairs)          # finetune val count
    K = int(args.train_limit) if args.train_limit is not None else ntr
    FID = int(args.fold_id)

    # EXACT tag format you requested
    tag = (
        f"cvfold{FID}_ntr{ntr}_nev{nev}_fttr{fttr}_ftval{ftval}_"
        f"fold{FID}_trlim{K}_seed{args.seed}"
    )
    init_tag = "scratch" if (args.pretrained_ckpt is None or str(args.pretrained_ckpt).strip() == "") else "pretrained"
    run_name = f"{pretty}_{tag}_{init_tag}"

    wandb_logger = WandbLogger(project=args.wandb_project, name=run_name) if args.wandb_project else None

    # model
    model = FinetuneUNetModule(
        pretrained_ckpt=args.pretrained_ckpt,
        lr=args.lr,
        weight_decay=args.weight_decay,
        loss_name=args.loss_name,
        encoder_lr_mult=args.encoder_lr_mult,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        freeze_bn_stats=args.freeze_bn_stats,
        channels=tuple(int(x) for x in args.unet_channels.split(",")),
        strides=tuple(int(x) for x in args.unet_strides.split(",")),
        num_res_units=args.unet_num_res_units,
        norm=args.unet_norm,
    )

    # outputs
    # checkpoints/<pretty>/<TAG>/
    ckpt_dir = Path(args.ckpt_dir) / pretty / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_ckpt = ModelCheckpoint(
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        dirpath=str(ckpt_dir),
        filename="finetune_unet_best",
    )
    early_stopping = EarlyStopping(
        monitor="val_loss",
        mode="min",
        patience=int(args.early_stopping_patience),
    )

    trainer = pl.Trainer(
        max_epochs=int(args.max_epochs),
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        logger=wandb_logger,
        callbacks=[model_ckpt, early_stopping],
        log_every_n_steps=10,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)

    best_ckpt = model_ckpt.best_model_path
    last_ckpt = str(ckpt_dir / "last.ckpt")
    if not os.path.exists(last_ckpt):
        trainer.save_checkpoint(last_ckpt)

    if not best_ckpt or not os.path.exists(best_ckpt):
        best_ckpt = last_ckpt
        print(f"[WARN] {subtype}: best ckpt missing, using last: {best_ckpt}", flush=True)

    infer_ckpt = best_ckpt if args.infer_ckpt == "best" else last_ckpt
    print(f"[INFO] {subtype}: infer_ckpt={infer_ckpt}", flush=True)

    # inference model
    infer_model = FinetuneUNetModule.load_from_checkpoint(infer_ckpt).to(device).eval()

    # preds/<pretty>/<TAG>/preds/
    preds_dir = Path(args.preds_root) / pretty / tag / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for batch in test_loader:
        x = batch["image"]  # (1,1,D,H,W)
        y = batch["label"]  # (1,1,D,H,W)
        fname = Path(batch["filename"][0])

        logits = predict_logits(infer_model, x)
        dice_050 = dice_at_threshold_from_logits(logits, y, threshold=0.5)

        probs = torch.sigmoid(logits)
        mask_bin = (probs >= 0.5).to(torch.uint8)

        base_stem = fname.stem.replace(".nii", "").replace(".gz", "")
        pred_path = preds_dir / f"{base_stem}_pred_{tag}.nii.gz"
        prob_path = preds_dir / f"{base_stem}_prob_{tag}.nii.gz"
        save_pred_nii(mask_bin, like_path=fname, out_path=pred_path)
        save_prob_nii(probs, like_path=fname, out_path=prob_path)

        rows.append({
            "subtype": subtype,
            "filename": fname.name,
            "image_path": str(fname),
            "dice_050": f"{dice_050:.6f}",
            "pred_path": str(pred_path),
            "prob_path": str(prob_path),
        })
        print(f"[INFO] {subtype}: {fname.name} Dice@0.5={dice_050:.6f}", flush=True)

    metrics_csv = preds_dir / f"metrics_test_{tag}.csv"
    if rows:
        mean_dice = float(np.mean([float(r["dice_050"]) for r in rows]))
        rows.append({
            "subtype": subtype,
            "filename": "MEAN",
            "image_path": "",
            "dice_050": f"{mean_dice:.6f}",
            "pred_path": "",
            "prob_path": "",
        })
        if wandb_logger:
            wandb_logger.experiment.summary[f"{subtype}/{tag}/test_mean_dice_050"] = mean_dice
        print(f"[INFO] {subtype}: mean Dice@0.5={mean_dice:.6f}", flush=True)

    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["subtype", "filename", "image_path", "dice_050", "pred_path", "prob_path"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # cleanup
    del train_loader, val_loader, test_loader
    gc.collect()

    return RunOutputs(best_ckpt=best_ckpt, metrics_csv=metrics_csv, preds_dir=preds_dir)


# -------------------------
# CLI / main
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # data
    p.add_argument("--root", type=str, required=True)
    p.add_argument("--subtypes", nargs="*", default=["ALL"])
    p.add_argument("--exclude_subtypes", nargs="*", default=[])
    p.add_argument("--channel_substr", type=str, default="ALL")

    # folds
    p.add_argument("--folds_json", type=str, default=None)
    p.add_argument("--fold_id", type=int, default=None)
    p.add_argument("--train_limit", type=int, default=None)

    # finetune train/val split
    p.add_argument("--val_percent", type=float, default=0.2)
    p.add_argument("--val_count", type=int, default=None)
    p.add_argument("--min_finetune_train", type=int, default=1)
    p.add_argument("--min_finetune_val", type=int, default=1)

    # training (defaults set to your requested values)
    p.add_argument("--init", type=str, choices=["pretrained", "random"], default="pretrained", help="Initialize from pretrained checkpoint or random weights.")
    p.add_argument("--pretrained_ckpt", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.0003)
    p.add_argument("--weight_decay", type=float, default=0.00001)
    p.add_argument("--max_epochs", type=int, default=600)
    p.add_argument("--early_stopping_patience", type=int, default=200)
    p.add_argument("--freeze_encoder_epochs", type=int, default=10)
    p.add_argument("--encoder_lr_mult", type=float, default=0.2)
    p.add_argument("--freeze_bn_stats", type=int, default=1)
    p.add_argument("--loss_name", type=str, choices=["dicece", "dicefocal"], default="dicefocal")
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=100)

    # unet arch
    p.add_argument("--unet_channels", type=str, default="32,64,128,256,512")
    p.add_argument("--unet_strides", type=str, default="2,2,2,1")
    p.add_argument("--unet_num_res_units", type=int, default=2)
    p.add_argument("--unet_norm", type=str, default="BATCH")

    # outputs/logging
    p.add_argument("--wandb_project", type=str, default="selma3d_unet_ft_infer")
    p.add_argument("--out_root", type=str, required=True, help="Root output dir containing subfolders: checkpoints/, preds/, logs/, cv_folds/")
    p.add_argument("--infer_ckpt", type=str, choices=["best", "last"], default="best")

    return p.parse_args()

def main():
    args = parse_args()
    _seed_everything(args.seed)

    if args.init == "pretrained":
        if args.pretrained_ckpt is None or str(args.pretrained_ckpt).strip() == "":
            raise ValueError("--init pretrained requires --pretrained_ckpt to be set")
    else:
        # scratch init: ignore ckpt even if user passes it
        args.pretrained_ckpt = None

    # Expand out_root and derive ckpt_dir + preds_root
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_root / "preds").mkdir(parents=True, exist_ok=True)
    (out_root / "logs").mkdir(parents=True, exist_ok=True)
    (out_root / "cv_folds").mkdir(parents=True, exist_ok=True)

    # set derived dirs
    args.ckpt_dir = str(out_root / "checkpoints")
    args.preds_root = str(out_root / "preds")

    root = Path(args.root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.perf_counter()
    print(f"[INFO] Using device: {device}", flush=True)
    print(f"[INFO] Start: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}", flush=True)
    print(f"[INFO] Root: {root}", flush=True)

    if any(s.upper() == "ALL" for s in args.subtypes):
        selected = list_available_subtypes(root)
        if args.exclude_subtypes:
            selected = [s for s in selected if s not in args.exclude_subtypes]
    else:
        selected = args.subtypes

    print(f"[INFO] Subtypes: {selected}", flush=True)

    for subtype in selected:
        subdir = root / subtype
        if not subdir.exists():
            print(f"[WARN] Missing subtype dir: {subdir}", flush=True)
            continue
        out = run_for_subtype(subdir, args, device)
        if out.best_ckpt:
            print(f"[INFO] {subtype}: best_ckpt={out.best_ckpt} metrics={out.metrics_csv} preds={out.preds_dir}", flush=True)

    dt = time.perf_counter() - t0
    print(f"[INFO] Finished. Runtime: {_format_hms(dt)} ({dt:.2f}s)", flush=True)

if __name__ == "__main__":
    main()








