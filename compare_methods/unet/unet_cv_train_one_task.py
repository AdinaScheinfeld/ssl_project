#!/usr/bin/env python3
"""unet_train_eval_from_folds_task.py

Runs ONE experiment task:
- one datatype (e.g., cell_nucleus_patches)
- one pool size K (train_limit) coming from the folds JSON
- one fold id (0..repeats-1)

It reads a folds JSON produced by your existing:
  /home/ads4015/ssl_project/src/get_selma_cross_val_folds.py

That folds JSON contains (per datatype) a list of folds with keys:
  - fold['train']: list of image paths (length = train_limit)
  - fold['eval'] : list of image paths (length = test_size, you use 2)

IMPORTANT: Your folds generator calls the held-out set "eval".
For this UNet experiment we interpret:
  - fold['eval']  -> TEST set (always 2 images)
  - fold['train'] -> TRAIN+VAL pool (size K)
And then we do an *internal* 80/20 split of the K pool into finetune train/val,
with the constraint: at least 1 train and at least 1 val.

Outputs (exactly like you requested):
OUT_ROOT/
  logs/<datatype>/<run_id>/
  checkpoints/<datatype>/<run_id>/best.pt
  preds/<datatype>/<run_id>/*.nii.gz
  splits/<datatype>/<run_id>/split.json
  metrics_<datatype>_<run_id>.csv

W&B:
- logs curves/*, system/* each epoch
- logs media/val_examples/epoch_### as a table with mid-slices
- logs final test dice

Example call (the array job will do this):
  python -u unet_train_eval_from_folds_task.py \
    --data_root /.../selma3d_finetune_patches \
    --out_root  /.../finetuned_cross_val \
    --fold_json /.../cv_folds/cell_nucleus_patches_folds_tr10_rep3.json \
    --datatype  cell_nucleus_patches \
    --fold      0 \
    --pool_n    10 \
    --seed      100

"""

import os
import re
import json
import time
import random
import argparse
import resource
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import nibabel as nib
import pandas as pd

import torch
from torch.utils.data import DataLoader

from monai.transforms import Compose, NormalizeIntensityd, EnsureTyped
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

import wandb


# -------------------------
# Filename conventions
# -------------------------
# Matches non-label files like patch_003_vol007_ch0.nii.gz
PATCH_RE = re.compile(r"^(patch_\d+_vol\d+)_ch([01])\.nii\.gz$")


# -------------------------
# Reproducibility helpers
# -------------------------

def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for determinism."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic convs / cudnn (slower but reproducible)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# -------------------------
# NIfTI IO helpers
# -------------------------

def load_nifti_array(path: str) -> np.ndarray:
    """Load NIfTI as float32 numpy array."""
    return nib.load(path).get_fdata(dtype=np.float32)


def make_input_from_path(image_path: str) -> np.ndarray:
    """Return input tensor as numpy with shape [1, H, W, D]."""
    x = load_nifti_array(image_path).astype(np.float32)
    return x[None, ...]


def make_label_from_path(label_path: str) -> np.ndarray:
    """Return binary label tensor as numpy with shape [1, H, W, D]."""
    y = load_nifti_array(label_path)
    y = (y > 0.5).astype(np.float32)
    return y[None, ...]


def save_pred_nifti(pred: np.ndarray, ref_path: str, out_path: str) -> None:
    """Save prediction NIfTI using affine/header from ref_path."""
    ref = nib.load(ref_path)
    img = nib.Nifti1Image(pred.astype(np.float32), affine=ref.affine, header=ref.header)
    nib.save(img, out_path)


# -------------------------
# MONAI transforms
# -------------------------

def make_transforms():
    """Basic transforms (same as your baseline): type + intensity normalize."""
    return Compose(
        [
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
            # normalize only nonzero voxels, channel-wise (still OK for 1 channel)
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ]
    )


# -------------------------
# Dataset
# -------------------------

class NiftiSingleChannelDataset(torch.utils.data.Dataset):
    """Dataset where each sample is a single-channel image + binary label."""

    def __init__(self, items: List[Dict], xform=None):
        self.items = items
        self.xform = xform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        x = make_input_from_path(it["image"])  # [1,...]
        y = make_label_from_path(it["label"])  # [1,...]
        data = {"image": x, "label": y, "id": it["id"]}
        if self.xform is not None:
            data = self.xform(data)
        return data


# -------------------------
# Inference helper
# -------------------------

@torch.no_grad()
def infer_batch(model, x, roi_size, sw_batch_size=1):
    """Sliding window inference so we can handle large 3D volumes."""
    return sliding_window_inference(
        x, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model
    )


def _get_batch_id(batch, i: int) -> str:
    """Safely get the i-th sample id from a batch (PyTorch collate friendly)."""
    bid = batch.get("id", None)
    if bid is None:
        return f"sample_{i}"
    if isinstance(bid, (list, tuple)):
        return str(bid[i])
    if isinstance(bid, torch.Tensor):
        return str(bid[i].item())
    return str(bid)


# -------------------------
# System metrics for W&B
# -------------------------

def _sys_metrics(step_time_sec: float, batch_size_for_throughput: int) -> Dict[str, float]:
    """CPU/GPU memory and throughput per epoch."""
    out = {"system/step_time_sec": float(step_time_sec)}

    # CPU RSS (Linux ru_maxrss is KB)
    try:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        out["system/cpu_mem_rss_mb"] = float(rss_kb) / 1024.0
    except Exception:
        pass

    # GPU memory
    if torch.cuda.is_available():
        out["system/gpu_mem_alloc_mb"] = float(torch.cuda.memory_allocated() / (1024**2))
        out["system/gpu_mem_reserved_mb"] = float(torch.cuda.memory_reserved() / (1024**2))

    # throughput
    if step_time_sec > 0:
        out["system/throughput_samples_per_sec"] = float(batch_size_for_throughput) / float(step_time_sec)

    return out


# -------------------------
# Dice calculation (per-sample, binary)
# -------------------------

def dice_binary(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    """Binary dice on numpy arrays."""
    pred = (pred > 0.5).astype(np.float32)
    gt = (gt > 0.5).astype(np.float32)
    inter = float((pred * gt).sum())
    denom = float(pred.sum() + gt.sum())
    return float((2.0 * inter + eps) / (denom + eps))


# -------------------------
# Label pairing logic (image -> label)
# -------------------------

def image_to_label_path(img_path: str) -> str:
    """Given /.../patch_###_vol###_chX.nii.gz, pick label path with fallback.

    Preference order:
      1) matching channel label: *_chX_label.nii.gz
      2) fallback: *_ch0_label.nii.gz
      3) fallback: *_ch1_label.nii.gz

    Raises if none exists.
    """
    p = Path(img_path)
    m = PATCH_RE.match(p.name)
    if not m:
        raise ValueError(f"Unexpected filename: {p.name}")

    base, ch = m.group(1), m.group(2)
    d = p.parent

    cand_match = d / f"{base}_ch{ch}_label.nii.gz"
    cand0 = d / f"{base}_ch0_label.nii.gz"
    cand1 = d / f"{base}_ch1_label.nii.gz"

    if cand_match.exists():
        return str(cand_match)
    if cand0.exists():
        return str(cand0)
    if cand1.exists():
        return str(cand1)

    raise FileNotFoundError(f"No label found for {img_path}")


def paths_to_items(img_paths: List[str]) -> List[Dict]:
    """Convert image paths to item dicts your Dataset expects."""
    items: List[Dict] = []
    for ip in img_paths:
        p = Path(ip)
        m = PATCH_RE.match(p.name)
        if not m:
            continue
        base, ch = m.group(1), m.group(2)
        sid = f"{base}_ch{ch}"
        lp = image_to_label_path(ip)
        items.append({"id": sid, "image": ip, "label": lp})
    if len(items) != len(img_paths):
        # This is fine if some paths were weird, but usually should match.
        # We print a warning so you notice.
        print(f"[WARN] Some image paths did not match PATCH_RE. kept={len(items)} requested={len(img_paths)}", flush=True)
    return items


# -------------------------
# Core: run a single task
# -------------------------

def run_one_task(args) -> None:
    """Run one (datatype, K, fold_id) task."""

    seed_everything(args.seed)

    data_root = Path(args.data_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # Load folds json produced by get_selma_cross_val_folds.py
    fold_json = Path(args.fold_json)
    with open(fold_json, "r") as f:
        obj = json.load(f)

    if args.datatype not in obj:
        raise KeyError(
            f"datatype '{args.datatype}' not found in {fold_json}. Keys={list(obj.keys())}"
        )

    folds = obj[args.datatype]["folds"]
    if not (0 <= args.fold < len(folds)):
        raise ValueError(f"fold must be in [0,{len(folds)-1}] but got {args.fold}")

    fold = folds[args.fold]

    # fold['eval'] is your held-out set (we treat as TEST)
    test_img_paths = list(fold["eval"])

    # fold['train'] is the pool of size K (we split into finetune train/val)
    pool_img_paths = list(fold["train"])

    # Sanity checks: must match user requirement
    if len(test_img_paths) != 2:
        print(f"[WARN] test set size is {len(test_img_paths)} (expected 2). continuing anyway.")

    if args.pool_n is not None and args.pool_n > 0 and len(pool_img_paths) != args.pool_n:
        print(
            f"[WARN] pool_n arg={args.pool_n} but fold train list has {len(pool_img_paths)} paths. Using fold list.",
            flush=True,
        )

    # Split pool into ft train/val using 80/20, but enforce >=1 each.
    # Deterministic given seed + fold.
    rng = random.Random(args.seed + 1000 * args.fold)
    pool_shuf = pool_img_paths[:]
    rng.shuffle(pool_shuf)

    pool_n = len(pool_shuf)
    if pool_n < 2:
        raise ValueError(f"Pool must be >=2 to have 1 train and 1 val, got {pool_n}")

    n_val = int(round(pool_n * 0.2))
    n_val = max(1, n_val)
    n_val = min(pool_n - 1, n_val)
    n_train = pool_n - n_val

    ft_train_paths = pool_shuf[:n_train]
    ft_val_paths = pool_shuf[n_train:]

    # Convert to dataset items (image path -> image+label dict)
    train_items = paths_to_items(ft_train_paths)
    val_items = paths_to_items(ft_val_paths)
    test_items = paths_to_items(test_img_paths)

    # Naming like your example
    datatype = args.datatype
    ntr = 2
    nev = 2
    fttr = len(train_items)
    ftval = len(val_items)

    run_id = (
        f"cvfold{args.fold}_ntr{ntr}_nev{nev}_fttr{fttr}_ftval{ftval}_"
        f"fold{args.fold}_trlim{pool_n}_seed{args.seed}"
    )

    # Output directories (datatype/run_id)
    out_logs = out_root / "logs" / datatype / run_id
    out_ckpt = out_root / "checkpoints" / datatype / run_id
    out_preds = out_root / "preds" / datatype / run_id
    out_splits = out_root / "splits" / datatype / run_id

    for p in [out_logs, out_ckpt, out_preds, out_splits]:
        p.mkdir(parents=True, exist_ok=True)

    # Save split json (ids only, similar to your baseline)
    split_obj = {
        "train": [x["id"] for x in train_items],
        "val": [x["id"] for x in val_items],
        "test": [x["id"] for x in test_items],
        "train_image_paths": ft_train_paths,
        "val_image_paths": ft_val_paths,
        "test_image_paths": test_img_paths,
        "fold_json": str(fold_json),
        "fold_index": args.fold,
        "pool_n": pool_n,
    }
    with open(out_splits / "split.json", "w") as f:
        json.dump(split_obj, f, indent=2)

    # Data loaders
    xform = make_transforms()
    train_ds = NiftiSingleChannelDataset(train_items, xform=xform)
    val_ds = NiftiSingleChannelDataset(val_items, xform=xform)
    test_ds = NiftiSingleChannelDataset(test_items, xform=xform)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Model (same as your baseline)
    model = UNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
        dropout=args.dropout,
    ).to(device)

    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # -------------------------
    # W&B init
    # -------------------------
    if args.wandb_mode:
        os.environ["WANDB_MODE"] = args.wandb_mode  # online|offline

    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        name=f"{datatype}_{run_id}",
        config={
            **vars(args),
            "datatype": datatype,
            "run_id": run_id,
            "pool_n": pool_n,
            "fttr": fttr,
            "ftval": ftval,
            "ntr": ntr,
            "nev": nev,
            "fold_json": str(fold_json),
        },
        dir=str(out_logs),
        reinit=True,
    )

    wandb.save(str(out_splits / "split.json"))

    # Define metrics so W&B plots align nicely
    wandb.define_metric("curves/*", step_metric="curves/epoch")
    wandb.define_metric("system/*", step_metric="system/epoch")
    wandb.define_metric("media/*", step_metric="curves/epoch")

    # -------------------------
    # Train loop
    # -------------------------
    best_val_dice = -1.0
    best_path = out_ckpt / "best.pt"

    patience = int(args.early_stop_patience)
    min_delta = float(args.early_stop_min_delta)
    epochs_no_improve = 0

    roi_size = tuple(int(x) for x in args.roi_size.split(","))

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()

        running_loss = 0.0
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))

        # ---- validation ----
        val_t0 = time.time()
        model.eval()
        val_losses: List[float] = []
        dice_metric.reset()

        # W&B example slices: collect up to max_logged_val_images per epoch
        logged_images = 0
        val_table = wandb.Table(columns=["Filename", "Image", "Label", "Prediction"])
        media_key = f"media/val_examples/epoch_{epoch:03d}"

        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device)
                y = batch["label"].to(device)

                logits = infer_batch(model, x, roi_size=roi_size, sw_batch_size=args.sw_batch_size)
                loss = loss_fn(logits, y)
                val_losses.append(loss.item())

                pred_prob = torch.sigmoid(logits)
                pred_bin = (pred_prob > args.thresh).float()
                dice_metric(y_pred=pred_bin, y=y)

                # log slices
                if logged_images < args.max_logged_val_images:
                    sid = _get_batch_id(batch, 0)

                    x_cpu = batch["image"].detach().cpu().float()[0, 0].numpy()
                    y_cpu = batch["label"].detach().cpu().float()[0].numpy().squeeze()
                    p_cpu = pred_bin.detach().cpu().float()[0, 0].numpy()

                    # Use axis 0 mid-slice (same convention you used)
                    mid = x_cpu.shape[0] // 2

                    val_table.add_data(
                        sid,
                        wandb.Image(x_cpu[mid]),
                        wandb.Image(y_cpu[mid]),
                        wandb.Image(p_cpu[mid]),
                    )
                    logged_images += 1

        val_loss = float(np.mean(val_losses)) if val_losses else float("nan")
        val_dice = float(dice_metric.aggregate().item()) if len(val_items) else float("nan")

        # scheduler step after epoch
        scheduler.step()

        # checkpoint best
        improved = val_dice > (best_val_dice + min_delta)
        if improved:
            best_val_dice = val_dice
            epochs_no_improve = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "best_val_dice": best_val_dice,
                    "run_id": run_id,
                    "args": vars(args),
                },
                best_path,
            )
        else:
            epochs_no_improve += 1

        epoch_time_sec = time.time() - t0
        val_time_sec = time.time() - val_t0

        # log once/epoch
        log_dict: Dict[str, object] = {}
        if logged_images > 0:
            log_dict[media_key] = val_table

        wandb.log(
            {
                **log_dict,
                # curves
                "curves/epoch": epoch,
                "curves/loss_train": train_loss,
                "curves/loss_val": val_loss,
                "curves/dice_val": val_dice,
                "curves/lr": optimizer.param_groups[0]["lr"],
                # best/early stop
                "best/val_dice": best_val_dice,
                "early_stop/epochs_no_improve": epochs_no_improve,
                "early_stop/patience": patience,
                "early_stop/min_delta": min_delta,
                # system
                "system/epoch": epoch,
                "system/epoch_time_sec": float(epoch_time_sec),
                "system/val_time_sec": float(val_time_sec),
                **_sys_metrics(step_time_sec=epoch_time_sec, batch_size_for_throughput=max(1, args.batch_size)),
            },
            step=epoch,
        )

        print(
            f"[{datatype} | {run_id}] epoch={epoch:03d} "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
            f"(best={best_val_dice:.4f})",
            flush=True,
        )

        # early stop
        if patience > 0 and epochs_no_improve >= patience:
            print(
                f"[EARLY STOP] No val_dice improvement for {epochs_no_improve} epochs. Stop at epoch {epoch}.",
                flush=True,
            )
            break

    # -------------------------
    # Load best and run TEST
    # -------------------------
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=device)
        model.load_state_dict(ckpt["model_state"], strict=False)

    model.eval()

    rows = []
    test_dices: List[float] = []

    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            sid = _get_batch_id(batch, 0)

            logits = infer_batch(model, x, roi_size=roi_size, sw_batch_size=1)
            prob = torch.sigmoid(logits)
            pred = (prob > args.thresh).float()

            # per-sample dice
            pred_np = pred[0, 0].detach().cpu().numpy()
            gt_np = y[0, 0].detach().cpu().numpy()
            sample_dice = dice_binary(pred_np, gt_np)
            test_dices.append(sample_dice)

            # save prediction
            ref_label = next(it["label"] for it in test_items if it["id"] == sid)
            out_pred_path = out_preds / f"{sid}_pred_{run_id}.nii.gz"
            save_pred_nifti(pred[0].detach().cpu().numpy(), ref_label, str(out_pred_path))

            rows.append(
                {
                    "datatype": datatype,
                    "run_id": run_id,
                    "cvfold": args.fold,
                    "pool_n": pool_n,
                    "fttr": fttr,
                    "ftval": ftval,
                    "seed": args.seed,
                    "test_id": sid,
                    "pred_path": str(out_pred_path),
                    "test_dice": sample_dice,
                }
            )

    test_dice = float(np.mean(test_dices)) if test_dices else float("nan")
    wandb.log({"test/dice": test_dice, "test/num_samples": len(test_items)})

    # Save per-run metrics CSV at OUT_ROOT/metrics_*.csv
    metrics_csv = out_root / f"metrics_{datatype}_{run_id}.csv"
    pd.DataFrame(rows).to_csv(metrics_csv, index=False)

    # Upload metrics to W&B artifacts list
    wandb.save(str(metrics_csv))

    wandb.finish()


# -------------------------
# CLI
# -------------------------

def parse_args():
    p = argparse.ArgumentParser()

    # where data lives
    p.add_argument(
        "--data_root",
        type=str,
        required=True,
        help="Root containing datatype subfolders (amyloid_plaque_patches, ...).",
    )

    # where outputs go
    p.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output root (/.../compare_methods/unet/finetuned_cross_val)",
    )

    # task identity
    p.add_argument("--fold_json", type=str, required=True, help="Fold json produced by get_selma_cross_val_folds.py")
    p.add_argument("--datatype", type=str, required=True, help="Datatype key inside fold_json")
    p.add_argument("--fold", type=int, required=True, help="Fold index (0..repeats-1)")
    p.add_argument(
        "--pool_n",
        type=int,
        default=-1,
        help="Optional: expected pool size K. Only used for logging/sanity warnings.",
    )

    # training hyperparams (match your baseline)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--thresh", type=float, default=0.5)
    p.add_argument("--roi_size", type=str, default="96,96,96")
    p.add_argument("--sw_batch_size", type=int, default=1)

    # early stopping
    p.add_argument("--early_stop_patience", type=int, default=3)
    p.add_argument("--early_stop_min_delta", type=float, default=0.0)

    # W&B
    p.add_argument("--wandb_project", type=str, default="selma3d_unet_cv")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--wandb_mode", type=str, default="", help="online|offline (optional)")
    p.add_argument("--max_logged_val_images", type=int, default=5)

    return p.parse_args()


def main():
    args = parse_args()
    # pool_n is only sanity; normalize
    if args.pool_n is not None and args.pool_n < 0:
        args.pool_n = None
    run_one_task(args)


if __name__ == "__main__":
    main()
