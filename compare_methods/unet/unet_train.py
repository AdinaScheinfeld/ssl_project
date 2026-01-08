#!/usr/bin/env python3
"""
Train a 3D UNet for light-sheet microscopy patch segmentation (NIfTI .nii.gz) with W&B logging.

UPDATED BEHAVIOR (per your request):
- Each image channel file is treated as a separate sample (in_channels = 1).
  Examples (each is its own sample if it exists):
    patch_000_vol003_ch0.nii.gz  -> sample id: patch_000_vol003_ch0
    patch_000_vol003_ch1.nii.gz  -> sample id: patch_000_vol003_ch1

- Label selection per sample (preferred order):
    1) matching channel label: patch_000_vol003_ch{X}_label.nii.gz
    2) fallback: ch0 label
    3) fallback: ch1 label

Splits:
- Hold out 2 samples for test
- Split remaining into train/val with 80/20

Outputs:
  OUT_ROOT/
    logs/
    checkpoints/
    preds/
    splits/

W&B:
- logs scalar curves (train loss, val loss, val dice, lr)
- logs a table EVERY epoch with up to 5 validation samples:
    image (mid-slice), ground truth (mid-slice), prediction (mid-slice)

Usage:
  python train_unet_selma3d.py --data_dir ... --out_root ...
"""

import os
import re
import glob
import json
import time
import random
import argparse
from pathlib import Path
import resource
from typing import Dict, List, Tuple, Optional
import multiprocessing as mp

import numpy as np
import nibabel as nib

import torch
from torch.utils.data import DataLoader

from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    EnsureTyped,
)
from monai.networks.nets import UNet
from monai.losses import DiceCELoss
from monai.metrics import DiceMetric
from monai.inferers import sliding_window_inference

import wandb


# -------------------------
# Utilities
# -------------------------

# Matches filenames like patch_000_vol003_ch0.nii.gz (not labels)
PATCH_RE = re.compile(r"^(patch_\d+_vol\d+)_ch([01])\.nii\.gz$")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism (can reduce speed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def ensure_dirs(out_root: Path) -> Dict[str, Path]:
    paths = {
        "out_root": out_root,
        "logs": out_root / "logs",
        "checkpoints": out_root / "checkpoints",
        "preds": out_root / "preds",
        "splits": out_root / "splits",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)
    return paths


def load_nifti_array(path: str) -> np.ndarray:
    return nib.load(path).get_fdata(dtype=np.float32)


def make_input_from_path(image_path: str) -> np.ndarray:
    """
    Returns input as [1, ...] float32 (single channel).
    """
    x = load_nifti_array(image_path).astype(np.float32)
    return x[None, ...]


def make_label_from_path(label_path: str) -> np.ndarray:
    """
    Returns label as [1, ...] float32, binarized.
    """
    y = load_nifti_array(label_path)
    y = (y > 0.5).astype(np.float32)
    return y[None, ...]


def find_samples(data_dir: Path) -> List[Dict]:
    """
    One sample per channel image file.

    Each sample dict:
      {
        "id": f"{base}_ch{ch}",
        "base": base,
        "ch": int(ch),
        "image": "/path/to/..._ch{ch}.nii.gz",
        "label": "/path/to/..._ch{ch}_label.nii.gz" (preferred) else fallback
      }
    """
    img_paths = sorted(glob.glob(str(data_dir / "patch_*_vol*_ch[01].nii.gz")))
    img_paths = [p for p in img_paths if not p.endswith("_label.nii.gz")]

    samples: List[Dict] = []
    skipped_no_label = 0

    for p in img_paths:
        fname = os.path.basename(p)
        m = PATCH_RE.match(fname)
        if not m:
            continue
        base, ch = m.group(1), m.group(2)

        # label preference: matching channel label, else fallback
        cand_match = str(data_dir / f"{base}_ch{ch}_label.nii.gz")
        cand0 = str(data_dir / f"{base}_ch0_label.nii.gz")
        cand1 = str(data_dir / f"{base}_ch1_label.nii.gz")

        if os.path.exists(cand_match):
            label = cand_match
        elif os.path.exists(cand0):
            label = cand0
        elif os.path.exists(cand1):
            label = cand1
        else:
            skipped_no_label += 1
            continue

        samples.append(
            {
                "id": f"{base}_ch{ch}",
                "base": base,
                "ch": int(ch),
                "image": p,
                "label": label,
            }
        )

    if len(samples) == 0:
        raise RuntimeError(
            f"No valid image/label samples found in {data_dir}. "
            f"Expected patch_###_vol###_ch0.nii.gz and corresponding *_label.nii.gz."
        )
    if skipped_no_label > 0:
        print(f"[WARN] Skipped {skipped_no_label} images with no label found.", flush=True)

    return samples


def save_split_json(paths: Dict[str, Path], train_ids: List[str], val_ids: List[str], test_ids: List[str]) -> None:
    split = {"train": train_ids, "val": val_ids, "test": test_ids}
    with open(paths["splits"] / "split.json", "w") as f:
        json.dump(split, f, indent=2)


def make_splits(
    samples: List[Dict],
    seed: int,
    test_n: int = 2,
    val_frac: float = 0.2,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Sample-level split:
      - first test_n (after shuffle) -> test
      - remaining split into train/val with val_frac
    NOTE: This can split ch0 and ch1 of the same base across different sets.
    """
    rng = random.Random(seed)
    idx = list(range(len(samples)))
    rng.shuffle(idx)

    test_idx = idx[:test_n]
    remain = idx[test_n:]

    n_val = int(round(len(remain) * val_frac))
    val_idx = remain[:n_val]
    train_idx = remain[n_val:]

    train = [samples[i] for i in train_idx]
    val = [samples[i] for i in val_idx]
    test = [samples[i] for i in test_idx]
    return train, val, test


def make_transforms():
    # You can add augmentations here later (RandFlipd, RandAffined, etc.)
    return Compose(
        [
            EnsureTyped(keys=["image", "label"], dtype=torch.float32),
            # Normalize nonzero voxels channel-wise (for 1 channel, it's still fine)
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
        ]
    )


def tensor_to_mid_slice_img(x: torch.Tensor) -> np.ndarray:
    """
    x: [C, ..., D] assumed depth is last dim.
    Returns 2D numpy image from channel 0, mid depth slice, scaled to [0,1].
    """
    x_np = x.detach().cpu().float().numpy()
    v = x_np[0]
    mid = v.shape[-1] // 2
    img2d = v[..., mid]
    mn, mx = float(np.min(img2d)), float(np.max(img2d))
    if mx > mn:
        img2d = (img2d - mn) / (mx - mn)
    return img2d


def tensor_to_mid_slice_mask(m: torch.Tensor) -> np.ndarray:
    """
    m: [1, ..., D] -> 2D slice (no scaling).
    """
    m_np = m.detach().cpu().float().numpy()[0]
    mid = m_np.shape[-1] // 2
    return m_np[..., mid]


def save_pred_nifti(pred: np.ndarray, ref_path: str, out_path: str) -> None:
    """
    pred: numpy array [1, ...] or [...]
    Uses affine/header from ref_path for convenience.
    """
    ref = nib.load(ref_path)
    affine = ref.affine
    hdr = ref.header
    pred = pred.astype(np.float32)
    img = nib.Nifti1Image(pred, affine=affine, header=hdr)
    nib.save(img, out_path)


def summarize_state_dict_load(model, load_result):
    """
    Print how many parameters were loaded, missing, and unexpected
    from a load_state_dict call.
    """
    model_keys = set(model.state_dict().keys())

    loaded_keys = model_keys - set(load_result.missing_keys)
    n_loaded = sum(model.state_dict()[k].numel() for k in loaded_keys)
    n_missing = len(load_result.missing_keys)
    n_unexpected = len(load_result.unexpected_keys)

    total_params = sum(p.numel() for p in model.parameters())

    print(
        f"[INFO] Model parameters: total={total_params:,} | "
        f"loaded={n_loaded:,} | "
        f"missing_keys={n_missing} | "
        f"unexpected_keys={n_unexpected}",
        flush=True,
    )

    if load_result.missing_keys:
        print("[DEBUG] Missing keys (first 10):", load_result.missing_keys[:10], flush=True)
    if load_result.unexpected_keys:
        print("[DEBUG] Unexpected keys (first 10):", load_result.unexpected_keys[:10], flush=True)


def print_load_state_dict_stats(model, load_result, prefix="[INFO]"):
    """
    Print loaded / missing / unexpected parameter counts after load_state_dict.
    """
    model_sd = model.state_dict()

    loaded_keys = set(model_sd.keys()) - set(load_result.missing_keys)
    n_loaded = sum(model_sd[k].numel() for k in loaded_keys)
    n_missing = len(load_result.missing_keys)
    n_unexpected = len(load_result.unexpected_keys)

    total_params = sum(p.numel() for p in model.parameters())

    print(
        f"{prefix} Checkpoint load stats | "
        f"total_params={total_params:,} | "
        f"loaded_params={n_loaded:,} | "
        f"missing_keys={n_missing} | "
        f"unexpected_keys={n_unexpected}",
        flush=True,
    )

    if load_result.missing_keys:
        print(f"{prefix} Missing keys (first 10): {load_result.missing_keys[:10]}", flush=True)
    if load_result.unexpected_keys:
        print(f"{prefix} Unexpected keys (first 10): {load_result.unexpected_keys[:10]}", flush=True)




# -------------------------
# Dataset
# -------------------------

class NiftiSingleChannelDataset(torch.utils.data.Dataset):
    def __init__(self, items: List[Dict], xform=None):
        self.items = items
        self.xform = xform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        x = make_input_from_path(it["image"])   # [1, ...]
        y = make_label_from_path(it["label"])   # [1, ...]
        data = {"image": x, "label": y, "id": it["id"]}
        if self.xform is not None:
            data = self.xform(data)
        return data


# -------------------------
# Train / Eval
# -------------------------

@torch.no_grad()
def infer_batch(model, x, roi_size, sw_batch_size=1):
    return sliding_window_inference(x, roi_size=roi_size, sw_batch_size=sw_batch_size, predictor=model)


def _get_batch_id(batch, i: int) -> str:
    """
    batch["id"] might be a list/tuple (default PyTorch collate) or a string.
    Return the i-th id as a string.
    """
    bid = batch.get("id", None)
    if bid is None:
        return f"sample_{i}"
    if isinstance(bid, (list, tuple)):
        return str(bid[i])
    if isinstance(bid, torch.Tensor):
        # rare, but handle
        return str(bid[i].item())
    return str(bid)

def train(args):
    seed_everything(args.seed)

    data_dir = Path(args.data_dir)
    out_root = Path(args.out_root)
    paths = ensure_dirs(out_root)

    samples = find_samples(data_dir)
    train_items, val_items, test_items = make_splits(samples, seed=args.seed, test_n=2, val_frac=0.2)

    print(f"[INFO] Found {len(samples)} samples total (one per channel file).", flush=True)
    print(f"[INFO] Split: train={len(train_items)} val={len(val_items)} test={len(test_items)}", flush=True)

    save_split_json(
        paths,
        train_ids=[x["id"] for x in train_items],
        val_ids=[x["id"] for x in val_items],
        test_ids=[x["id"] for x in test_items],
    )

    xform = make_transforms()

    train_ds = NiftiSingleChannelDataset(train_items, xform=xform)
    val_ds = NiftiSingleChannelDataset(val_items, xform=xform)
    test_ds = NiftiSingleChannelDataset(test_items, xform=xform)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )
    test_loader = DataLoader(
        test_ds, batch_size=1, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        persistent_workers=(args.num_workers > 0),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(
        spatial_dims=3,
        in_channels=1,   # <-- single channel
        out_channels=1,
        channels=(16, 32, 64, 128, 256),
        strides=(2, 2, 2, 2),
        num_res_units=2,
        norm="INSTANCE",
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"[INFO] UNet initialized: total_params={total_params:,}, "
        f"trainable_params={trainable_params:,}",
        flush=True,
    )

    loss_fn = DiceCELoss(sigmoid=True, squared_pred=True, reduction="mean")
    dice_metric = DiceMetric(include_background=True, reduction="mean", get_not_nans=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ---- W&B ----
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity if args.wandb_entity else None,
        name=args.run_name if args.run_name else None,
        config=vars(args),
        dir=str(paths["logs"]),
    )
    wandb.save(str(paths["splits"] / "split.json"))

    best_val_dice = -1.0
    best_path = None

    # (optional) help keep W&B panels organized by metric prefixes
    # We'll log curves/*, media/*, system/* keys consistently.

    # ---- Early stopping state ----
    patience = int(args.early_stop_patience)
    min_delta = float(args.early_stop_min_delta)
    epochs_no_improve = 0


    roi_size = tuple(int(x) for x in args.roi_size.split(","))  # e.g. "96,96,96"

    def _sys_metrics(step_time_sec: float, batch_size_for_throughput: int) -> Dict[str, float]:
        out = {
            "system/step_time_sec": float(step_time_sec),
        }
        # CPU RSS (MB)
        try:
            rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            # ru_maxrss is KB on Linux, bytes on some systems; on your cluster it should be KB.
            out["system/cpu_mem_rss_mb"] = float(rss_kb) / 1024.0
        except Exception:
            pass
        # GPU mem (MB)
        if torch.cuda.is_available():
            out["system/gpu_mem_alloc_mb"] = float(torch.cuda.memory_allocated() / (1024**2))
            out["system/gpu_mem_reserved_mb"] = float(torch.cuda.memory_reserved() / (1024**2))
        # Throughput
        if step_time_sec > 0:
            out["system/throughput_samples_per_sec"] = float(batch_size_for_throughput) / float(step_time_sec)
        return out

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()

        running_loss = 0.0
        for batch in train_loader:
            x = batch["image"].to(device)  # [B,1,...]
            y = batch["label"].to(device)  # [B,1,...]

            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))

        val_t0 = time.time()

        # ---- Validation ----
        model.eval()
        val_losses = []
        dice_metric.reset()

        # ---- Lightning-style table lifecycle ----
        # create fresh table each epoch, fill during val, log ONCE at end
        logged_images = 0
        val_table = wandb.Table(columns=["Filename", "Image", "Label", "Prediction"])

        # Log image tables under a dedicated media/* namespace so they're easy to find
        # and don't clutter the generic "Tables" list with val_examples_XX keys.
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

                # ---- "validation_step" image logging (Lightning-style) ----
                if logged_images < 5:
                    bsz = x.shape[0]  # usually 1
                    take = min(5 - logged_images, bsz)

                    x_cpu = batch["image"].detach().to("cpu", dtype=torch.float32)
                    y_cpu = batch["label"].detach().to("cpu", dtype=torch.float32)
                    p_cpu = pred_bin.detach().to("cpu", dtype=torch.float32)

                    for i in range(take):
                        sid = _get_batch_id(batch, i)

                        img_np = x_cpu[i, 0].numpy()          # [H,W,D] or [D,H,W] depending on data
                        lbl_np = y_cpu[i].numpy().squeeze()   # same spatial shape
                        prd_np = p_cpu[i, 0].numpy()
                        mid = img_np.shape[0] // 2            # match your Lightning code (uses axis 0)

                        val_table.add_data(
                            sid,
                            wandb.Image(img_np[mid]),
                            wandb.Image(lbl_np[mid]),
                            wandb.Image(prd_np[mid]),
                        )
                        logged_images += 1

        val_loss = float(np.mean(val_losses)) if len(val_losses) else float("nan")
        val_dice = float(dice_metric.aggregate().item()) if len(val_items) else float("nan")

        # ---- Log EXACTLY ONCE per epoch (images table) ----
        # IMPORTANT: do NOT also log val_examples_{epoch} (it clutters the Tables list).
        if logged_images > 0:
            wandb.log({media_key: val_table}, step=epoch)

        scheduler.step()

        # ---- Checkpoint best ----
        improved = (val_dice > (best_val_dice + min_delta))
        if improved:
            best_val_dice = val_dice
            epochs_no_improve = 0
            best_path = paths["checkpoints"] / "best.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optim_state": optimizer.state_dict(),
                    "best_val_dice": best_val_dice,
                    "args": vars(args),
                },
                best_path,
            )
        else:
            epochs_no_improve += 1

        # ---- W&B scalar logs ----
        epoch_time_sec = time.time() - t0
        val_time_sec = time.time() - val_t0
        wandb.log(
            {
                # Curves (keep in one section)
                "curves/epoch": epoch,
                "curves/loss_train": train_loss,
                "curves/loss_val": val_loss,
                "curves/dice_val": val_dice,
                "curves/lr": optimizer.param_groups[0]["lr"],

                # Early stopping
                "early_stop/epochs_no_improve": epochs_no_improve,
                "early_stop/patience": patience,
                "early_stop/min_delta": min_delta,
                "best/val_dice": best_val_dice,

                # Timing / system
                "system/epoch_time_sec": float(epoch_time_sec),
                "system/val_time_sec": float(val_time_sec),
                **_sys_metrics(step_time_sec=epoch_time_sec, batch_size_for_throughput=max(1, args.batch_size)),
            },
            step=epoch,
        )


        print(
            f"[Epoch {epoch:03d}] "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_dice={val_dice:.4f} "
            f"(best={best_val_dice:.4f})", flush=True
        )

        # ---- Early stopping check ----
        # disable if patience <= 0
        if patience > 0 and epochs_no_improve >= patience:
            print(
                f"[EARLY STOP] No improvement in val_dice for {epochs_no_improve} epochs "
                f"(patience={patience}, min_delta={min_delta}). Stopping at epoch {epoch}.",
                flush=True,
            )
            break

    # ---- Final: run on test set, save preds ----
    if best_path and best_path.exists():
        ckpt = torch.load(best_path, map_location=device)

        load_result = model.load_state_dict(ckpt["model_state"], strict=False)

        print(
            f"[INFO] Loaded best checkpoint from {best_path} "
            f"(best_val_dice={ckpt.get('best_val_dice', float('nan')):.4f})",
            flush=True,
        )

        print_load_state_dict_stats(
            model,
            load_result,
            prefix="[INFO][INFERENCE]",
        )


    model.eval()
    test_rows = []
    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device)
            sid = batch["id"][0]

            logits = infer_batch(model, x, roi_size=roi_size, sw_batch_size=1)
            prob = torch.sigmoid(logits)[0].detach().cpu().numpy()  # [1,...]
            pred = (prob > args.thresh).astype(np.float32)          # [1,...]

            # Save prediction as NIfTI (use label as ref)
            ref_label = next(it["label"] for it in test_items if it["id"] == sid)
            out_pred_path = paths["preds"] / f"{sid}_pred.nii.gz"
            save_pred_nifti(pred, ref_label, str(out_pred_path))

            test_rows.append((sid, float(pred.mean()), str(out_pred_path)))

    wandb.log({"test/num_samples": len(test_items)})

    test_table = wandb.Table(columns=["id", "pred_foreground_frac", "pred_path"])
    for sid, frac, pth in test_rows:
        test_table.add_data(sid, frac, pth)
    wandb.log({"test/preds": test_table})

    print(f"[DONE] Outputs in: {out_root}", flush=True)
    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data_dir",
        type=str,
        required=True,
        help="Folder containing patch_*.nii.gz and *_label.nii.gz",
    )
    p.add_argument(
        "--out_root",
        type=str,
        required=True,
        help="Output root with logs/, checkpoints/, preds/",
    )

    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--thresh", type=float, default=0.5, help="Sigmoid threshold for binarizing predictions")
    p.add_argument("--roi_size", type=str, default="96,96,96", help="Sliding window ROI size, e.g. '96,96,96'")
    p.add_argument("--sw_batch_size", type=int, default=1)

    # Early stopping
    p.add_argument("--early_stop_patience", type=int, default=20,
                   help="Stop if val_dice doesn't improve for this many epochs. Set 0 to disable.")
    p.add_argument("--early_stop_min_delta", type=float, default=0.0,
                   help="Minimum val_dice improvement to reset patience.")

    # W&B
    p.add_argument("--wandb_project", type=str, default="selma3d_unet")
    p.add_argument("--wandb_entity", type=str, default="")
    p.add_argument("--run_name", type=str, default="")

    return p.parse_args()


if __name__ == "__main__":
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    args = parse_args()
    train(args)
