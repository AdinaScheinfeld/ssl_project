#!/usr/bin/env python

# /home/ads4015/ssl_project/compare_methods/micro_sam/micro_sam_finetune_cv.py

"""
microsam_finetune_cv.py

Finetune microSAM with the UNETR decoder on selma3d patches, per datatype,
with random splits (train/val/test) for a given pool size and fold index.

- Uses 2 held-out test volumes for this datatype.
- From the remaining volumes, takes a pool of size `pool_size` for train+val.
- Splits that pool 80/20 into train/val, but enforces at least 1 train and 1 val.
- Trains for 500 epochs with early stopping patience 50.
- Saves checkpoint and test predictions under:
    /midtier/paetzollab/scratch/ads4015/compare_methods/micro_sam/finetuned_cross_val/

You call this script with:
    --data-root
    --class-name
    --pool-size
    --fold-index
    --output-root

W&B logging is optional:
    set WANDB_PROJECT and WANDB_ENTITY env vars as usual to enable.
"""

import os
import argparse
from pathlib import Path
from collections import defaultdict

import numpy as np
import nibabel as nib
import pandas as pd
from skimage.measure import label as connected_components
import scipy.ndimage as ndi

import torch
from torch.utils.data import Dataset, DataLoader

import micro_sam.training as sam_training
from micro_sam.automatic_segmentation import (
    get_predictor_and_segmenter,
    automatic_instance_segmentation,
)

# Optional wandb logging (summary + curves after training)
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


# -----------------------------------------------------------------------------
# Helper functions for label transforms (distances etc.)
# -----------------------------------------------------------------------------
def compute_distance_to_center(inst_mask: np.ndarray) -> np.ndarray:
    """
    For each instance ID > 0, compute distance to its center of mass.
    Normalize to [0,1] per instance and invert so that center=1, edges=0.
    Outside objects: 0.
    """
    H, W = inst_mask.shape
    out = np.zeros((H, W), dtype=np.float32)

    instance_ids = np.unique(inst_mask)
    instance_ids = instance_ids[instance_ids != 0]

    yy, xx = np.indices((H, W), dtype=np.float32)

    for iid in instance_ids:
        mask = (inst_mask == iid)
        if not np.any(mask):
            continue

        cy, cx = ndi.center_of_mass(mask.astype(np.float32))
        dist = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

        maxd = dist[mask].max() + 1e-6
        dist_norm = dist / maxd  # 0..1 inside object
        out[mask] = 1.0 - dist_norm[mask]  # 1 at center, 0 at furthest

    return out


def compute_distance_to_boundary(inst_mask: np.ndarray) -> np.ndarray:
    """
    For each instance ID > 0, compute distance transform inside the object.
    Normalize to [0,1] per instance. Outside objects: 0.
    """
    H, W = inst_mask.shape
    out = np.zeros((H, W), dtype=np.float32)

    instance_ids = np.unique(inst_mask)
    instance_ids = instance_ids[instance_ids != 0]

    for iid in instance_ids:
        mask = (inst_mask == iid)
        if not np.any(mask):
            continue

        dist = ndi.distance_transform_edt(mask)
        maxd = dist[mask].max() + 1e-6
        dist_norm = dist / maxd  # 0..1 inside object
        out[mask] = dist_norm[mask]

    return out


# -----------------------------------------------------------------------------
# Dataset: random foreground slice from 3D NIfTI into 2D sample for microSAM
# -----------------------------------------------------------------------------
class Selma2DSliceDataset(Dataset):
    def __init__(self, raw_label_pairs):
        """
        raw_label_pairs: list of (raw_path, label_path)
        Each raw_path is a single-channel NIfTI:
        - e.g. *_ch0.nii.gz or *_ch1.nii.gz
        """
        self.pairs = list(raw_label_pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        raw_path, label_path = self.pairs[idx]
        raw_path = Path(raw_path)
        label_path = Path(label_path)

        # --- load raw and label volumes ---
        raw_vol = nib.load(str(raw_path)).get_fdata().astype(np.float32)
        lab_vol = nib.load(str(label_path)).get_fdata().astype(np.float32)

        assert raw_vol.shape == lab_vol.shape, f"shape mismatch: {raw_vol.shape} vs {lab_vol.shape}"
        Z = raw_vol.shape[0]

        # --- find a slice that contains at least one object ---
        for _ in range(20):
            z = np.random.randint(0, Z)
            lab_slice = lab_vol[z]
            if lab_slice.max() > 0:
                break
        else:
            # fallback: deterministic scan
            found = False
            for z in range(Z):
                lab_slice = lab_vol[z]
                if lab_slice.max() > 0:
                    found = True
                    break
            if not found:
                # no instances in this volume at all; pick a different pair
                return self.__getitem__((idx + 1) % len(self.pairs))

        img_slice = raw_vol[z]
        lab_slice = lab_vol[z]

        # --- instance mask (connected components) ---
        bin_mask = (lab_slice > 0).astype(np.uint8)
        inst_mask = connected_components(bin_mask, connectivity=1).astype(np.int32)

        if inst_mask.max() == 0:
            # no instance after CC, rare corner case
            return self.__getitem__((idx + 1) % len(self.pairs))

        # --- robust image normalization to [0,255] for microSAM ---
        p1 = np.percentile(img_slice, 1)
        p99 = np.percentile(img_slice, 99)
        if p99 > p1:
            img_slice_norm = (img_slice - p1) / (p99 - p1)
        else:
            img_slice_norm = np.zeros_like(img_slice, dtype=np.float32)

        img_slice_norm = np.clip(img_slice_norm, 0.0, 1.0)
        img_slice_uint = (img_slice_norm * 255.0).astype(np.float32)

        # --- label channels ---
        # channel 0: instance ids
        instance_channel = inst_mask.astype(np.float32)

        # foreground mask
        fg = (inst_mask > 0).astype(np.float32)

        # distances
        dist_center = compute_distance_to_center(inst_mask)
        dist_boundary = compute_distance_to_boundary(inst_mask)

        # mask distances and clamp to [0,1]
        dist_center = np.clip(dist_center * fg, 0.0, 1.0)
        dist_boundary = np.clip(dist_boundary * fg, 0.0, 1.0)

        # assemble y: (4, H, W)
        #   y[0] = instance ids (integer-valued)
        #   y[1] = foreground (0..1)
        #   y[2] = center distance (0..1)
        #   y[3] = boundary distance (0..1)
        y_np = np.stack(
            [instance_channel, fg, dist_center, dist_boundary],
            axis=0
        ).astype(np.float32)

        # to torch
        x = torch.from_numpy(img_slice_uint[None, ...])  # (1,H,W), float32 in [0,255]
        y = torch.from_numpy(y_np)                        # (4,H,W)

        return x, y


# -----------------------------------------------------------------------------
# Binary metrics
# -----------------------------------------------------------------------------
def compute_binary_metrics(gt, pred):
    """Compute Dice and IoU for binary masks (numpy arrays of 0/1)."""
    gt = gt.astype(bool)
    pred = pred.astype(bool)

    intersection = np.logical_and(gt, pred).sum()
    union = np.logical_or(gt, pred).sum()
    gt_sum = gt.sum()
    pred_sum = pred.sum()

    denom = gt_sum + pred_sum
    dice = 2.0 * intersection / denom if denom > 0 else np.nan
    iou = intersection / union if union > 0 else np.nan

    return dice, iou


# -----------------------------------------------------------------------------
# AIS helpers (2D slice via automatic_instance_segmentation, stacked over Z)
# -----------------------------------------------------------------------------
def segment_slice_2d(predictor, segmenter, img2d, tile_shape=None, halo=None, verbose=False):
    """Run AIS on a single 2D slice."""
    img2d = img2d.astype(np.float32)

    # normalize to [0,255] similarly to training
    p1 = np.percentile(img2d, 1)
    p99 = np.percentile(img2d, 99)
    if p99 > p1:
        img2d = (img2d - p1) / (p99 - p1)
    else:
        img2d = np.zeros_like(img2d)
    img2d = np.clip(img2d, 0.0, 1.0)
    img2d = (img2d * 255.0).astype(np.float32)

    instances = automatic_instance_segmentation(
        predictor=predictor,
        segmenter=segmenter,
        input_path=img2d,
        ndim=2,
        tile_shape=tile_shape,
        halo=halo,
        verbose=verbose,
    )

    return instances


def segment_volume_slices(predictor, segmenter, vol, tile_shape=None, halo=None, verbose=False):
    """
    vol: numpy array (Z, Y, X)
    returns: instances (Z, Y, X) with per-slice instance labels.
    """
    Z, Y, X = vol.shape
    instances_vol = np.zeros((Z, Y, X), dtype=np.int32)

    for z in range(Z):
        if verbose:
            print(f"Segmenting slice {z+1}/{Z}...")
        img2d = vol[z]
        seg2d = segment_slice_2d(
            predictor,
            segmenter,
            img2d,
            tile_shape=tile_shape,
            halo=halo,
            verbose=False,
        )
        instances_vol[z] = seg2d.astype(np.int32)

    return instances_vol


# -----------------------------------------------------------------------------
# Main experiment logic
# -----------------------------------------------------------------------------
def make_splits_for_class(all_pairs, pool_size, fold_index, rng_seed):
    """
    all_pairs: list[(raw_path, label_path)] for a single class.
    pool_size: number of volumes in train+val.
    fold_index: 0, 1, 2
    rng_seed: base seed for reproducibility.

    Returns:
        train_pairs, val_pairs, test_pairs
    """
    rng = np.random.default_rng(rng_seed + fold_index)

    n_total = len(all_pairs)
    if n_total < pool_size + 2:
        raise ValueError(
            f"Not enough samples for pool_size={pool_size}: "
            f"need at least pool_size + 2 test = {pool_size + 2}, but have {n_total}"
        )

    # Shuffle all volumes
    indices = np.arange(n_total)
    rng.shuffle(indices)
    shuffled = [all_pairs[i] for i in indices]

    # 2 held-out test volumes for this fold
    test_pairs = shuffled[:2]

    # Pool from remaining volumes
    remaining = shuffled[2:]
    pool = remaining[:pool_size]

    # 80/20 split for train/val, but enforce >=1 each
    n_train = int(round(0.8 * pool_size))
    if n_train < 1:
        n_train = 1
    if n_train > pool_size - 1:
        n_train = pool_size - 1
    n_val = pool_size - n_train

    train_pairs = pool[:n_train]
    val_pairs = pool[n_train:]

    assert len(train_pairs) >= 1 and len(val_pairs) >= 1

    return train_pairs, val_pairs, test_pairs


def run_experiment(
    data_root: Path,
    output_root: Path,
    class_name: str,
    pool_size: int,
    fold_index: int,
    n_epochs: int = 500,
    early_stopping: int = 50,
    n_objects_per_batch: int = 5,
):
    """
    Run one finetuning + test-eval experiment for a single datatype (class_name),
    pool_size and fold_index.
    """

    # -------------------------
    # Collect labeled volumes
    # -------------------------
    class_dir = data_root / class_name
    if not class_dir.is_dir():
        raise RuntimeError(f"Class dir not found: {class_dir}")

    all_pairs = []
    for raw_path in sorted(class_dir.glob("*_ch*.nii.gz")):
        if "_label" in raw_path.name:
            continue
        label_path = raw_path.with_name(raw_path.name.replace(".nii.gz", "_label.nii.gz"))
        if label_path.exists():
            all_pairs.append((raw_path, label_path))

    if len(all_pairs) < 4:
        raise RuntimeError(
            f"Class '{class_name}' has too few labeled volumes for this experiment "
            f"(need at least 4, found {len(all_pairs)})."
        )

    print(f"[INFO] {class_name}: found {len(all_pairs)} labeled channel-volumes.")

    # -------------------------
    # Make splits
    # -------------------------
    base_seed = 100  # fixed
    train_pairs, val_pairs, test_pairs = make_splits_for_class(
        all_pairs, pool_size, fold_index, rng_seed=base_seed
    )

    # number of train/val/test volumes for naming and folder structure
    n_train = len(train_pairs)
    n_val = len(val_pairs)
    n_test = len(test_pairs)

    print(f"[INFO] Splits for class={class_name}, pool={pool_size}, fold={fold_index}:")
    print(f"  Train: {n_train}  Val: {n_val}  Test: {n_test}")

    # -------------------------
    # Datasets & loaders
    # -------------------------
    train_dataset = Selma2DSliceDataset(train_pairs)
    val_dataset = Selma2DSliceDataset(val_pairs)

    batch_size = 1  # keep small for microSAM

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # microSAM's JointSamTrainer expects these attributes
    train_loader.shuffle = True
    val_loader.shuffle = False

    print("[INFO] Train samples:", len(train_dataset), "Val samples:", len(val_dataset))

    # -------------------------
    # microSAM training
    # -------------------------
    model_type = "vit_l_lm"  # microSAM LM model
    exp_name = f"{class_name}_pool{pool_size}_fold{fold_index}"

    # microSAM saves to save_root/checkpoints/<name>/...
    checkpoints_root = output_root / "checkpoints" / class_name
    checkpoints_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("[INFO] Using device:", device)
    print("[INFO] Experiment name:", exp_name)
    print("[INFO] Checkpoints root:", checkpoints_root)

    # Optional W&B run
    wandb_run = None
    if HAS_WANDB and os.environ.get("WANDB_PROJECT"):
        wandb_run = wandb.init(
            project=os.environ.get("WANDB_PROJECT"),
            entity=os.environ.get("WANDB_ENTITY", None),
            name=exp_name,
            config={
                "class_name": class_name,
                "pool_size": pool_size,
                "fold_index": fold_index,
                "model_type": model_type,
                "n_epochs": n_epochs,
                "early_stopping": early_stopping,
                "n_objects_per_batch": n_objects_per_batch,
                "n_train_volumes": n_train,
                "n_val_volumes": n_val,
                "n_test_volumes": n_test,
            },
        )

    sam_training.train_sam(
        name=exp_name,
        save_root=str(checkpoints_root),
        model_type=model_type,
        train_loader=train_loader,
        val_loader=val_loader,
        n_epochs=n_epochs,
        early_stopping=early_stopping,
        n_objects_per_batch=n_objects_per_batch,
        with_segmentation_decoder=True,
        device=device,
    )

    # -------------------------
    # Load best checkpoint
    # -------------------------
    best_checkpoint = checkpoints_root / "checkpoints" / exp_name / "best.pt"
    print("[INFO] Best checkpoint:", best_checkpoint, "exists:", best_checkpoint.exists())
    if not best_checkpoint.exists():
        raise RuntimeError(f"Best checkpoint not found at {best_checkpoint}")

    predictor, segmenter = get_predictor_and_segmenter(
        model_type=model_type,
        checkpoint=str(best_checkpoint),
        device=device,
        is_tiled=False,
    )

    # -------------------------
    # Evaluate on held-out test set & save predictions
    # Directory layout:
    #   preds/<class_name>/cvfold{fold}_ntr{pool}_nev{n_test}_fttr{n_train}_ftval{n_val}_fold{fold}_trlim{pool}_seed{base_seed}/patches/*.nii.gz
    # Example:
    #   cvfold0_ntr2_nev2_fttr1_ftval1_fold0_trlim2_seed100
    # -------------------------
    fold_tag = (
        f"cvfold{fold_index}"
        f"_ntr{pool_size}"
        f"_nev{n_test}"
        f"_fttr{n_train}"
        f"_ftval{n_val}"
        f"_fold{fold_index}"
        f"_trlim{pool_size}"
        f"_seed{base_seed}"
    )

    fold_dir = (
        output_root
        / "preds"
        / class_name
        / fold_tag
        / "patches"
    )
    fold_dir.mkdir(parents=True, exist_ok=True)


    results = []  # per-volume metrics

    eval_tile_shape = None
    eval_halo = None

    for raw_path, label_path in test_pairs:
        raw_path = Path(raw_path)
        label_path = Path(label_path)

        print(f"[EVAL] {raw_path.name}")

        raw_vol = nib.load(str(raw_path)).get_fdata().astype(np.float32)
        lab_vol = nib.load(str(label_path)).get_fdata().astype(np.float32)

        instances = segment_volume_slices(
            predictor,
            segmenter,
            raw_vol,
            tile_shape=eval_tile_shape,
            halo=eval_halo,
            verbose=False,
        )

        # Save prediction with detailed name
        raw_stem = raw_path.name  # e.g. "patch_007_vol005_ch0.nii.gz"
        if raw_stem.endswith(".nii.gz"):
            raw_stem = raw_stem[:-7]  # -> "patch_007_vol005_ch0"

        # Filename:
        #   patch_007_vol005_ch0_pred_cvfold0_ntr2_nev2_fttr1_ftval1_fold0_trlim2_seed100.nii.gz
        pred_fname = f"{raw_stem}_pred_{fold_tag}.nii.gz"
        pred_path = fold_dir / pred_fname


        pred_nii = nib.Nifti1Image(
            instances.astype(np.int32),
            affine=nib.load(str(raw_path)).affine,
        )
        nib.save(pred_nii, pred_path)
        print(f"  Saved prediction to: {pred_path}")

        # Compute slice-wise metrics
        Z = raw_vol.shape[0]
        dice_list = []
        iou_list = []

        for z in range(Z):
            gt_slice = lab_vol[z]
            if gt_slice.max() == 0:
                continue

            pred_slice = instances[z]
            gt_bin = (gt_slice > 0).astype(np.uint8)
            pred_bin = (pred_slice > 0).astype(np.uint8)

            dice, iou = compute_binary_metrics(gt_bin, pred_bin)
            if not np.isnan(dice):
                dice_list.append(dice)
            if not np.isnan(iou):
                iou_list.append(iou)

        if len(dice_list) == 0:
            mean_dice = np.nan
            mean_iou = np.nan
        else:
            mean_dice = float(np.mean(dice_list))
            mean_iou = float(np.mean(iou_list))

        results.append(
            {
                "class": class_name,
                "pool_size": pool_size,
                "fold_index": fold_index,
                "n_train_volumes": n_train,
                "n_val_volumes": n_val,
                "n_test_volumes": n_test,
                "raw_path": str(raw_path),
                "label_path": str(label_path),
                "pred_path": str(pred_path),
                "mean_dice": mean_dice,
                "mean_iou": mean_iou,
                "n_slices_with_fg": len(dice_list),
            }
        )

    # Save metrics next to the prediction patches
    df = pd.DataFrame(results)
    metrics_path = fold_dir / f"metrics_{class_name}_{fold_tag}.csv"
    df.to_csv(metrics_path, index=False)
    print("[INFO] Saved metrics to:", metrics_path)

    # Log summary metrics to W&B (if enabled)
    if wandb_run is not None:
        overall = df[["mean_dice", "mean_iou"]].mean()
        wandb.log(
            {
                "test_mean_dice_macro": float(overall["mean_dice"]),
                "test_mean_iou_macro": float(overall["mean_iou"]),
            }
        )
        wandb.log({"test_volume_metrics": wandb.Table(dataframe=df)})
        wandb_run.finish()



def parse_args():
    p = argparse.ArgumentParser(description="Finetune microSAM on selma3d patches (cross-val).")
    p.add_argument(
        "--data-root",
        type=str,
        default="/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches",
        help="Root directory with class subfolders.",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default="/midtier/paetzollab/scratch/ads4015/compare_methods/micro_sam/finetuned_cross_val",
        help="Root directory for checkpoints, preds, and metrics.",
    )
    p.add_argument(
        "--class-name",
        type=str,
        required=True,
        help="Datatype / class folder name (e.g. amyloid_plaque_patches).",
    )
    p.add_argument(
        "--pool-size",
        type=int,
        required=True,
        help="Number of volumes in train+val pool (after holding out 2 test volumes).",
    )
    p.add_argument(
        "--fold-index",
        type=int,
        required=True,
        help="Fold index (0, 1, or 2).",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Max number of training epochs.",
    )
    p.add_argument(
        "--early-stopping",
        type=int,
        default=50,
        help="Early stopping patience (in epochs without improvement).",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    data_root = Path(args.data_root)
    output_root = Path(args.output_root)

    print("[INFO] Data root:", data_root)
    print("[INFO] Output root:", output_root)
    print("[INFO] Class:", args.class_name, "pool_size:", args.pool_size, "fold:", args.fold_index)

    run_experiment(
        data_root=data_root,
        output_root=output_root,
        class_name=args.class_name,
        pool_size=args.pool_size,
        fold_index=args.fold_index,
        n_epochs=args.epochs,
        early_stopping=args.early_stopping,
    )
