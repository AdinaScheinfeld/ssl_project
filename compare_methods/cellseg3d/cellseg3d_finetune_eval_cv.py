#!/usr/bin/env python3
# /home/ads4015/ssl_project/compare_methods/cellseg3d/cellseg3d_finetune_eval_cv.py
"""
cellseg3d_finetune_cv.py

Full cross‑validation pipeline for finetuning CellSeg3D's WNet3D model
on 3D light‑sheet microscopy patches.

Features:
- Automatically builds train/val/test splits for each pool_size × fold_index
- Converts NIfTI → TIF while keeping CellSeg3D's (Z,Y,X) shape
- Runs WNet3D finetuning using napari_cellseg3d Colab training worker
- Wraps the checkpoint for CellSeg3D inference
- Runs inference on held‑out test volumes
- Saves metrics + predictions using your naming convention
- Outputs to structured folder tree under finetuned_cross_val

You run this script by calling:
    python cellseg3d_finetune_cv.py --pool-size 3 --fold-index 0
"""

import os
import csv
import json
import random
from pathlib import Path
from copy import deepcopy

import numpy as np
import nibabel as nib
import tifffile as tiff
import pandas as pd

import torch
from collections import OrderedDict

# CellSeg3D imports
from napari_cellseg3d.dev_scripts import colab_training as c
from napari_cellseg3d.dev_scripts import remote_inference as cs3d
from napari_cellseg3d.utils import LOGGER as logger
from napari_cellseg3d.config import (
    WNetTrainingWorkerConfig,
    WandBConfig,
    WeightsInfo,
    ModelInfo,
)

logger.setLevel("INFO")

# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------
DATA_ROOT = Path("/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches")
CLASS_NAME = "cell_nucleus_patches"
OUTPUT_ROOT = Path("/midtier/paetzollab/scratch/ads4015/compare_methods/cellseg3d/finetuned_cross_val")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Training hyperparameters
NUM_EPOCHS = 250
LR = 2e-5
BATCH_SIZE = 4
VAL_EVERY = 2
NUM_CLASSES = 2
WEIGHT_DECAY = 0.01
INT_SIGMA = 1.0
SPATIAL_SIGMA = 4.0
NCUT_RADIUS = 2
REC_LOSS = "MSE"
NCW = 0.5
RLW = 0.005
BASE_SEED = 100

# ------------------------------------------------------------
# Helper: load and reshape nifti
# ------------------------------------------------------------
def load_nifti_as_zyx(path: Path):
    nii = nib.load(str(path))
    arr = nii.get_fdata().astype(np.float32)
    arr = np.squeeze(arr)
    if arr.ndim != 3:
        raise ValueError(f"Expected 3D volume at {path}, got {arr.shape}")
    # NIfTI = (X,Y,Z) → TIFF = (Z,Y,X)
    return np.transpose(arr, (2, 1, 0))


def img_to_label_path(img_path: Path) -> Path:
    return img_path.with_name(img_path.name.replace(".nii.gz", "_label.nii.gz"))


# ------------------------------------------------------------
# Split logic (train/val/test)
# ------------------------------------------------------------
def make_splits(all_imgs, pool_size, fold_index):
    rng = random.Random(BASE_SEED + fold_index)
    shuffled = all_imgs.copy()
    rng.shuffle(shuffled)

    test_imgs = shuffled[:2]
    remaining = shuffled[2:]
    pool = remaining[:pool_size]

    n_train = max(1, int(round(0.8 * pool_size)))
    if n_train >= pool_size:
        n_train = pool_size - 1
    n_val = pool_size - n_train

    train_imgs = pool[:n_train]
    val_imgs = pool[n_train:]

    return train_imgs, val_imgs, test_imgs, n_train, n_val, len(test_imgs)


# ------------------------------------------------------------
# Convert NIfTI → TIFF into temp folders
# ------------------------------------------------------------
def convert_split_to_tif(split_name, img_paths, out_vol_dir, out_lab_dir):
    rows = []
    out_vol_dir.mkdir(parents=True, exist_ok=True)
    out_lab_dir.mkdir(parents=True, exist_ok=True)

    for img_path in img_paths:
        lab_path = img_to_label_path(img_path)
        img = load_nifti_as_zyx(img_path).astype(np.float32)

        # normalize image to [0,1]
        vmin, vmax = img.min(), img.max()
        if vmax > vmin:
            img = (img - vmin) / (vmax - vmin)
        else:
            img = np.zeros_like(img, dtype=np.float32)

        lab = load_nifti_as_zyx(lab_path).astype(np.uint8)
        base = img_path.name.replace(".nii.gz", "")

        tif_img = out_vol_dir / f"{base}.tif"
        tif_lab = out_lab_dir / f"{base}_label.tif"

        tiff.imwrite(tif_img, img.astype(np.float32))
        tiff.imwrite(tif_lab, lab)

        rows.append({
            "split": split_name,
            "nii_image": str(img_path),
            "nii_label": str(lab_path),
            "tif_image": str(tif_img),
            "tif_label": str(tif_lab),
        })
    return rows


# ------------------------------------------------------------
# Wrap checkpoint for CellSeg3D inference
# ------------------------------------------------------------
def wrap_checkpoint(src_path: Path, dst_path: Path):
    raw = torch.load(src_path, map_location="cpu")
    new_state = OrderedDict((k.replace("module.", ""), v) for k, v in raw.items())
    wrapped = {"state_dict": new_state}
    torch.save(wrapped, dst_path)
    return dst_path


# ------------------------------------------------------------
# Inference on TIFF volume
# ------------------------------------------------------------
def run_inference(vol_zyx, ckpt_path):
    inference_cfg = deepcopy(cs3d.CONFIG)
    inference_cfg.model_info = ModelInfo(name="WNet3D", model_input_size=[64,64,64], num_classes=2)
    inference_cfg.weights_config = WeightsInfo(path=str(ckpt_path), use_custom=True, use_pretrained=False)

    result_list = cs3d.inference_on_images(vol_zyx, config=inference_cfg)
    semantic = result_list[0].semantic_segmentation
    if semantic.ndim == 4:  # (C,Z,Y,X)
        semantic = semantic[1]
    inst, _ = cs3d.post_processing(semantic, config=cs3d.PostProcessConfig())
    return semantic.astype(np.uint8), inst.astype(np.uint16)


# ------------------------------------------------------------
# Main execution for a single pool_size × fold_index
# ------------------------------------------------------------
def run_experiment(pool_size, fold_index):
    class_dir = DATA_ROOT / CLASS_NAME
    all_imgs = sorted([f for f in class_dir.glob("*.nii.gz") if f.name.endswith("_ch0.nii.gz") and "_label" not in f.name])

    train_imgs, val_imgs, test_imgs, n_train, n_val, n_test = make_splits(all_imgs, pool_size, fold_index)

    # Temporary TIF directories
    tmp_root = OUTPUT_ROOT / "tmp" / f"pool{pool_size}" / f"fold{fold_index}"
    train_vol = tmp_root / "train/vol"
    train_lab = tmp_root / "train/lab"
    val_vol   = tmp_root / "val/vol"
    val_lab   = tmp_root / "val/lab"
    test_vol  = tmp_root / "test/vol"
    test_lab  = tmp_root / "test/lab"

    split_rows = []
    split_rows += convert_split_to_tif("train", train_imgs, train_vol, train_lab)
    split_rows += convert_split_to_tif("val",   val_imgs,   val_vol,   val_lab)
    split_rows += convert_split_to_tif("test",  test_imgs,  test_vol,  test_lab)

    # Save split index
    tmp_root.mkdir(parents=True, exist_ok=True)
    csv_path = tmp_root / "splits.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=split_rows[0].keys())
        w.writeheader()
        w.writerows(split_rows)

    # Train config
    train_cfg = WNetTrainingWorkerConfig(
        device="cuda:0",
        max_epochs=NUM_EPOCHS,
        learning_rate=LR,
        validation_interval=VAL_EVERY,
        batch_size=BATCH_SIZE,
        num_workers=2,
        weights_info=WeightsInfo(),
        results_path_folder=str(tmp_root / "checkpoints"),
        train_data_dict=c.create_dataset_dict_no_labs(train_vol),
        eval_volume_dict=c.create_eval_dataset_dict(val_vol, val_lab),
        num_classes=NUM_CLASSES,
        weight_decay=WEIGHT_DECAY,
        intensity_sigma=INT_SIGMA,
        spatial_sigma=SPATIAL_SIGMA,
        radius=NCUT_RADIUS,
        reconstruction_loss=REC_LOSS,
        n_cuts_weight=NCW,
        rec_loss_weight=RLW,
    )

    wandb_cfg = WandBConfig(mode="disabled")
    checkpoint_dir = tmp_root / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    worker = c.get_colab_worker(worker_config=train_cfg, wandb_config=wandb_cfg)

    # Debugging: ensure training will actually start
    print("[DEBUG] Train TIFF count:", len(list(train_vol.glob('*.tif'))), flush=True)
    print("[DEBUG] Val TIFF count:", len(list(val_vol.glob('*.tif'))), flush=True)
    print("[DEBUG] Val LABEL count:", len(list(val_lab.glob('*.tif'))), flush=True)
    print("[DEBUG] Beginning training NOW...", flush=True)

    for epoch in worker.train():
        print(f"[DEBUG] Epoch {epoch} completed.", flush=True)

    print(f"[DEBUG] Training completed. Locating checkpoint...{tmp_root}", flush=True)

    # Locate checkpoint
    ckpt_src = tmp_root / "checkpoints/wnet_best_metric.pth"
    ckpt_infer = tmp_root / "checkpoints/wnet_best_metric_for_inference.pth"

# ------------------------------------------------------------
# SAFETY CHECK — Did training actually produce the checkpoint?
# ------------------------------------------------------------
    if not ckpt_src.exists():
        raise FileNotFoundError(
            f"WNet3D training did NOT produce a checkpoint.\n"
            f"Expected: {ckpt_src}\n"
            f"Possible causes:\n"
            f" - Validation TIFFs unreadable\n"
            f" - Validation labels empty\n"
            f" - CellSeg3D crashed internally before epoch 1\n"
            f" - TIF dtype/shapes not acceptable\n"
            f" - Windowing (64³) incompatible with data\n"
        )

    wrap_checkpoint(ckpt_src, ckpt_infer)

    # Evaluate on test set
    fold_tag = (
        f"cvfold{fold_index}_ntr{pool_size}_nev{n_test}_fttr{n_train}_ftval{n_val}_"
        f"fold{fold_index}_trlim{pool_size}_seed{BASE_SEED}"
    )

    pred_dir = OUTPUT_ROOT / "preds" / CLASS_NAME / fold_tag
    pred_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    for tif_path in sorted(test_vol.glob("*.tif")):
        vol = tiff.imread(str(tif_path)).astype(np.float32)
        vmin, vmax = vol.min(), vol.max()
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)

        sem, inst = run_inference(vol, ckpt_infer)

        base = tif_path.stem
        out_sem = pred_dir / f"{base}_semantic.tif"
        out_inst = pred_dir / f"{base}_instances.tif"
        tiff.imwrite(out_sem, sem)
        tiff.imwrite(out_inst, inst)

        metrics.append({
            "volume": base,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
        })

    pd.DataFrame(metrics).to_csv(pred_dir / f"metrics_{CLASS_NAME}_{fold_tag}.csv", index=False)


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--pool-size", type=int, required=True)
    p.add_argument("--fold-index", type=int, required=True)
    args = p.parse_args()

    run_experiment(args.pool_size, args.fold_index)
    print("Done.", flush=True)
