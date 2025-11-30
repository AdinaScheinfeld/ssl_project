#!/usr/bin/env python3
# /home/ads4015/ssl_project/compare_methods/cellseg3d/cellseg3d_finetune_eval_cv.py

"""
Fully fixed + stable cross-validation finetuning script for CellSeg3D WNet.

FEATURES:
    - Restored old working split behavior (automatic directory creation)
    - Notebook-style inference (remote_inference)
    - Correct checkpoint rewriting (model_state_dict only, no module.*)
    - Stable NIfTI loading (ZYX), TIFF conversion, normalization
    - Uses pretrained WNet weights for training by default
"""

import os
import csv
import random
from pathlib import Path
from copy import deepcopy
from collections import OrderedDict

import numpy as np
import nibabel as nib
import tifffile as tiff
import pandas as pd
import torch

from napari_cellseg3d.dev_scripts import colab_training as c
from napari_cellseg3d.dev_scripts import remote_inference as cs3d
from napari_cellseg3d.config import (
    WNetTrainingWorkerConfig,
    WandBConfig,
    WeightsInfo,
    ModelInfo,
)
from napari_cellseg3d.utils import LOGGER as logger

logger.setLevel("INFO")

# ============================================================
# CONSTANT PATHS
# ============================================================
DATA_ROOT = Path("/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches")

# *** SINGLE CLASS ONLY ***
CLASS_NAME = "cell_nucleus_patches"

OUTPUT_ROOT = Path("/midtier/paetzollab/scratch/ads4015/compare_methods/cellseg3d/finetuned_cross_val")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# ============================================================
# TRAINING SETTINGS
# ============================================================
NUM_EPOCHS = 250
VAL_EVERY = 2
LR = 2e-5
BATCH_SIZE = 4
NUM_CLASSES = 2
WEIGHT_DECAY = 0.01

INT_SIGMA = 1.0
SPATIAL_SIGMA = 4.0
NCUT_RADIUS = 2

REC_LOSS = "MSE"
NCUT_WEIGHT = 0.5
REC_WEIGHT = 0.005

BASE_SEED = 100

# ============================================================
# HELPERS
# ============================================================
def is_valid_nii_gz(path: Path):
    return (
        path.is_file()
        and "".join(path.suffixes) == ".nii.gz"
        and "_ch0" in path.name
        and "_label" not in path.name
    )


def load_nifti_as_zyx(path: Path):
    vol = nib.load(str(path)).get_fdata()
    vol = np.squeeze(vol).astype(np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D vol, got {vol.shape} at {path}")
    return np.transpose(vol, (2, 1, 0))  # XYZ → ZYX


def label_path(img_path: Path):
    return img_path.with_name(img_path.name.replace(".nii.gz", "_label.nii.gz"))


# ============================================================
# SPLITTING (RESTORED OLD WORKING BEHAVIOR)
# ============================================================
def make_splits(all_imgs, pool_size, fold_index):
    rng = random.Random(BASE_SEED + fold_index)
    shuffled = all_imgs.copy()
    rng.shuffle(shuffled)

    test_imgs = shuffled[:2]     # exactly 2 test volumes
    remaining = shuffled[2:]

    pool = remaining[:pool_size]

    # 80/20 split with guarantees
    n_train = max(1, int(round(0.8 * pool_size)))
    if n_train >= pool_size:
        n_train = pool_size - 1

    n_val = pool_size - n_train
    train_imgs = pool[:n_train]
    val_imgs = pool[n_train:]

    return train_imgs, val_imgs, test_imgs, n_train, n_val, len(test_imgs)


# ============================================================
# NIFTI → TIFF CONVERSION (FULLY RESTORED)
# ============================================================
def convert_split(name, imgs, out_vol, out_lab):
    """
    Converts NIfTI imgs and labels into TIFFs under vol/ and lab/.
    ALWAYS creates directories first (old behavior).
    """
    rows = []
    out_vol.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    for img in imgs:
        lbl = label_path(img)
        if not lbl.exists():
            raise FileNotFoundError(f"Missing label {lbl}")

        arr = load_nifti_as_zyx(img)
        lab = load_nifti_as_zyx(lbl).astype(np.uint8)

        # Normalize volume
        vmin, vmax = arr.min(), arr.max()
        if vmax > vmin:
            arr = (arr - vmin) / (vmax - vmin)
        else:
            arr = np.zeros_like(arr)

        base = img.name.replace(".nii.gz", "")
        tif_img = out_vol / f"{base}.tif"
        tif_lab = out_lab / f"{base}_label.tif"

        tiff.imwrite(tif_img, arr.astype(np.float32))
        tiff.imwrite(tif_lab, lab)

        rows.append({
            "split": name,
            "nii_image": str(img),
            "nii_label": str(lbl),
            "tif_image": str(tif_img),
            "tif_label": str(tif_lab),
        })

    return rows


# ============================================================
# CHECKPOINT WRAPPER
# ============================================================
def wrap_checkpoint(src_ckpt, dst_ckpt):
    """
    Wrap ANY CellSeg3D or PyTorch checkpoint into a standard:
        {"model_state_dict": cleaned_state_dict}
    This supports:
        - {"state_dict": {...}}
        - {"model_state_dict": {...}}
        - raw OrderedDict (CellSeg3D best-metric format)
    """
    raw = torch.load(src_ckpt, map_location="cpu")

    # Case A: CellSeg3D training wrapper format
    if isinstance(raw, dict) and "state_dict" in raw:
        sd = raw["state_dict"]

    # Case B: Inference wrapper format
    elif isinstance(raw, dict) and "model_state_dict" in raw:
        sd = raw["model_state_dict"]

    # Case C: Raw OrderedDict directly containing weights
    elif isinstance(raw, OrderedDict) or isinstance(raw, dict):
        sd = raw

    else:
        raise ValueError(
            f"Unrecognized checkpoint format (type={type(raw)}): {src_ckpt}"
        )

    # Clean prefixes added by DataParallel or Worker
    new_sd = {}
    for k, v in sd.items():
        nk = k.replace("module.", "").replace("model.", "")
        new_sd[nk] = v

    # Save in standard inference-ready format
    torch.save({"model_state_dict": new_sd}, dst_ckpt)

    print(f"[INFO] Wrapped checkpoint saved to {dst_ckpt}", flush=True)



# ============================================================
# NOTEBOOK-STYLE INFERENCE
# ============================================================
def infer_volume(vol, ckpt):
    cfg = deepcopy(cs3d.CONFIG)

    cfg.model_info = ModelInfo(
        name="WNet3D",
        model_input_size=[64, 64, 64],
        num_classes=2,
    )

    cfg.weights_config = WeightsInfo(
        path=str(ckpt),
        use_custom=True,
        use_pretrained=False,
    )

    print(f"[CellSeg3D] Using FINETUNED custom weights: {ckpt}", flush=True)

    out = cs3d.inference_on_images(vol, config=cfg)[0]
    sem = out.semantic_segmentation

    # For WNet: [C,Z,Y,X] → take nucleus class if needed
    if sem.ndim == 4:
        sem = sem[1]

    inst, _ = cs3d.post_processing(sem, config=cs3d.PostProcessConfig())
    return sem.astype(np.uint8), inst.astype(np.uint16)


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_experiment(pool_size, fold_index):

    # ----------------------------- LOAD DATA -----------------------------
    class_dir = DATA_ROOT / CLASS_NAME
    all_imgs = sorted([f for f in class_dir.iterdir() if is_valid_nii_gz(f)])

    # ----------------------------- SPLIT -----------------------------
    train_imgs, val_imgs, test_imgs, n_train, n_val, n_test = \
        make_splits(all_imgs, pool_size, fold_index)

    # ----------------------------- TEMP DIR -----------------------------
    tmp_root = OUTPUT_ROOT / "tmp" / f"pool{pool_size}" / f"fold{fold_index}"

    train_vol = tmp_root / "train/vol"
    train_lab = tmp_root / "train/lab"
    val_vol   = tmp_root / "val/vol"
    val_lab   = tmp_root / "val/lab"
    test_vol  = tmp_root / "test/vol"
    test_lab  = tmp_root / "test/lab"

    # ----------------------------- CONVERT TO TIFF -----------------------------
    rows = []
    rows += convert_split("train", train_imgs, train_vol, train_lab)
    rows += convert_split("val",   val_imgs,   val_vol,   val_lab)
    rows += convert_split("test",  test_imgs,  test_vol,  test_lab)

    tmp_root.mkdir(parents=True, exist_ok=True)
    with open(tmp_root / "splits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    print("[DEBUG] Train TIFFs:", sorted(p.name for p in train_vol.glob("*.tif")), flush=True)
    print("[DEBUG] Val TIFFs:", sorted(p.name for p in val_vol.glob("*.tif")), flush=True)

    # ----------------------------- TRAIN -----------------------------
    ckpt_dir = tmp_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pretrained = WeightsInfo(use_pretrained=True, use_custom=False)
    print("[INFO] Using PRETRAINED WNet weights for training.", flush=True)

    train_cfg = WNetTrainingWorkerConfig(
        device="cuda:0",
        max_epochs=NUM_EPOCHS,
        learning_rate=LR,
        validation_interval=VAL_EVERY,
        batch_size=BATCH_SIZE,
        num_workers=2,
        weights_info=pretrained,
        results_path_folder=str(ckpt_dir),
        train_data_dict=c.create_dataset_dict_no_labs(train_vol),
        eval_volume_dict=c.create_eval_dataset_dict(val_vol, val_lab),
        num_classes=NUM_CLASSES,
        weight_decay=WEIGHT_DECAY,
        intensity_sigma=INT_SIGMA,
        spatial_sigma=SPATIAL_SIGMA,
        radius=NCUT_RADIUS,
        reconstruction_loss=REC_LOSS,
        n_cuts_weight=NCUT_WEIGHT,
        rec_loss_weight=REC_WEIGHT,
    )

    worker = c.get_colab_worker(train_cfg, WandBConfig(mode="disabled"))

    print("[INFO] Training starts…", flush=True)

    last_logged = 0
    for _ in worker.train():
        epoch = len(worker.total_losses)
        if epoch <= last_logged:
            continue
        last_logged = epoch

        train_loss = worker.total_losses[-1]
        rec_loss   = worker.rec_losses[-1]
        ncut_loss  = worker.ncuts_losses[-1]

        msg = f"[EPOCH {epoch}] train={train_loss:.4f} rec={rec_loss:.4f} ncuts={ncut_loss:.4f}"
        if worker.dice_values:
            msg += f" val_dice={worker.dice_values[-1]:.4f}"
        print(msg, flush=True)

    print("[INFO] Training completed.", flush=True)

    # ----------------------------- CHECKPOINT WRAP -----------------------------
    ckpt_best = ckpt_dir / "wnet_best_metric.pth"
    ckpt_inf  = ckpt_dir / "wnet_best_metric_for_inference.pth"    
    wrap_checkpoint(ckpt_best, ckpt_inf)

    # ----------------------------- INFERENCE -----------------------------
    fold_tag = (
        f"cvfold{fold_index}_ntr{pool_size}_nev{n_test}_"
        f"fttr{n_train}_ftval{n_val}_"
        f"trlim{pool_size}_seed{BASE_SEED}"
    )

    pred_dir = OUTPUT_ROOT / "preds" / CLASS_NAME / fold_tag
    pred_dir.mkdir(parents=True, exist_ok=True)

    metrics = []

    for tif_path in sorted(test_vol.glob("*.tif")):
        vol = tiff.imread(str(tif_path)).astype(np.float32)

        vmin, vmax = vol.min(), vol.max()
        if vmax > vmin:
            vol = (vol - vmin) / (vmax - vmin)

        sem, inst = infer_volume(vol, ckpt_inf)

        base = tif_path.stem
        tiff.imwrite(pred_dir / f"{base}_semantic.tif",  sem)
        tiff.imwrite(pred_dir / f"{base}_instances.tif", inst)

        metrics.append({
            "volume": base,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "pool_size": pool_size,
            "fold_index": fold_index,
        })

    pd.DataFrame(metrics).to_csv(
        pred_dir / f"metrics_{CLASS_NAME}_{fold_tag}.csv",
        index=False,
    )

    print("[INFO] Experiment complete.", flush=True)


# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-size", type=int, required=True)
    parser.add_argument("--fold-index", type=int, required=True)
    args = parser.parse_args()
    run_experiment(args.pool_size, args.fold_index)
