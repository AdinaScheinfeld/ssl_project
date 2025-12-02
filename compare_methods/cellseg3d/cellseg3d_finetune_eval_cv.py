#!/usr/bin/env python3

# /home/ads4015/ssl_project/compare_methods/cellseg3d/cellseg3d_finetune_eval_cv.py

"""
FINAL CellSeg3D Finetune + Eval Script (Option A)
- Original splitting behavior
- NIfTI → TIFF conversion
- Finetune WNet
- Wrap checkpoint correctly
- Inference EXACTLY like working notebook
- AUTO mode for weights:
      custom → pretrained → random
"""

import argparse
import os
import csv
import random
from pathlib import Path
from copy import deepcopy

import numpy as np
import nibabel as nib
import tifffile as tiff
import pandas as pd
import torch
from collections import OrderedDict

# --- CellSeg3D imports ---
from napari_cellseg3d.dev_scripts import colab_training as c
from napari_cellseg3d.dev_scripts import remote_inference as cs3d
from napari_cellseg3d.config import (
    WNetTrainingWorkerConfig,
    WandBConfig,
    WeightsInfo,
    ModelInfo,
    InferenceWorkerConfig,
)
from napari_cellseg3d.utils import LOGGER as logger
from napari_cellseg3d.code_models.models.wnet.model import WNet

logger.setLevel("INFO")

# ============================================================
# USER PATHS
# ============================================================
DATA_ROOT = Path("/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches")
CLASS_NAME = "cell_nucleus_patches"

OUTPUT_ROOT = Path("/midtier/paetzollab/scratch/ads4015/compare_methods/cellseg3d/finetuned_cross_val")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Pretrained WNet shipped with your installation
PRETRAINED_WNET = (
    "/home/ads4015/micromamba/envs/cellseg3d-env1/lib/python3.10/"
    "site-packages/napari_cellseg3d/code_models/models/pretrained/wnet_latest.pth"
)

# ============================================================
# TRAINING PARAMS
# ============================================================
NUM_EPOCHS = 250
VAL_EVERY = 1
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
    vol = nib.load(str(path)).get_fdata().astype(np.float32)
    vol = np.squeeze(vol)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D, got {vol.shape}")
    return np.transpose(vol, (2, 1, 0))  # XYZ → ZYX


def label_path(img_path: Path):
    return img_path.with_name(img_path.name.replace(".nii.gz", "_label.nii.gz"))


# ============================================================
# ORIGINAL SPLIT LOGIC
# ============================================================
def make_splits(all_imgs, pool_size, fold_index):
    
    # Use SEED that depends on pool_size AND fold_index.
    # This guarantees every (pool_size, fold_index) pair is unique.
    seed = BASE_SEED + (pool_size * 1000) + fold_index
    rng = random.Random(seed)

    # Shuffle copies
    shuffled = all_imgs.copy()
    rng.shuffle(shuffled)

    # Exactly 2 test images
    test_imgs = shuffled[:2]
    remaining = shuffled[2:]

    pool = remaining[:pool_size]

    # Train/val (80/20, at least 1 each)
    n_train = max(1, int(round(0.8 * pool_size)))
    if n_train >= pool_size:
        n_train = pool_size - 1
    n_val = pool_size - n_train

    train_imgs = pool[:n_train]
    val_imgs = pool[n_train:]

    return train_imgs, val_imgs, test_imgs, n_train, n_val, len(test_imgs)


# ============================================================
# NIFTI → TIFF CONVERSION
# ============================================================
def convert_split(name, imgs, out_vol, out_lab):
    rows = []
    out_vol.mkdir(parents=True, exist_ok=True)
    out_lab.mkdir(parents=True, exist_ok=True)

    for img in imgs:
        lbl = label_path(img)
        if not lbl.exists():
            raise FileNotFoundError(f"Missing label {lbl}")

        arr = load_nifti_as_zyx(img)
        lab = load_nifti_as_zyx(lbl).astype(np.uint8)

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
# WRAP CHECKPOINT INTO INFERENCE FORMAT
# ============================================================
def wrap_checkpoint(src_ckpt, dst_ckpt):
    raw = torch.load(src_ckpt, map_location="cpu")

    if "model_state_dict" in raw:
        sd = raw["model_state_dict"]
    else:
        sd = raw

    torch.save(sd, dst_ckpt)   # <-- SAVE ONLY THE RAW DICT, NO WRAPPER

    print(f"[INFO] Wrapped checkpoint → {dst_ckpt}")



# ============================================================
# INFERENCE (Notebook-style)
# ============================================================
def infer_volume(vol, ckpt_path):
    """
    Uses CellSeg3D inference API.
    AUTO mode:
        1. Finetuned (if exists)
        2. Pretrained WNet
        3. Random
    Returns (sem, inst, src)
    """

    # Use the full default config (same as notebook)
    cfg = deepcopy(cs3d.CONFIG)

    # Override model_info but KEEP window inference enabled
    cfg.model_info = ModelInfo(
        name="WNet3D",
        model_input_size=[96,96,96],
        num_classes=2,
    )

    # Ensure sliding window is on
    cfg.sliding_inference = True
    cfg.window_size = [64,64,64]
    cfg.window_overlap = 0.25


    # Choose weights
    if ckpt_path and ckpt_path.exists():
        print(f"[AUTO] Using FINETUNED checkpoint: {ckpt_path}")
        cfg.weights_config = WeightsInfo(
            use_custom=True,
            use_pretrained=False,
            path=str(ckpt_path),
        )
        src = "custom"

        # EXTRA SAFETY: completely disable pretrained fallback
        cfg.weights_config.use_pretrained = False
        cfg.weights_config.use_custom = True
        cfg.weights_config.path = str(ckpt_path)
        print(f"[DEBUG] FORCING FINETUNED CHECKPOINT: {ckpt_path}")


        # ============================================================
        # DEBUG: Verify checkpoint matches WNet model architecture
        # ============================================================
        print("\n[DEBUG] --- Verifying checkpoint → WNet compatibility (inference) ---")

        debug_model = WNet(
            in_channels=1,
            out_channels=1,
            num_classes=2,
            dropout=0.65,
        )

        # Load checkpoint raw dict
        raw_ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in raw_ckpt:
            raw_sd = raw_ckpt["model_state_dict"]
        else:
            raw_sd = raw_ckpt

        # Try loading with strict=False to inspect mismatches
        load_result = debug_model.load_state_dict(raw_sd, strict=False)

        print("[DEBUG] Missing keys (model expected but NOT in checkpoint):")
        for k in load_result.missing_keys:
            print("   -", k)

        print("\n[DEBUG] Unexpected keys (checkpoint has but model does NOT):")
        for k in load_result.unexpected_keys:
            print("   -", k)

        print("\n[DEBUG] Total checkpoint params:", len(raw_sd))
        print("[DEBUG] Total model params:", len(debug_model.state_dict()))
        print("[DEBUG] Number of matched keys:",
            len(raw_sd) - len(load_result.unexpected_keys))

        print("[DEBUG] --- END inference weight-load debug ---\n")
  

    elif Path(PRETRAINED_WNET).exists():
        print(f"[AUTO] Finetuned missing → Using PRETRAINED: {PRETRAINED_WNET}")
        cfg.weights_config = WeightsInfo(
            use_custom=False,
            use_pretrained=True,
            path=str(PRETRAINED_WNET),
        )
        src = "pretrained"

    else:
        print("[AUTO] WARNING: No weights found → RANDOM initialization")
        cfg.weights_config = WeightsInfo(
            use_custom=False,
            use_pretrained=False,
            path=None,
        )
        src = "random"

    v = vol.astype(float)
    vmin, vmax = v.min(), v.max()
    v = (v - vmin) / (vmax - vmin) if vmax > vmin else v

    out = cs3d.inference_on_images(v, config=cfg)[0]
    sem = out.semantic_segmentation

    # ---- FIX: REDUCE TO [Z,Y,X] ----
    # Case 1: [1, C, Z, Y, X]
    if sem.ndim == 5:
        sem = sem[0]

    # Case 2: [C, Z, Y, X] → argmax
    if sem.ndim == 4:
        if sem.shape[0] == 1:
            sem = sem[0]                      # [Z,Y,X]
        else:
            sem = sem.argmax(0)               # multi-class → 3D

    # Now sem MUST be [Z,Y,X]
    if sem.ndim != 3:
        raise RuntimeError(f"Semantic segmentation has invalid shape: {sem.shape}")

    sem = (sem > 0).astype(np.uint8)

    # ---- Instance segmentation ----
    inst, _ = cs3d.post_processing(sem)


    return sem.astype(np.uint8), inst.astype(np.uint16), src

# ============================================================
# SANITY CHECK: Confirm checkpoint, loaded model, and random model differ
# ============================================================
def run_sanity_check(ckpt_path):
    

    print("\n========== SANITY CHECK ==========")
    print(f"Checking checkpoint: {ckpt_path}")

    # -----------------------------------------------------------
    # Load FINETUNED checkpoint (raw or wrapped formats)
    # -----------------------------------------------------------
    raw = torch.load(ckpt_path, map_location="cpu")

    if isinstance(raw, dict) and "model_state_dict" in raw:
        state_dict = raw["model_state_dict"]
    elif isinstance(raw, dict):
        state_dict = raw
    else:
        raise ValueError("Invalid checkpoint format")

    param_name = "encoder.conv1.module.0.weight"

    # Value inside the checkpoint
    ckpt_val = state_dict[param_name][0,0,0,0,0].item()
    print(f"Checkpoint value:         {ckpt_val}")

    # -----------------------------------------------------------
    # FINETUNED model value
    # -----------------------------------------------------------
    model_finetuned = WNet(
        in_channels=1,
        out_channels=1,
        num_classes=2,
        dropout=0.65,
    )
    model_finetuned.load_state_dict(state_dict, strict=True)
    finetuned_val = model_finetuned.encoder.conv1.module[0].weight[0,0,0,0,0].item()

    print(f"Finetuned model value:    {finetuned_val}")
    print(f"Match ckpt <-> loaded?:   {ckpt_val == finetuned_val}")

    # -----------------------------------------------------------
    # PRETRAINED WNet comparison
    # -----------------------------------------------------------
    try:
        pretrained_raw = torch.load(PRETRAINED_WNET, map_location="cpu")

        if "model_state_dict" in pretrained_raw:
            pretrained_sd = pretrained_raw["model_state_dict"]
        else:
            pretrained_sd = pretrained_raw

        model_pretrained = WNet(
            in_channels=1,
            out_channels=1,
            num_classes=2,
            dropout=0.65,
        )
        model_pretrained.load_state_dict(pretrained_sd, strict=True)

        pretrained_val = model_pretrained.encoder.conv1.module[0].weight[0,0,0,0,0].item()
        print(f"Pretrained model value:   {pretrained_val}")
        print(f"Pretrained == finetuned?: {pretrained_val == finetuned_val}")
    except Exception as e:
        print(f"[WARNING] Could not load pretrained weights: {e}")
        pretrained_val = None

    # -----------------------------------------------------------
    # RANDOM initialization comparison
    # -----------------------------------------------------------
    model_random = WNet(
        in_channels=1,
        out_channels=1,
        num_classes=2,
        dropout=0.65,
    )
    random_val = model_random.encoder.conv1.module[0].weight[0,0,0,0,0].item()

    print(f"Random-init value:        {random_val}")
    print(f"Random == finetuned?:     {random_val == finetuned_val}")

    # -----------------------------------------------------------
    # Diagnostics / warnings
    # -----------------------------------------------------------
    print("\n----- SANITY DIAGNOSTIC -----")

    if random_val == finetuned_val:
        print("[ERROR] Finetuned checkpoint matches RANDOM initialization!")
        print("        → Training did NOT update weights.")

    if pretrained_val is not None and pretrained_val == finetuned_val:
        print("[ERROR] Finetuned checkpoint equals PRETRAINED weights!")
        print("        → Finetuning produced NO change.")
    if pretrained_val is not None and ckpt_val == pretrained_val:
        print("[ERROR] Raw checkpoint equals PRETRAINED!")
        print("        → Checkpoint was NOT updated by training.")

    print("==========================================\n")    


# ============================================================
# MAIN EXPERIMENT
# ============================================================
def run_experiment(pool_size, fold_index):

    class_dir = DATA_ROOT / CLASS_NAME
    all_imgs = sorted([f for f in class_dir.iterdir() if is_valid_nii_gz(f)])

    train_imgs, val_imgs, test_imgs, n_train, n_val, n_test = make_splits(all_imgs, pool_size, fold_index)

    print("----------------------------------------------------------")
    print(f"[DEBUG] pool_size={pool_size} fold_index={fold_index}")
    print(f"Train ({len(train_imgs)}): {[p.name for p in train_imgs]}")
    print(f"Val   ({len(val_imgs)}):  {[p.name for p in val_imgs]}")
    print(f"Test  ({len(test_imgs)}): {[p.name for p in test_imgs]}")
    print("----------------------------------------------------------")


    # Working directory
    tmp_root = OUTPUT_ROOT / "tmp" / f"pool{pool_size}" / f"fold{fold_index}"
    train_vol = tmp_root / "train/vol"
    train_lab = tmp_root / "train/lab"
    val_vol   = tmp_root / "val/vol"
    val_lab   = tmp_root / "val/lab"
    test_vol  = tmp_root / "test/vol"
    test_lab  = tmp_root / "test/lab"

    rows = []
    rows += convert_split("train", train_imgs, train_vol, train_lab)
    rows += convert_split("val",   val_imgs,   val_vol,   val_lab)
    rows += convert_split("test",  test_imgs,  test_vol,  test_lab)

    tmp_root.mkdir(parents=True, exist_ok=True)
    with open(tmp_root / "splits.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)

    # ============================================================
    # DEBUG: WHAT FILES ARE ACTUALLY BEING LOADED?
    # ============================================================
    print("\n[DEBUG] Actual training files loaded:")
    train_data_list_debug = c.create_dataset_dict_no_labs(train_vol)
    for item in train_data_list_debug:
        print("  ", item)
    print("----------------------------------------------------------\n")

    # TRAINING CONFIG
    ckpt_dir = tmp_root / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    pretrained = WeightsInfo(use_pretrained=True, use_custom=False)

    # ============================================================
    # DEBUG: CHECK PRETRAINED WEIGHT LOADING BEFORE TRAINING
    # ============================================================
    print("\n[DEBUG] --- Checking pretrained weight loading into WNet ---")

    # Build a WNet with the SAME config CellSeg3D will use
    debug_model = WNet(
        in_channels=1,
        out_channels=1,
        num_classes=2,
        dropout=0.65,
    )

    pretrained_raw = torch.load(PRETRAINED_WNET, map_location="cpu")
    if "model_state_dict" in pretrained_raw:
        pretrained_sd = pretrained_raw["model_state_dict"]
    else:
        pretrained_sd = pretrained_raw

    # Try loading with strict=False to get missing/unexpected keys
    load_result = debug_model.load_state_dict(pretrained_sd, strict=False)

    print("\n[DEBUG] Missing keys (not found in checkpoint):")
    for k in load_result.missing_keys:
        print("   -", k)

    print("\n[DEBUG] Unexpected keys (checkpoint has these but model does not):")
    for k in load_result.unexpected_keys:
        print("   -", k)

    print("\n[DEBUG] Total checkpoint params:", len(pretrained_sd))
    print("[DEBUG] Total model params:", len(debug_model.state_dict()))
    print("[DEBUG] Number of matched keys:",
        len(pretrained_sd) - len(load_result.unexpected_keys))

    print("\n[DEBUG] --- END pretrained weight debug ---\n")


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

    print("[INFO] Training starts…")
    last_logged = 0

    for _ in worker.train():
        epoch = len(worker.total_losses)
        if epoch > last_logged:
            last_logged = epoch
            train_loss = worker.total_losses[-1]
            rec_loss = worker.rec_losses[-1]
            ncut_loss = worker.ncuts_losses[-1]
            msg = f"[EPOCH {epoch}] train={train_loss:.4f} rec={rec_loss:.4f} ncuts={ncut_loss:.4f}"
            if worker.dice_values:
                msg += f" val_dice={worker.dice_values[-1]:.4f}"
            print(msg)

    print("[INFO] Training done.")

    # WRAP CHECKPOINT
    ckpt_best = ckpt_dir / "wnet_best_metric.pth"
    if not ckpt_best.exists():
        raise FileNotFoundError("No checkpoint saved!")

    ckpt_inf = ckpt_dir / "wnet_best_metric_for_inference.pth"
    wrap_checkpoint(ckpt_best, ckpt_inf)

    # ============================================================
    # DEBUG: Compare pretrained vs finetuned weights (layer diffs)
    # ============================================================
    print("\n[DEBUG] --- Analyzing diffs between pretrained and finetuned ---")

    finetuned_raw = torch.load(ckpt_best, map_location="cpu")
    if "model_state_dict" in finetuned_raw:
        finetuned_sd = finetuned_raw["model_state_dict"]
    else:
        finetuned_sd = finetuned_raw

    changed_layers = []
    unchanged_layers = []

    for k, v in finetuned_sd.items():
        if k in pretrained_sd:
            if not torch.allclose(v.cpu(), pretrained_sd[k].cpu()):
                changed_layers.append(k)
            else:
                unchanged_layers.append(k)

    print("[DEBUG] Layers changed during training:", len(changed_layers))
    for k in changed_layers[:20]:
        print("   CHANGED:", k)

    print("[DEBUG] Layers unchanged:", len(unchanged_layers))
    for k in unchanged_layers[:20]:
        print("   SAME:", k)

    print("[DEBUG] --- END finetuned/pretrained diff analysis ---\n")


    # run sanity check
    run_sanity_check(ckpt_best)

    # INFERENCE
    actual_seed = BASE_SEED + (pool_size * 1000) + fold_index
    fold_tag = (
        f"cvfold{fold_index}_ntr{pool_size}_nev{n_test}_"
        f"fttr{n_train}_ftval{n_val}_trlim{pool_size}_seed{actual_seed}"
    )

    pred_dir = OUTPUT_ROOT / "preds" / CLASS_NAME / fold_tag
    pred_dir.mkdir(parents=True, exist_ok=True)

    metrics = []

    for tif_path in sorted(test_vol.glob("*.tif")):
        print(f"[INFO] Inference on {tif_path.name}")
        vol = tiff.imread(str(tif_path)).astype(np.float32)

        sem, inst, src = infer_volume(vol, ckpt_inf)

        base = tif_path.stem
        tiff.imwrite(pred_dir / f"{base}_semantic.tif", sem)
        tiff.imwrite(pred_dir / f"{base}_instances.tif", inst)

        metrics.append({
            "volume": base,
            "used_weights": src,
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

    print("[INFO] Experiment complete.")

# ============================================================
# CLI
# ============================================================
if __name__ == "__main__":
    
    p = argparse.ArgumentParser()
    p.add_argument("--pool-size", type=int, required=True)
    p.add_argument("--fold-index", type=int, required=True)
    args = p.parse_args()

    run_experiment(args.pool_size, args.fold_index)