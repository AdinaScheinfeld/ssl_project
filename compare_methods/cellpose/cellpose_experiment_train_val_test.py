#!/usr/bin/env python3

"""
Run Cellpose-SAM finetuning & evaluation for a single split JSON file.

Usage:
    python cellpose_train_eval.py --split-json <path>

This script:
- Loads train/val/test volumes per split
- Finetunes Cellpose-SAM (2D slice training)
- Runs 2D inference on test volumes
- Runs 3D inference on test volumes
- Computes AP metrics
- Saves metrics + predictions
"""

import argparse
import json
import math
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm
from scipy.ndimage import label
from cellpose import models, train, metrics

# -----------------------------------------------------
# Helpers
# -----------------------------------------------------

def clean_name(n):
    return n.replace("_ch0", "").replace("_ch1", "")

def load_patch_as_2D_slices(img_path, lab_path):
    img_nii = nib.load(img_path)
    lab_nii = nib.load(lab_path)
    img = img_nii.get_fdata().astype(np.float32)
    lab = lab_nii.get_fdata().astype(np.int32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    inst_lab = np.zeros_like(lab, dtype=np.int32)
    for z in range(lab.shape[0]):
        inst_lab[z], _ = label(lab[z] > 0)

    imgs_2d = [img[z,:,:] for z in range(img.shape[0])]
    labs_2d = [inst_lab[z,:,:] for z in range(img.shape[0])]
    return imgs_2d, labs_2d

def binary_to_instances(vol):
    out = np.zeros_like(vol, dtype=np.int32)
    for z in range(vol.shape[0]):
        out[z], _ = label(vol[z] > 0)
    return out

# -----------------------------------------------------
# Main
# -----------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--split-json", required=True)
args = parser.parse_args()

with open(args.split_json, "r") as f:
    S = json.load(f)

fold   = S["fold"]
seed   = S["seed"]
ntr    = S["ntr"]
nev    = S["nev"]
fttr   = S["fttr"]
ftval  = S["ftval"]
trlim  = S["trlim"]
test_images = S["test_images"]
pool_images = S["pool_images"]

DATA_ROOT = Path("/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches/cell_nucleus_patches")
OUT_ROOT  = Path("/midtier/paetzollab/scratch/ads4015/compare_methods/cellpose/cross_val")

pred_root = OUT_ROOT / "preds" / "cell_nucleus_patches"
ckpt_root = OUT_ROOT / "checkpoints"
log_root  = OUT_ROOT / "logs"

for d in [pred_root, ckpt_root, log_root]:
    d.mkdir(parents=True, exist_ok=True)

tag = f"cvfold{fold}_ntr{ntr}_nev{nev}_fttr{fttr}_ftval{ftval}_trlim{trlim}_seed{seed}"
exp_dir = pred_root / tag

# -----------------------------------------------------
# Skip if exists
# -----------------------------------------------------
if exp_dir.exists():
    print(f"[SKIP] {tag} already computed.", flush=True)
    exit(0)

exp_dir.mkdir(parents=True)

# -----------------------------------------------------
# Build train/val/test paths
# -----------------------------------------------------
def img_path(n):  return DATA_ROOT / f"{n}_ch0.nii.gz"
def lab_path(n):  return DATA_ROOT / f"{n}_ch0_label.nii.gz"

# train pool
train_pool = pool_images[:fttr]
val_pool   = pool_images[fttr:fttr+ftval]

# -----------------------------------------------------
# Build training dataset (2D slices)
# -----------------------------------------------------
train_data, train_labels = [], []

for name in train_pool:
    imgs2d, labs2d = load_patch_as_2D_slices(img_path(name), lab_path(name))
    train_data.extend(imgs2d)
    train_labels.extend(labs2d)

# -----------------------------------------------------
# Train Cellpose
# -----------------------------------------------------
model_name = tag
model = models.CellposeModel(gpu=True, model_type=None)

ckpt_subdir = ckpt_root / "models"
ckpt_subdir.mkdir(exist_ok=True)

new_model_path, train_losses, test_losses = train.train_seg(
    model.net,
    train_data=train_data,
    train_labels=train_labels,
    batch_size=8,
    n_epochs=100,
    nimg_per_epoch=max(20, len(train_data)),
    learning_rate=1e-5,
    weight_decay=0.1,
    model_name=model_name,
    save_path=str(ckpt_root),
    min_train_masks=0,
    compute_flows=True
)

# Load model
finetuned = models.CellposeModel(gpu=True, pretrained_model=new_model_path)

# -----------------------------------------------------
# Test inference
# -----------------------------------------------------
results = []

for name in test_images:
    img_nii = nib.load(img_path(name))
    gt_nii  = nib.load(lab_path(name))

    img = img_nii.get_fdata().astype(np.float32)
    gt  = gt_nii.get_fdata().astype(np.int32)

    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    Z,H,W = img.shape

    # ---------- 2D inference ----------
    pred2d = np.zeros((Z,H,W), dtype=np.int32)
    for z in range(Z):
        slice2d = img[z,:,:]
        p,_,_ = finetuned.eval(
            slice2d,
            batch_size=1,
            channel_axis=None,
            z_axis=None,
            do_3D=False
        )
        pred2d[z] = p

    # ---------- 3D inference ----------
    masks3d,flows3d,styles3d = finetuned.eval(
        img,
        batch_size=16,
        channel_axis=None,
        z_axis=0,
        do_3D=True,
        flow3D_smooth=1
    )

    # ---------- Metrics ----------
    gt_inst = binary_to_instances(gt)
    try:
        ap_slice = metrics.average_precision(gt_inst, pred2d)[0]
        ap2d = float(np.mean(ap_slice))
    except:
        ap2d = float("nan")

    try:
        ap_slice = metrics.average_precision(gt_inst, masks3d)[0]
        ap3d = float(np.mean(ap_slice))
    except:
        ap3d = float("nan")

    results.append((name, ap2d, ap3d))

    # ---------- Save predictions ----------
    nib.save(nib.Nifti1Image(pred2d.astype(np.int16), img_nii.affine, img_nii.header),
             exp_dir / f"{name}_pred2d_{tag}.nii.gz")

    nib.save(nib.Nifti1Image(masks3d.astype(np.int16), img_nii.affine, img_nii.header),
             exp_dir / f"{name}_pred3d_{tag}.nii.gz")

# Save metrics CSV
import csv
csv_path = exp_dir / f"metrics_cell_nucleus_patches_{tag}.csv"
with open(csv_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["volume", "AP_2D", "AP_3D"])
    for r in results:
        w.writerow(r)

print(f"\nDONE: {tag}", flush=True)




