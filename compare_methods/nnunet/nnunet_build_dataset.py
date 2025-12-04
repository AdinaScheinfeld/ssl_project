#!/usr/bin/env python3

# /home/ads4015/ssl_project/compare_methods/nnunet/nnunet_build_dataset.py

import os
from pathlib import Path
import shutil
import json
import numpy as np
import nibabel as nib
import random

# --- paths ---
src = Path("/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches")
raw_root = Path(os.environ["nnUNet_raw"]) / "Dataset001_LSM"

imagesTr = raw_root / "imagesTr"
labelsTr = raw_root / "labelsTr"
imagesTs = raw_root / "imagesTs"
labelsTs = raw_root / "labelsTs"

imagesTr.mkdir(parents=True, exist_ok=True)
labelsTr.mkdir(parents=True, exist_ok=True)
imagesTs.mkdir(parents=True, exist_ok=True)
labelsTs.mkdir(parents=True, exist_ok=True)

subfolders = [
    "amyloid_plaque_patches",
    "c_fos_positive_patches",
    "cell_nucleus_patches",
    "vessels_patches",
]

# --- collect image-label pairs ---
pairs = []

for sub in subfolders:
    folder = src / sub
    for img in sorted(folder.glob("*.nii.gz")):
        if "_label" in img.name:
            continue
        base = img.name.replace(".nii.gz", "")
        lab = img.with_name(base + "_label.nii.gz")
        if lab.exists():
            pairs.append((img, lab))

print("Total patches found:", len(pairs))

# --- reserve 10 for test ---
random.seed(0)
test_pairs = random.sample(pairs, 10)
train_pairs = [p for p in pairs if p not in test_pairs]

# --- helper: save label as uint8 ---
def save_uint8_label(src_path, dst_path):
    lab = nib.load(str(src_path))
    arr = lab.get_fdata()
    arr = arr.astype(np.uint8)
    nib.save(nib.Nifti1Image(arr, lab.affine), str(dst_path))

# --- write training files ---
training_entries = []
case_id = 0

for img, lab in train_pairs:
    cid = f"case_{case_id:05d}"
    shutil.copy2(img, imagesTr / f"{cid}_0000.nii.gz")
    save_uint8_label(lab, labelsTr / f"{cid}.nii.gz")
    training_entries.append({
        "image": f"./imagesTr/{cid}_0000.nii.gz",
        "label": f"./labelsTr/{cid}.nii.gz"
    })
    case_id += 1

# --- write test files ---
test_entries = []
for img, lab in test_pairs:
    cid = f"case_{case_id:05d}"
    shutil.copy2(img, imagesTs / f"{cid}_0000.nii.gz")
    save_uint8_label(lab, labelsTs / f"{cid}.nii.gz")
    test_entries.append({
        "image": f"./imagesTs/{cid}_0000.nii.gz",
        "label": f"./labelsTs/{cid}.nii.gz"
    })
    case_id += 1

# --- write dataset.json ---
dataset_json = {
    "channel_names": {"0": "LSM"},
    "labels": {
        "background": "0",
        "foreground": "1"
    },
    "regions_class_order": [],
    "file_ending": ".nii.gz",
    "numTraining": len(training_entries),
    "numTest": len(test_entries),
    "training": training_entries,
    "test": test_entries
}

with open(raw_root / "dataset.json", "w") as f:
    json.dump(dataset_json, f, indent=4)

print("Done! Training:", len(training_entries), "Test:", len(test_entries))




