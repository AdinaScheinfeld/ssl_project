#!/usr/bin/env python3

# /home/ads4015/ssl_project/compare_methods/cellseg3d/make_cellseg3d_cv_index.py

"""
Generate CV index for CellSeg3D finetuning experiments.

Output: JSON list of entries, each entry specifying:
    {
        "pool_size": int,
        "fold_index": int
    }

You will use these entries with the Slurm array job to run
cellseg3d_finetune_cv.py.
"""

import json
from pathlib import Path
import numpy as np

# ----------------------------------------
# Paths
# ----------------------------------------
DATA_ROOT = Path("/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches")
CLASS_NAME = "cell_nucleus_patches"
OUTFILE = Path("/midtier/paetzollab/scratch/ads4015/compare_methods/cellseg3d/cv_index.json")

# ----------------------------------------
# Scan images for this class
# ----------------------------------------
class_dir = DATA_ROOT / CLASS_NAME
if not class_dir.is_dir():
    raise RuntimeError(f"Class directory not found: {class_dir}")

all_imgs = []
for f in sorted(class_dir.glob("*.nii.gz")):
    if f.name.endswith("_ch0.nii.gz") and "_label" not in f.name:
        all_imgs.append(f)

n_items = len(all_imgs)
if n_items < 4:
    raise RuntimeError(
        f"Need at least 4 labeled volumes for 2 test + pool. Found {n_items}."
    )

print(f"Found {n_items} usable volumes in {CLASS_NAME}")

# ----------------------------------------
# Build list of CV tasks
# ----------------------------------------
cv_entries = []

# Always 2 test → remaining for train+val pools
max_pool = n_items - 2  # maximum possible train+val pool size

for pool_size in range(2, max_pool + 1):
    for fold_idx in range(3):  # 3-fold CV
        cv_entries.append({
            "pool_size": pool_size,
            "fold_index": fold_idx
        })

print(f"Total tasks generated: {len(cv_entries)}")
print(f"Pool sizes: 2 → {max_pool}")

# ----------------------------------------
# Save JSON
# ----------------------------------------
OUTFILE.parent.mkdir(parents=True, exist_ok=True)
with OUTFILE.open("w") as f:
    json.dump(cv_entries, f, indent=2)

print(f"Saved CV index to: {OUTFILE}")
