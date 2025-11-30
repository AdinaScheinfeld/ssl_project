#!/usr/bin/env python3
# /home/ads4015/ssl_project/compare_methods/cellseg3d/make_cellseg3d_cv_index.py

"""
Generate CV index for CellSeg3D finetuning experiments.

For a dataset of N volumes:
    - always hold out 2 for testing
    - remaining (N-2) volumes are used for train/val pools
    - pool sizes range from 2 → (N-2)
    - for each pool size, generate 3 randomized folds
      (each fold has independent train/val/test splits)

Output saved to:
    /midtier/paetzollab/scratch/ads4015/compare_methods/cellseg3d/cv_index.json
"""

import json
from pathlib import Path

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
DATA_ROOT = Path("/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches")
CLASS_NAME = "cell_nucleus_patches"

OUTFILE = Path("/midtier/paetzollab/scratch/ads4015/compare_methods/cellseg3d/cv_index.json")
OUTFILE.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Load available volumes
# ---------------------------------------------------------------------
class_dir = DATA_ROOT / CLASS_NAME
if not class_dir.is_dir():
    raise RuntimeError(f"Class directory not found: {class_dir}")

all_imgs = sorted(
    f for f in class_dir.glob("*.nii.gz")
    if f.name.endswith("_ch0.nii.gz") and "_label" not in f.name
)

n_items = len(all_imgs)
print(f"[INFO] Found {n_items} usable volumes in {CLASS_NAME}")

if n_items < 4:
    raise RuntimeError(
        f"[ERROR] Need at least 4 volumes (2 test + pool >= 2). Found: {n_items}"
    )

# ---------------------------------------------------------------------
# Build CV index list
# ---------------------------------------------------------------------
cv_entries = []
max_pool = n_items - 2  # always hold out 2 for testing

print(f"[INFO] Pool sizes: 2 → {max_pool}")
print(f"[INFO] Folds per pool size: 3")

for pool_size in range(2, max_pool + 1):
    for fold_idx in range(3):
        cv_entries.append({
            "pool_size": pool_size,
            "fold_index": fold_idx
        })

print(f"[INFO] Total CV tasks generated: {len(cv_entries)}")

# ---------------------------------------------------------------------
# Save JSON index
# ---------------------------------------------------------------------
with OUTFILE.open("w") as f:
    json.dump(cv_entries, f, indent=2)

print(f"[INFO] Saved CV index JSON → {OUTFILE}")
