#!/usr/bin/env python3

"""
Generate all train/val/test splits for Cellpose-SAM finetuning.

For each pool size p = 2,3,...,N-2 and each fold f=0,1,2:
- randomly pick 2 test images
- randomly pick p pool images from the remaining
- split pool 80/20 (min 1 each)
- save JSON config used by training/eval script
"""

import json
import random
import math
from pathlib import Path

DATA_ROOT = Path("/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches/cell_nucleus_patches")
OUT_ROOT  = Path("/midtier/paetzollab/scratch/ads4015/compare_methods/cellpose/cross_val")
SPLIT_DIR = OUT_ROOT / "splits"
SPLIT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------
# Collect all images (only ch0, labels inferred automatically)
# ---------------------------------------------------------
imgs = sorted(DATA_ROOT.glob("*_ch0.nii.gz"))
img_names = [p.name.replace("_ch0.nii.gz", "") for p in imgs]

num_imgs = len(img_names)
print(f"Found {num_imgs} volumes.", flush=True)

# We will generate many splits — track index
split_idx = 0
all_splits = []

# ---------------------------------------------------------
# Loop pool sizes and folds
# ---------------------------------------------------------
for pool_size in range(2, num_imgs - 1):   # must leave at least 2 for test
    for fold in range(3):
        seed = 100 + fold*1000 + pool_size
        rng = random.Random(seed)

        # Shuffle volumes
        shuffled = img_names.copy()
        rng.shuffle(shuffled)

        # Pick 2 test images
        test = shuffled[:2]
        remaining = shuffled[2:]

        # Pick pool of p images
        if len(remaining) < pool_size:
            continue
        pool = remaining[:pool_size]

        # 80/20 split
        fttr = max(1, math.floor(0.8 * pool_size))
        ftval = max(1, pool_size - fttr)

        trlim = 2  # always

        # Save split json
        split_info = {
            "fold": fold,
            "seed": seed,
            "pool_size": pool_size,
            "ntr": pool_size,
            "nev": 2,
            "fttr": fttr,
            "ftval": ftval,
            "trlim": trlim,
            "test_images": test,
            "pool_images": pool,
        }

        split_path = SPLIT_DIR / f"split_{split_idx}.json"
        with open(split_path, "w") as f:
            json.dump(split_info, f, indent=2)

        all_splits.append(str(split_path))
        print(f"Saved {split_path}", flush=True)
        split_idx += 1

print(f"\nGenerated {len(all_splits)} splits total.", flush=True)




