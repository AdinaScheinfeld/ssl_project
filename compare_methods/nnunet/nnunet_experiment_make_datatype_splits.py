#!/usr/bin/env python3

# /home/ads4015/ssl_project/compare_methods/nnunet/nnunet_experiment_make_datatype_splits.py


import json
import random
from pathlib import Path

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
SRC_ROOT = Path("/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches")
OUT_ROOT = Path("/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/cross_val/splits")
OUT_ROOT.mkdir(parents=True, exist_ok=True)

DATATYPES = [
    "amyloid_plaque_patches",
    "c_fos_positive_patches",
    "cell_nucleus_patches",
    "vessels_patches",
]

N_TEST = 2     # hold-out test images
N_FOLDS = 3    # number of CV seeds
# ------------------------------------------------------------

def get_pairs(folder: Path):
    """Return list of (img, label) pairs ignoring .pt."""
    imgs = sorted(f for f in folder.glob("*_ch*.nii.gz") if "_label" not in f.name)
    pairs = []
    for img in imgs:
        base = img.name.replace(".nii.gz", "")
        lab = folder / f"{base}_label.nii.gz"
        if lab.exists():
            pairs.append((img, lab))
    return pairs


def make_splits(pairs):
    """Generate splits: each split is a dict."""
    all_splits = []
    total = len(pairs)

    for pool in range(2, total + 1):
        for seed in range(N_FOLDS):
            rng = random.Random(seed)

            if total < N_TEST + 2:
                continue

            # Choose test set
            test = rng.sample(pairs, N_TEST)
            remaining = [p for p in pairs if p not in test]

            if len(remaining) < 2:
                continue

            pool_items = rng.sample(remaining, min(pool, len(remaining)))
            if len(pool_items) < 2:
                continue

            # 80/20 split
            k = max(1, int(0.8 * len(pool_items)))
            if k == len(pool_items):
                k -= 1

            train = pool_items[:k]
            val = pool_items[k:]

            split = {
                "seed": seed,
                "pool_size": pool,
                "train_cases": [str(p[0]) for p in train],
                "val_cases": [str(p[0]) for p in val],
                "test_cases": [str(p[0]) for p in test],
            }
            all_splits.append(split)

    return all_splits


def main():
    for dtype in DATATYPES:
        print(f"\n=== Processing {dtype} ===")
        folder = SRC_ROOT / dtype

        if not folder.exists():
            print(f"Skipping {dtype}: folder does not exist")
            continue

        pairs = get_pairs(folder)
        print(f"Found {len(pairs)} image/label pairs")

        if len(pairs) < 4:
            print(f"Skipping {dtype}: not enough patches for splits")
            continue

        splits = make_splits(pairs)

        # Per-datatype directory
        dtype_dir = OUT_ROOT / dtype
        dtype_dir.mkdir(exist_ok=True)

        # Write each split as its own JSON
        for i, sp in enumerate(splits):
            pool_size = sp["pool_size"]
            seed = sp["seed"]
            fname = f"split_{i:04d}_pool{pool_size}_seed{seed}.json"
            with open(dtype_dir / fname, "w") as f:
                json.dump(sp, f, indent=4)

        print(f"✓ Created {len(splits)} individual split JSON files in: {dtype_dir}")


if __name__ == "__main__":
    main()
