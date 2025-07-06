# split_train_val_by_volume.py

import os
import random
from pathlib import Path
import shutil
import argparse

# For each datatype (e.g., cfos, plaque...), ensure that no volume contributes to both train and val.

def extract_volume_id(file_path):
    """Extract volume ID from filename assuming format like: patch_000_vol005_ch0.pt"""
    stem = file_path.stem
    parts = stem.split("_")
    for p in parts:
        if p.startswith("vol"):
            return p
    return None

def group_by_volume(pt_files):
    volume_to_files = {}
    for file in pt_files:
        vol_id = extract_volume_id(file)
        if vol_id is None:
            continue
        volume_to_files.setdefault(vol_id, []).append(file)
    return volume_to_files

def split_volumes(volume_to_files, val_frac, seed):
    volume_ids = list(volume_to_files.keys())
    random.seed(seed)
    random.shuffle(volume_ids)

    n_val = max(1, int(len(volume_ids) * val_frac))
    val_vols = set(volume_ids[:n_val])
    train_vols = set(volume_ids[n_val:])

    train_files = [f for vid in train_vols for f in volume_to_files[vid]]
    val_files = [f for vid in val_vols for f in volume_to_files[vid]]
    return train_files, val_files, train_vols, val_vols

def split_and_copy(input_root, output_root, val_frac=0.2, seed=42):
    input_root = Path(input_root)
    output_root = Path(output_root)
    classes = [d for d in input_root.iterdir() if d.is_dir()]

    for class_dir in classes:
        print(f"\nProcessing {class_dir.name}...")
        pt_files = list(class_dir.glob("*.pt"))
        volume_to_files = group_by_volume(pt_files)
        train_files, val_files, train_vols, val_vols = split_volumes(volume_to_files, val_frac, seed)

        print("  Train volumes:", sorted(train_vols))
        print("  Val volumes:", sorted(val_vols))

        for split, file_list in zip(["train", "val"], [train_files, val_files]):
            split_dir = output_root / split / class_dir.name
            split_dir.mkdir(parents=True, exist_ok=True)
            print(f"  {split.upper()} PATCHES:")
            for file in sorted(file_list):
                print(f"    {file.name}")
                shutil.copy(file, split_dir / file.name)

        print(f"  -> {len(train_files)} train files, {len(val_files)} val files")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_root", type=str, required=True, help="Input root containing per-class folders of .pt files")
    parser.add_argument("--output_root", type=str, required=True, help="Output root to store train/val split")
    parser.add_argument("--val_frac", type=float, default=0.2, help="Fraction of volumes to use for validation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    split_and_copy(args.input_root, args.output_root, args.val_frac, args.seed)
