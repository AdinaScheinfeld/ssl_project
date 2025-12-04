#!/usr/bin/env python3
import json
import shutil
from pathlib import Path
import nibabel as nib
import numpy as np
import re
import csv

# -------------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------------
CROSSVAL_ROOT = Path("/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/cross_val")
SPLITS_ROOT = CROSSVAL_ROOT / "splits"
RAW_ROOT = CROSSVAL_ROOT / "raw"

RAW_ROOT.mkdir(parents=True, exist_ok=True)

# starting nnUNet dataset ID (nnUNet requires DatasetXXXX)
DATASET_START_ID = 1000

# registry file that maps dataset_id → split metadata
REGISTRY_CSV = CROSSVAL_ROOT / "dataset_registry.csv"


# -------------------------------------------------------------------------
# HELPERS
# -------------------------------------------------------------------------
def save_uint8_label(src, dst):
    """Load a label, cast to uint8, save."""
    lab = nib.load(str(src))
    arr = lab.get_fdata().astype(np.uint8)
    nib.save(nib.Nifti1Image(arr, lab.affine), str(dst))


def parse_split_filename(fname):
    """
    From: split_0012_pool6_seed2.json
    Return:
        {"split_id":12, "pool":6, "seed":2}
    """

    pat = r"split_(\d+)_pool(\d+)_seed(\d+)\.json"
    m = re.match(pat, fname)
    if not m:
        return None

    return {
        "split_id": int(m.group(1)),
        "pool": int(m.group(2)),
        "seed": int(m.group(3))
    }


def add_item(src_img: Path, out_img_dir: Path, out_lab_dir: Path, entry_list: list):
    """
    Copy image + label pair to nnUNet folder using preserved filenames.
    """
    base = src_img.name                       # patch_001_vol011_ch0.nii.gz
    base_noext = base.replace(".nii.gz", "")  # patch_001_vol011_ch0

    # label path
    label_src = src_img.with_name(f"{base_noext}_label.nii.gz")

    # case ID preserves original filename
    case_id = f"case_{base_noext}"

    dst_img = out_img_dir / f"{case_id}_0000.nii.gz"
    dst_lab = out_lab_dir / f"{case_id}.nii.gz"

    # copy image
    shutil.copy2(src_img, dst_img)

    # cast label to uint8 → required for nnUNet
    save_uint8_label(label_src, dst_lab)

    # add to dataset.json entries
    entry_list.append({
        "image": f"./{out_img_dir.name}/{case_id}_0000.nii.gz",
        "label": f"./{out_lab_dir.name}/{case_id}.nii.gz"
    })


def build_dataset_for_split(split_json: Path, dataset_id: int):
    """
    Build an nnUNet raw dataset directory for one split.
    """
    with open(split_json, "r") as f:
        split = json.load(f)

    train_imgs = [Path(p) for p in split["train_cases"]]
    val_imgs   = [Path(p) for p in split["val_cases"]]
    test_imgs  = [Path(p) for p in split["test_cases"]]

    # train + val combined for nnUNet
    train_all = train_imgs + val_imgs

    # output directory
    droot = RAW_ROOT / f"Dataset{dataset_id}"
    IMTR = droot / "imagesTr"
    LBTR = droot / "labelsTr"
    IMTS = droot / "imagesTs"
    LBTS = droot / "labelsTs"

    for p in [IMTR, LBTR, IMTS, LBTS]:
        p.mkdir(parents=True, exist_ok=True)

    training_entries = []
    test_entries = []

    # ---- add training items ----
    for img_path in train_all:
        add_item(img_path, IMTR, LBTR, training_entries)

    # ---- add test items ----
    for img_path in test_imgs:
        add_item(img_path, IMTS, LBTS, test_entries)

    # ---- write dataset.json ----
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

    with open(droot / "dataset.json", "w") as f:
        json.dump(dataset_json, f, indent=4)

    return len(training_entries), len(test_entries)


# -------------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------------
def main():
    dataset_id = DATASET_START_ID
    registry_rows = []

    for dtype_dir in sorted(SPLITS_ROOT.iterdir()):
        if not dtype_dir.is_dir():
            continue

        dtype = dtype_dir.name
        print(f"\n=== Processing datatype: {dtype} ===")

        split_files = sorted(dtype_dir.glob("split_*.json"))
        print(f"Found {len(split_files)} splits")

        for split_path in split_files:
            meta = parse_split_filename(split_path.name)
            if meta is None:
                print(f"  Skipping malformed filename: {split_path.name}")
                continue

            print(f"  → Building Dataset{dataset_id} for {split_path.name}")

            ntrain, ntest = build_dataset_for_split(split_path, dataset_id)

            registry_rows.append([
                dataset_id,
                dtype,
                meta["split_id"],
                meta["pool"],
                meta["seed"],
                ntrain,
                ntest,
                str(split_path)
            ])

            dataset_id += 1

    # final registry CSV
    with open(REGISTRY_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "dataset_id",
            "datatype",
            "split_id",
            "pool_size",
            "seed",
            "num_train",
            "num_test",
            "split_json_path"
        ])
        writer.writerows(registry_rows)

    print("\n✓ Done building datasets")
    print(f"✓ Registry saved to {REGISTRY_CSV}")


if __name__ == "__main__":
    main()
