#!/usr/bin/env python3
import csv
import json
from pathlib import Path

# --------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------
CROSSVAL_ROOT = Path("/midtier/paetzollab/scratch/ads4015/compare_methods/nnunet/cross_val")
RAW_ROOT = CROSSVAL_ROOT / "raw"
REGISTRY = CROSSVAL_ROOT / "dataset_registry.csv"

# --------------------------------------------------------------------------
def extract_case_id(path_str):
    """
    Convert full path:
        /.../patch_001_vol011_ch0.nii.gz
    to nnU-Net case ID:
        case_patch_001_vol011_ch0
    """
    p = Path(path_str)
    base = p.name.replace(".nii.gz", "")
    return f"case_{base}"


def main():
    if not REGISTRY.exists():
        raise RuntimeError(f"Registry file not found: {REGISTRY}")

    rows = []
    with open(REGISTRY, "r") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)

    print(f"Loaded {len(rows)} dataset entries from registry")

    for r in rows:
        dataset_id = int(r["dataset_id"])
        split_json_path = Path(r["split_json_path"])

        # Paths
        droot = RAW_ROOT / f"Dataset{dataset_id}"
        out_split_file = droot / "splits_final.json"

        # Read split JSON (original train/val/test)
        with open(split_json_path, "r") as f:
            split_data = json.load(f)

        # Extract cases
        train_cases = [extract_case_id(p) for p in split_data["train_cases"]]
        val_cases   = [extract_case_id(p) for p in split_data["val_cases"]]

        final_split = [
            {
                "train": train_cases,
                "val": val_cases
            }
        ]

        # Write final split file
        with open(out_split_file, "w") as f:
            json.dump(final_split, f, indent=4)

        print(f"✓ Wrote {out_split_file}  (train={len(train_cases)}, val={len(val_cases)})")

    print("\nALL DONE — nnUNet can now train without internal CV!")

# --------------------------------------------------------------------------
if __name__ == "__main__":
    main()
