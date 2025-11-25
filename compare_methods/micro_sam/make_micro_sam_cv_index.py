import json
from pathlib import Path
from collections import defaultdict
import numpy as np

# ----------------------------
# Configuration
# ----------------------------
ROOT = Path("/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches")
OUTFILE = Path("/midtier/paetzollab/scratch/ads4015/compare_methods/micro_sam/cv_index.json")

rng = np.random.default_rng(42)

# ----------------------------
# Step 1: collect all (class_name, raw_path, label_path)
# ----------------------------
records = []  # (class_name, raw_path, label_path)

for class_dir in ROOT.iterdir():
    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    for raw_path in sorted(class_dir.glob("*_ch*.nii.gz")):
        if "_label" in raw_path.name:
            continue
        label_path = raw_path.with_name(raw_path.name.replace(".nii.gz", "_label.nii.gz"))
        if label_path.exists():
            records.append((class_name, raw_path, label_path))

# ----------------------------
# Step 2: group by datatype
# ----------------------------
by_class = defaultdict(list)
for cls, raw, lab in records:
    by_class[cls].append((raw, lab))

# ----------------------------
# Step 3: build CV index entries
# ----------------------------
cv_entries = []

for cls, pairs in by_class.items():
    n_items = len(pairs)

    if n_items < 4:
        print(f"Skipping {cls} (only {n_items} items)")
        continue

    # always 2 test images
    # remaining go into pools 2,3,4,... up to (n_items - 2)
    max_pool = n_items - 2

    for pool_size in range(2, max_pool + 1):
        for fold_index in range(3):  # 3-fold CV
            cv_entries.append({
                "class_name": cls,
                "pool_size": pool_size,
                "fold_index": fold_index
            })

print(f"Total CV tasks: {len(cv_entries)}")
print("Saving JSON index to:", OUTFILE)

OUTFILE.parent.mkdir(parents=True, exist_ok=True)
with OUTFILE.open("w") as f:
    json.dump(cv_entries, f, indent=2)

print("Done.")



