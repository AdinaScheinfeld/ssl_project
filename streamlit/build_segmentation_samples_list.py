#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import pandas as pd

CVFOLD_RE = re.compile(r"^cvfold(\d+)_")

def list_runfolders(root: Path, datatype: str):
    base = root / "preds" / datatype
    if not base.exists():
        return []
    return sorted([p for p in base.iterdir() if p.is_dir()])

def pick_one_runfolder_per_fold(runfolders):
    """
    Deterministic: choose lexicographically first runfolder for each fold id found.
    Returns dict fold_id -> Path(runfolder)
    """
    fold_map = {}
    for rf in runfolders:
        m = CVFOLD_RE.match(rf.name)
        if not m:
            continue
        fold = int(m.group(1))
        fold_map.setdefault(fold, []).append(rf)

    # pick first sorted per fold
    picked = {}
    for fold, rfs in fold_map.items():
        picked[fold] = sorted(rfs, key=lambda p: p.name)[0]
    return picked

def list_pred_files(runfolder: Path):
    pred_dir = runfolder / "preds"
    if not pred_dir.exists():
        return []
    return sorted(pred_dir.glob("*.nii.gz"))

def corresponding_runfolder(model_root: Path, datatype: str, runfolder_name: str):
    return model_root / "preds" / datatype / runfolder_name

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image_clip_root", type=Path, required=True)
    ap.add_argument("--image_only_root", type=Path, required=True)
    ap.add_argument("--random_root", type=Path, required=True)

    ap.add_argument("--datatypes", nargs="+",
                    default=["amyloid_plaque", "c_fos_positive", "cell_nucleus", "vessels"])
    ap.add_argument("--z_planes", nargs="+", type=int, default=[32, 64])
    ap.add_argument("--preds_per_fold", type=int, default=2)   # you want 2 preds per fold folder
    ap.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2])  # default 3-fold CV
    ap.add_argument("--out_csv", type=Path, default=Path("human_eval_samples.csv"))
    args = ap.parse_args()

    roots = {
        "image_clip": args.image_clip_root,
        "image_only": args.image_only_root,
        "random":     args.random_root,
    }

    records = []

    for datatype in args.datatypes:
        print(f"\nProcessing datatype: {datatype}", flush=True)

        # 1) Find runfolders in image_clip root
        runfolders = list_runfolders(roots["image_clip"], datatype)
        if not runfolders:
            raise SystemExit(f"No runfolders found for datatype={datatype} under {roots['image_clip_root']}")

        # 2) Pick ONE runfolder per fold deterministically
        picked = pick_one_runfolder_per_fold(runfolders)

        # Ensure we have desired folds
        missing_folds = [f for f in args.folds if f not in picked]
        if missing_folds:
            raise SystemExit(f"Missing folds {missing_folds} for datatype={datatype} in {roots['image_clip']}")

        for fold in args.folds:
            rf_clip = picked[fold]
            run_name = rf_clip.name

            # 3) Ensure corresponding runfolders exist for other models
            rf_only = corresponding_runfolder(roots["image_only"], datatype, run_name)
            rf_rand = corresponding_runfolder(roots["random"], datatype, run_name)

            if not rf_only.exists():
                raise SystemExit(f"Missing image_only runfolder: {rf_only}")
            if not rf_rand.exists():
                raise SystemExit(f"Missing random runfolder: {rf_rand}")

            # 4) Pick exactly preds_per_fold files from the image_clip runfolder
            clip_files = list_pred_files(rf_clip)
            if len(clip_files) < args.preds_per_fold:
                raise SystemExit(f"Found only {len(clip_files)} preds in {rf_clip}/preds, expected {args.preds_per_fold}")

            chosen_clip = clip_files[:args.preds_per_fold]

            for clip_path in chosen_clip:
                only_path = rf_only / "preds" / clip_path.name
                rand_path = rf_rand / "preds" / clip_path.name

                if not only_path.exists():
                    raise SystemExit(f"Missing matching file in image_only: {only_path}")
                if not rand_path.exists():
                    raise SystemExit(f"Missing matching file in random: {rand_path}")

                for z in args.z_planes:
                    sample_id = f"{datatype}_{run_name}_{clip_path.stem}_z{z}"
                    records.append({
                        "sample_id": sample_id,
                        "datatype": datatype,
                        "cvfold": fold,
                        "runfolder": run_name,
                        "filename": clip_path.name,
                        "z": z,
                        "image_clip_path": str(clip_path),
                        "image_only_path": str(only_path),
                        "random_path": str(rand_path),
                    })

        # helpful logging
        per_dtype = len(args.folds) * args.preds_per_fold * len(args.z_planes)
        print(f"  Added {per_dtype} rows for {datatype} "
              f"({len(args.folds)} folds × {args.preds_per_fold} preds × {len(args.z_planes)} z-planes)", flush=True)

    df = pd.DataFrame(records)
    expected = len(args.datatypes) * len(args.folds) * args.preds_per_fold * len(args.z_planes)
    print(f"\nTotal samples collected: {len(df)}", flush=True)
    if len(df) != expected:
        raise SystemExit(f"ERROR: expected {expected} rows but got {len(df)}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}", flush=True)

if __name__ == "__main__":
    main()



