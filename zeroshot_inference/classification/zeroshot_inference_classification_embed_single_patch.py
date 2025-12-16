#!/usr/bin/env python
# Zero-shot classification (embedding extraction) on a single 3D patch
#
# Outputs: for each patch and each model:
#   - embedding (.npy)
#   - metadata (.json)
#
# Then notebook does nearest-centroid classification + metrics.

import os
import re
import glob
import json
import time
import argparse
import numpy as np
import nibabel as nib
import torch
import sys

from monai.networks.nets import SwinUNETR

# -----------------------------
# Logging
# -----------------------------
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


DEVICE = "cuda"

# -----------------------------
# Data roots
# -----------------------------
SELMA_ROOT = "/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
MESO_ROOT  = "/midtier/paetzollab/scratch/ads4015/all_mesospim_patches"
ALLEN_ROOT = "/midtier/paetzollab/scratch/ads4015/all_allen_human_patches"

OUT_ROOT   = "/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_zeroshot"

# Selma classes are folder names
SELMA_CLASSES = [
    "amyloid_plaque_patches",
    "c_fos_positive_patches",
    "cell_nucleus_patches",
    "vessels_patches",
]

# mesoSPIM classes appear in filename
MESO_CLASSES = ["VIP_ASLM_on", "VIP_ASLM_off", "TPH2"]

# Allen human class is second group in filename: allenhuman_<class>_...
ALLEN_CLASSES = ["cr", "lec", "nn", "npy", "yo"]


# -----------------------------
# Models
# -----------------------------
MODELS = {
    "image_clip": {
        "ckpt": "/midtier/paetzollab/scratch/ads4015/checkpoints/autumn_sweep_27/all_datasets_clip_pretrained-updated-epochepoch=354-val-reportval_loss_report=0.0968-stepstep=20590.ckpt",
        "feature_size": 36,
    },
    "image_only": {
        "ckpt": "/ministorage/adina/pretrain_sweep_no_clip/checkpoints/r605gzgj/all_datasets_pretrained_no_clip-epochepoch=183-valval_loss=0.0201-stepstep=10672.ckpt",
        "feature_size": 24,
    },
    "random": {
        "ckpt": None,
        "feature_size": 36,
    },
}


# -----------------------------
# IO utils
# -----------------------------
def load_nifti(path):
    nii = nib.load(path)
    return nii.get_fdata().astype(np.float32), nii.affine


def normalize01(x):
    mn = float(np.min(x))
    mx = float(np.max(x))
    return (x - mn) / (mx - mn + 1e-8)


# -----------------------------
# Build patch list (global)
# -----------------------------
def discover_items():
    """
    Returns list of dicts:
      {
        "source": "selma"|"mesospim"|"allen_human",
        "label": <class_str>,
        "path": <nii.gz path>,
        "rel_id": <stable id string for naming>,
      }
    """
    items = []

    # ---- SELMA ----
    for cls in SELMA_CLASSES:
        files = sorted(
            f for f in glob.glob(os.path.join(SELMA_ROOT, cls, "*.nii.gz"))
            if not f.endswith("_label.nii.gz")
        )
        for p in files:
            base = os.path.basename(p).replace(".nii.gz", "")
            items.append({
                "source": "selma",
                "label": cls.replace("_patches", ""),  # nicer label
                "path": p,
                "rel_id": f"selma/{cls}/{base}",
            })

    # ---- mesoSPIM ----
    meso_files = sorted(glob.glob(os.path.join(MESO_ROOT, "*.nii.gz")))
    for p in meso_files:
        bn = os.path.basename(p)
        label = None
        for c in MESO_CLASSES:
            if c in bn:
                label = c
                break
        if label is None:
            continue
        base = bn.replace(".nii.gz", "")
        items.append({
            "source": "mesospim",
            "label": label,
            "path": p,
            "rel_id": f"mesospim/{label}/{base}",
        })

    # ---- Allen human ----
    allen_files = sorted(glob.glob(os.path.join(ALLEN_ROOT, "*.nii.gz")))
    for p in allen_files:
        bn = os.path.basename(p)
        # expected: allenhuman_<class>_...
        m = re.match(r"allenhuman_([^_]+)_", bn)
        if m is None:
            continue
        label = m.group(1)
        if label not in ALLEN_CLASSES:
            continue
        base = bn.replace(".nii.gz", "")
        items.append({
            "source": "allen_human",
            "label": label,
            "path": p,
            "rel_id": f"allen_human/{label}/{base}",
        })

    return items


# -----------------------------
# Model loading (backbone only)
# -----------------------------
def build_model(ckpt_path, feature_size, name):
    log(f"Building model '{name}' with feature_size={feature_size}")

    model = SwinUNETR(
        img_size=(96, 96, 96),   # warning ok; can remove later
        in_channels=1,
        out_channels=1,
        feature_size=feature_size,
        use_checkpoint=False,
    ).to(DEVICE)

    if ckpt_path is not None:
        log(f"Loading checkpoint for '{name}' from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        raw_state = ckpt["state_dict"]

        # pull student Swin backbone
        swin_state = {}
        for k, v in raw_state.items():
            if k.startswith("student_encoder.swinViT."):
                new_k = k.replace("student_encoder.swinViT.", "")
                swin_state[new_k] = v

        log(f"Found {len(swin_state)} Swin backbone keys in checkpoint")

        missing, unexpected = model.swinViT.load_state_dict(swin_state, strict=False)
        num_model_keys = len(model.swinViT.state_dict())
        num_loaded = num_model_keys - len(missing)
        log(f"'{name}' Swin backbone loaded: {num_loaded}/{num_model_keys} "
            f"(missing={len(missing)}, unexpected={len(unexpected)})")

    else:
        log(f"'{name}' using RANDOM initialization")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    return model


# -----------------------------
# Feature extraction
# -----------------------------
def extract_decoder1_embedding(model, img_np):
    """
    Returns a 1D embedding vector from decoder1 via global average pooling.
    """
    img_np = normalize01(img_np)
    x = torch.from_numpy(img_np[None, None]).to(DEVICE)  # (1,1,96,96,96)

    feats = {}

    def hook_fn(m, inp, out):
        feats["f"] = out

    h = model.decoder1.register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(x)
    h.remove()

    f = feats["f"]  # (1, C, 96, 96, 96)
    emb = f.mean(dim=(2, 3, 4)).squeeze(0)  # (C,)
    emb = emb.detach().float().cpu().numpy().astype(np.float32)
    return emb


# -----------------------------
# Main
# -----------------------------
def main(task_id: int, print_num_tasks: bool = False):
    items = discover_items()
    log(f"Discovered {len(items)} total classification patches")

    if print_num_tasks:
        print(len(items))
        return

    if task_id < 0 or task_id >= len(items):
        log(f"Task {task_id} out of range — exiting")
        return

    item = items[task_id]
    src = item["source"]
    label = item["label"]
    path = item["path"]
    rel_id = item["rel_id"]

    log(f"Task {task_id} assigned item: {rel_id}")
    img_np, _ = load_nifti(path)
    log(f"Loaded image shape={img_np.shape}, dtype={img_np.dtype}, "
        f"range=[{img_np.min():.3f}, {img_np.max():.3f}]")

    # output dirs
    os.makedirs(OUT_ROOT, exist_ok=True)
    os.makedirs(os.path.join(OUT_ROOT, "logs"), exist_ok=True)

    # We will store embeddings here:
    # OUT_ROOT/embeddings/<model>/<source>/<class>/<basename>.npy
    base = os.path.basename(path).replace(".nii.gz", "")

    # Load models once for this patch
    models = {}
    for mname, cfg in MODELS.items():
        models[mname] = build_model(cfg["ckpt"], cfg["feature_size"], mname)

    # Extract + save
    for mname, model in models.items():
        out_dir = os.path.join(OUT_ROOT, "embeddings", mname, src, label)
        os.makedirs(out_dir, exist_ok=True)

        emb = extract_decoder1_embedding(model, img_np)

        emb_path = os.path.join(out_dir, f"{base}.npy")
        np.save(emb_path, emb)

        meta = {
            "task_id": task_id,
            "source": src,
            "label": label,
            "path": path,
            "rel_id": rel_id,
            "model": mname,
            "feature_size": int(emb.shape[0]),
        }
        meta_path = os.path.join(out_dir, f"{base}.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        log(f"Saved embedding for '{mname}' → {emb_path} (dim={emb.shape[0]})")

    log(f"Task {task_id} completed successfully")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", type=int, required=False, default=0)
    ap.add_argument("--print-num-tasks", action="store_true")
    args = ap.parse_args()

    main(args.task_id, print_num_tasks=args.print_num_tasks)







