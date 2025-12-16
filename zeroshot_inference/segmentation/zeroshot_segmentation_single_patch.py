#!/usr/bin/env python

# /home/ads4015/ssl_project/zeroshot_inference/segmentation/zeroshot_segmentation_single_patch.py - Zero-shot segmentation inference on a single 3D patch

# imports
import os
import glob
import argparse
import torch
import numpy as np
import nibabel as nib
from sklearn.cluster import KMeans
from monai.networks.nets import SwinUNETR
import sys
import time

# logging utility
def log(msg):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}")
    sys.stdout.flush()


# device
DEVICE = "cuda"

# paths
DATA_ROOT = "/midtier/paetzollab/scratch/ads4015/data_selma3d/selma3d_finetune_patches"
OUT_ROOT  = "/midtier/paetzollab/scratch/ads4015/temp_selma_segmentation_preds_zeroshot"

# model configurations
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
        "feature_size": 36,  # choose one baseline; just be explicit
    },
}


# datasets
DATASETS = [
    "amyloid_plaque_patches",
    "c_fos_positive_patches",
    "cell_nucleus_patches",
    "vessels_patches",
]


# -----------------------------
# Utils
# -----------------------------
def load_nifti(path):
    nii = nib.load(path)
    return nii.get_fdata().astype(np.float32), nii.affine


def build_model(ckpt_path, feature_size, name):
    log(f"Building model '{name}' with feature_size={feature_size}")

    model = SwinUNETR(
        img_size=(96,96,96),
        in_channels=1,
        out_channels=1,
        feature_size=feature_size,
        use_checkpoint=False,
    ).to(DEVICE)

    if ckpt_path is not None:
        log(f"Loading checkpoint for '{name}' from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

        raw_state = ckpt["state_dict"]

        # --- Extract and remap Swin backbone weights ---
        swin_state = {}
        for k, v in raw_state.items():
            if k.startswith("student_encoder.swinViT."):
                new_k = k.replace("student_encoder.swinViT.", "")
                swin_state[new_k] = v

        log(f"Found {len(swin_state)} Swin backbone keys in checkpoint")

        missing, unexpected = model.swinViT.load_state_dict(
            swin_state, strict=False
        )

        num_model_keys = len(model.swinViT.state_dict())
        num_loaded_keys = num_model_keys - len(missing)

        log(
            f"'{name}' Swin backbone loaded: "
            f"{num_loaded_keys}/{num_model_keys} keys "
            f"(missing={len(missing)}, unexpected={len(unexpected)})"
        )

        print(f"[DEBUG] Missing keys: {missing[:10]}", flush=True)
        print(f"[DEBUG] Unexpected keys: {unexpected[:10]}", flush=True)



    else:
        log(f"'{name}' using RANDOM initialization")

    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    log(f"Model '{name}' ready on {DEVICE}")
    return model


# extract features from a specific layer (decoder1 is the final decoder output)
def extract_features(model, x, layer="decoder1"):
    feats = {}
    def hook_fn(m, inp, out):
        feats["f"] = out
    h = getattr(model, layer).register_forward_hook(hook_fn)
    with torch.no_grad():
        _ = model(x)
    h.remove()
    return feats["f"]


# zero shot segmentation using model features + kmeans
def zero_shot_segment(model, img_np):
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
    x = torch.from_numpy(img_np[None, None]).to(DEVICE)

    f = extract_features(model, x)
    _, C, H, W, D = f.shape

    log(f"Extracted features shape={f.shape} "
    f"(C={f.shape[1]}, HWD={f.shape[2:]})")

    # mask zero values (padding)
    valid_mask = img_np > 0 + 1e-5 # add epsilon for numerical stability

    # log percentage of voxels masked out
    num_total = valid_mask.size
    num_valid = valid_mask.sum()
    num_masked = num_total - num_valid
    pct_masked = (num_masked / num_total) * 100.0
    log(f'Masked voxels (padding): {num_masked}/{num_total} ({pct_masked:.2f}%)')

    f_flat = f[0].permute(1,2,3,0).reshape(-1, C)
    valid_idx = valid_mask.reshape(-1)
    f_valid = f_flat[valid_idx].cpu().numpy()

    # kmeans clustering with k=2 (foreground vs background)
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=10).fit(f_valid)
    centers = kmeans.cluster_centers_

    d0 = np.linalg.norm(f_valid - centers[0], axis=1)
    d1 = np.linalg.norm(f_valid - centers[1], axis=1)

    n0 = (d0 < d1).sum()
    n1 = (d1 < d0).sum()
    fg = 0 if n0 < n1 else 1

    log(f"Cluster voxel counts: n0={n0}, n1={n1}, fg_cluster={fg}")

    dist_fg = d0 if fg == 0 else d1
    dist_bg = d1 if fg == 0 else d0

    score_full = np.full((H*W*D), -np.inf, dtype=np.float32)
    score_full[valid_idx] = (dist_bg - dist_fg)

    score = score_full.reshape(H, W, D)
    prob  = 1 / (1 + np.exp(-score))
    pred  = (score > 0).astype(np.float32)

    return prob.astype(np.float32), pred


# simple thresholding baseline
def threshold_segment(img_np, q=99.0):
    """
    Simple intensity-based thresholding baseline.
    Uses quantile thresholding on nonzero voxels.
    """
    img_norm = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)

    valid_mask = img_norm > 0
    vals = img_norm[valid_mask]

    if vals.size == 0:
        thr = 1.0
    else:
        thr = np.percentile(vals, q)

    prob = img_norm.copy()
    prob[~valid_mask] = 0.0

    pred = (img_norm >= thr).astype(np.float32)

    log(f"[THRESH] Quantile={q}, threshold={thr:.4f}, "
        f"foreground voxels={pred.sum()}")

    return prob.astype(np.float32), pred



# -----------------------------
# Main
# -----------------------------
def main(task_id: int, threshold_only: bool = False):

    # build flat patch list
    patch_list = []
    for dataset in DATASETS:
        files = sorted(
            f for f in glob.glob(os.path.join(DATA_ROOT, dataset, "*.nii.gz"))
            if not f.endswith("_label.nii.gz")
        )
        for f in files:
            patch_list.append((dataset, f))

    log(f"Discovered {len(patch_list)} total patches")

    if task_id >= len(patch_list):
        log(f"Task {task_id} out of range — exiting")
        return

    dataset, img_path = patch_list[task_id]
    log(f"Task {task_id} assigned patch: {dataset}/{os.path.basename(img_path)}")

    base = os.path.basename(img_path).replace(".nii.gz", "")

    # create results directory for this dataset
    results_root = os.path.join(OUT_ROOT, "results")
    os.makedirs(results_root, exist_ok=True)

    results_dir = os.path.join(results_root, dataset)
    os.makedirs(results_dir, exist_ok=True)

    img_np, affine = load_nifti(img_path)

    log(f"Loaded image shape={img_np.shape}, dtype={img_np.dtype}")
    log(f"Intensity range before norm: "
        f"[{img_np.min():.3f}, {img_np.max():.3f}]")


    # load all models once
    models = {}

    if not threshold_only:
        for name, cfg in MODELS.items():
            models[name] = build_model(
                ckpt_path=cfg["ckpt"],
                feature_size=cfg["feature_size"],
                name=name,
            )
    else:
        log("Using thresholding baseline only - skipping model loading")



    print(f"[Task {task_id}] Processing {dataset}/{base}", flush=True)

    # ---- model-based zero-shot segmentation ----
    if not threshold_only:
        log("Starting zero-shot segmentation inference...")
        for model_name, model in models.items():
            prob, pred = zero_shot_segment(model, img_np)

            nib.save(
                nib.Nifti1Image(pred, affine),
                os.path.join(results_dir, f"{base}_pred_{model_name}.nii.gz")
            )
            nib.save(
                nib.Nifti1Image(prob, affine),
                os.path.join(results_dir, f"{base}_prob_{model_name}.nii.gz")
            )

            log(f"Saved outputs for model '{model_name}' "
            f"→ {results_dir}/{base}_*_{model_name}.nii.gz")

    # ---- thresholding baseline ----
    log("Starting thresholding baseline inference...")
    prob_thr, pred_thr = threshold_segment(img_np)

    nib.save(
        nib.Nifti1Image(pred_thr, affine),
        os.path.join(results_dir, f"{base}_pred_threshold.nii.gz")
    )
    nib.save(
        nib.Nifti1Image(prob_thr, affine),
        os.path.join(results_dir, f"{base}_prob_threshold.nii.gz")
    )
    log(f"Saved outputs for thresholding baseline "
        f"→ {results_dir}/{base}_*_threshold.nii.gz")
    
    log(f"Task {task_id} completed successfully")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    parser.add_argument('--threshold-only', action='store_true', help='Use thresholding baseline instead of model-based segmentation')
    args = parser.parse_args()

    main(args.task_id, threshold_only=args.threshold_only)











