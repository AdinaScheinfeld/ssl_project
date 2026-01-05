# Complete 3D ResNet baseline using SAME MONAI transforms as SwinUNETR finetuning
# Fair comparison baseline

## UP TO HERE

"""
/home/ads4015/ssl_project/baselines/classification/resnet_baseline.py

Key points:
- Uses MONAI ResNet (resnet18 by default) for 3D volumes (96^3)
- Uses EXACT SAME transforms as SwinUNETR finetuning:
    * get_load_transforms
    * get_finetune_train_transforms -> get_classification_train_transforms
    * get_finetune_val_transforms -> get_classification_val_transforms
- Uses SAME fold JSONs (train/test), and SAME internal train/val split
- Implements early stopping (CLI-configurable)
- Logs metrics to Weights & Biases
- Writes metrics in SAME folder/CSV format as other baselines

This script is intentionally explicit (not Lightning) so behavior is transparent.
"""

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import random
import wandb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import nibabel as nib

from monai.networks.nets import resnet18
from monai.data import Dataset
from monai.transforms import Compose

from sklearn.metrics import accuracy_score, f1_score

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    Resized,
    ScaleIntensityRangePercentilesd,
    # SqueezeDimd,
    ToTensord
)

# -----------------------------
# Import FAIR transforms (same as SwinUNETR)
# -----------------------------
import sys
sys.path.append('/home/ads4015/ssl_project/src')
from all_datasets_transforms import (
    get_load_transforms,
)

# -----------------------------
# Label inference (same as PCA+LR)
# -----------------------------

def infer_label_from_path(path: str) -> str:
    p = str(path)
    base = os.path.basename(p)

    if "selma3d_finetune_patches" in p:
        for cname in (
            "amyloid_plaque_patches",
            "c_fos_positive_patches",
            "cell_nucleus_patches",
            "vessels_patches",
        ):
            if f"/{cname}/" in p:
                return cname

    if "VIP_ASLM_off" in base:
        return "VIP_ASLM_off"
    if "VIP_ASLM_on" in base:
        return "VIP_ASLM_on"
    if "TPH2" in base:
        return "TPH2"

    if "_cr_" in base:
        return "stain-CR"
    if "_lec_" in base:
        return "stain-LEC"
    if "_nn_" in base:
        return "stain-NN"
    if "_npy_" in base:
        return "stain-NPY"
    if "_yo_" in base:
        return "stain-YO"

    raise ValueError(f"Could not infer label from {path}")


# -----------------------------
# Transforms helper functions
# -----------------------------

# transform to clamp image intensity between 0-1
class ClampIntensityd(MapTransform):
    def __init__(self, keys, minv=0.0, maxv=1.0):
        super().__init__(keys)
        self.minv = minv
        self.maxv = maxv

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.clip(d[key], self.minv, self.maxv)
        return d

def get_classification_train_transforms():
    return Compose([
        # spatial augments — IMAGE ONLY
        RandFlipd(keys=["image"], spatial_axis=[0, 1, 2], prob=0.2),
        RandRotate90d(keys=["image"], prob=0.2, max_k=3),
        RandAffined(
            keys=["image"],
            rotate_range=(0.1, 0.1, 0.1),
            scale_range=(0.1, 0.1, 0.1),
            prob=0.2
        ),

        # intensity augments — IMAGE ONLY
        RandGaussianNoised(keys=["image"], prob=0.2, mean=0.0, std=0.02),
        RandGaussianSmoothd(keys=["image"], prob=0.2),
        RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.2),
        RandShiftIntensityd(keys=["image"], offsets=0.2, prob=0.2),
        ClampIntensityd(keys=["image"], minv=0.0, maxv=1.0),

        ToTensord(keys=["image"])
    ])


def get_classification_val_transforms():
    return Compose([
        ToTensord(keys=["image"])
    ])


# -----------------------------
# Dataset wrapper
# -----------------------------

class NiftiClassificationDataset(Dataset):
    def __init__(self, paths, class_to_idx, transforms):
        self.paths = list(paths)
        self.class_to_idx = class_to_idx
        self.transforms = transforms

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label_name = infer_label_from_path(path)
        y = self.class_to_idx[label_name]

        sample = {"image": path}
        sample = self.transforms(sample)
        x = sample["image"]  # tensor [C, D, H, W]

        return x, y, path


# -----------------------------
# Training / evaluation loops
# -----------------------------

def run_epoch(model, loader, optimizer=None, device="cuda", return_probs=False):
    train = optimizer is not None
    model.train(train)

    all_y, all_p = [], []
    all_probs = []
    all_paths = []
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    for x, y, paths in loader:
        x = x.to(device)
        y = y.to(device)

        if train:
            optimizer.zero_grad()

        logits = model(x)
        loss = criterion(logits, y)

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * x.size(0)

        preds = logits.argmax(dim=1)
        all_y.append(y.detach().cpu().numpy())
        all_p.append(preds.detach().cpu().numpy())

        if return_probs:
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.detach().cpu().numpy())
            all_paths.extend(list(paths))

    all_y = np.concatenate(all_y)
    all_p = np.concatenate(all_p)
    loss = total_loss / len(all_y)
    acc = accuracy_score(all_y, all_p)
    macro_f1 = f1_score(all_y, all_p, average="macro")

    if return_probs:
        all_probs = np.concatenate(all_probs, axis=0) if len(all_probs) else None
        return loss, acc, macro_f1, all_y, all_p, all_probs, all_paths

    return loss, acc, macro_f1


# -----------------------------
# Metrics helper functions
# -----------------------------

def make_tag(fold_id: int, ntr: int, ntest: int, fttr: int, ftval: int, seed: int) -> str:
    return f"resnet_cvfold{fold_id}_ntr{ntr}_ntest{ntest}_fttr{fttr}_ftval{ftval}_seed{seed}"

def metrics_from_counts(cm: np.ndarray):
    K = cm.shape[0]
    support = cm.sum(axis=1)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec  = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1   = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    acc = tp.sum() / max(1, cm.sum())
    macro_f1 = f1.mean() if K > 0 else 0.0
    return acc, macro_f1, prec, rec, f1, support


# -----------------------------
# Main experiment
# -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold_json", required=True)
    p.add_argument("--fold_id", type=int, required=True)
    p.add_argument("--output_root", required=True)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--use_wandb", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    with open(args.fold_json) as f:
        meta = json.load(f)

    classes = meta["classes"]
    class_to_idx = {c: i for i, c in enumerate(classes)}

    fold = meta["folds"][args.fold_id]
    train_paths = fold["train"]
    test_paths = fold["test"]

    ntr = len(train_paths)
    ntest = len(test_paths)

    # mimic fttr/ftval logic (same as pca_lr_baseline.py)
    val_percent = 0.2
    if ntr <= 1:
        ftval = 0
        fttr = ntr
    else:
        n_val = int(round(ntr * val_percent))
        if n_val < 1:
            n_val = 1
        if n_val > ntr - 1:
            n_val = ntr - 1
        ftval = n_val
        fttr = ntr - n_val

    tag = make_tag(args.fold_id, ntr, ntest, fttr, ftval, args.seed)
    print(f"[INFO] Running ResNet baseline for tag: {tag}", flush=True)

    metrics_root = Path(args.output_root) / "cls_metrics" / "classification" / tag
    metrics_root.mkdir(parents=True, exist_ok=True)

    epoch_csv = metrics_root / f"epoch_metrics_{tag}.csv"
    if not epoch_csv.exists():
        with open(epoch_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                "timestamp","epoch",
                "train_loss","train_acc","train_macro_f1",
                "val_loss","val_acc","val_macro_f1",
                "best_val_macro_f1","bad_epochs"
            ])


    # wandb logging
    run = None
    if args.use_wandb:
        run = wandb.init(
            project="selma3d_classification_baselines",
            name=f"resnet3d_fold{args.fold_id}_{Path(args.fold_json).stem}",
            config={
                "model": "resnet18_3d",
                "fold_id": args.fold_id,
                "json": Path(args.fold_json).name,
                "epochs": args.epochs,
                "patience": args.patience,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "seed": args.seed,
                "num_classes": len(classes),
            },
        )


    # internal train/val split (same logic as SwinUNETR finetune)
    train_paths = sorted(train_paths)
    tr_paths = train_paths[:fttr]
    va_paths = train_paths[fttr:fttr + ftval]

    # transforms
    load_tf = get_load_transforms(target_size=96)
    train_tf = Compose([load_tf, get_classification_train_transforms()])
    val_tf = Compose([load_tf, get_classification_val_transforms()])

    train_ds = NiftiClassificationDataset(tr_paths, class_to_idx, train_tf)
    val_ds = NiftiClassificationDataset(va_paths, class_to_idx, val_tf)
    test_ds = NiftiClassificationDataset(test_paths, class_to_idx, val_tf)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True, persistent_workers=(args.num_workers > 0)
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}", flush=True)

    # wandb logging
    if run is not None:
        wandb.config.update({
            "n_train": len(tr_paths),
            "n_val": len(va_paths),
            "n_test": len(test_paths),
        })


    model = resnet18(
        spatial_dims=3,
        n_input_channels=1,
        num_classes=len(classes)
    ).to(device)

    print(f"[INFO] Model has {sum(p.numel() for p in model.parameters())} parameters", flush=True)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    best_ckpt_path = metrics_root / f"checkpoint_best_{tag}.pt"
    last_ckpt_path = metrics_root / f"checkpoint_last_{tag}.pt"


    best_val = -np.inf
    bad_epochs = 0

    for epoch in range(args.epochs):
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, optimizer, device)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, None, device)

        # append epoch metrics CSV (NOT just header)
        with open(epoch_csv, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([
                datetime.now().isoformat(), epoch,
                tr_loss, tr_acc, tr_f1,
                va_loss, va_acc, va_f1,
                best_val, bad_epochs
            ])

        print(f"[Epoch {epoch}] train_loss={tr_loss:.4f} val_f1={va_f1:.4f}", flush=True)

        if run is not None:
            wandb.log({
                "epoch": epoch,

                "train_loss": tr_loss,
                "train_accuracy": tr_acc,
                "train_macro_f1": tr_f1,

                "val_loss": va_loss,
                "val_accuracy": va_acc,
                "val_macro_f1": va_f1,
            })


        if va_f1 > best_val:
            best_val = va_f1
            bad_epochs = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_macro_f1": float(best_val),
                    "tag": tag,
                    "classes": classes,
                    "args": vars(args),
                },
                best_ckpt_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("[INFO] Early stopping", flush=True)

                if run is not None:
                    wandb.log({
                        "early_stop_epoch": epoch,
                        "best_val_macro_f1": best_val,
                    })

                break

        # always save "last" checkpoint each epoch (after best_val is updated)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_macro_f1": float(best_val),
                "tag": tag,
                "classes": classes,
                "args": vars(args),
            },
            last_ckpt_path,
        )

    # load best ckpt (fallback to last if best was never saved)
    if not best_ckpt_path.exists():
        print("[WARN] Best checkpoint was never saved; using last checkpoint instead.", flush=True)
        best_ckpt_path = last_ckpt_path

    best_ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    te_loss, te_acc, te_f1, y_true, y_pred, probs, paths = run_epoch(
        model, test_loader, optimizer=None, device=device, return_probs=True
    )

    print(f"[TEST] acc={te_acc:.4f} macro_f1={te_f1:.4f}", flush=True)

    # confusion matrix
    K = len(classes)
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1

    acc2, macro_f12, prec, rec, f1, support = metrics_from_counts(cm)

    # 1) Predictions CSV (same format as PCA+LR)
    preds_csv = metrics_root / f"preds_{tag}.csv"
    with preds_csv.open("w") as f:
        f.write("filename,true_label,predicted_label,pred_idx,true_idx,prob_pred\n")
        for path, t_idx, p_idx, prob_vec in zip(paths, y_true, y_pred, probs):
            fname = os.path.basename(path)
            true_label = classes[int(t_idx)]
            pred_label = classes[int(p_idx)]
            prob_pred = float(prob_vec[int(p_idx)])
            f.write(f"{fname},{true_label},{pred_label},{int(p_idx)},{int(t_idx)},{prob_pred:.8f}\n")

    # 2) Confusion matrix CSV
    cm_csv = metrics_root / f"confusion_matrix_{tag}.csv"
    with cm_csv.open("w") as f:
        f.write("," + ",".join(classes) + "\n")
        for i, cname in enumerate(classes):
            row = ",".join(str(int(v)) for v in cm[i].tolist())
            f.write(f"{cname},{row}\n")

    # 3) Per-class metrics CSV
    per_class_csv = metrics_root / f"per_class_metrics_{tag}.csv"
    with per_class_csv.open("w") as f:
        f.write("class_name,support,precision,recall,f1_score\n")
        for i, cname in enumerate(classes):
            f.write(f"{cname},{int(support[i])},{prec[i]:.6f},{rec[i]:.6f},{f1[i]:.6f}\n")
        f.write(f"__OVERALL__,{int(cm.sum())},ACC={te_acc:.6f},MACRO_F1={te_f1:.6f},\n")

    print(f"[INFO] Saved predictions to: {preds_csv}", flush=True)
    print(f"[INFO] Saved confusion matrix to: {cm_csv}", flush=True)
    print(f"[INFO] Saved per-class metrics to: {per_class_csv}", flush=True)
    print(f"[INFO] Saved best checkpoint to: {best_ckpt_path}", flush=True)
    print(f"[INFO] Saved last checkpoint to: {last_ckpt_path}", flush=True)


    # wandb logging
    if run is not None:
        wandb.log({
            "test/loss": te_loss,
            "test/accuracy": te_acc,
            "test/macro_f1": te_f1,
        })
        run.finish()


if __name__ == "__main__":
    main()
