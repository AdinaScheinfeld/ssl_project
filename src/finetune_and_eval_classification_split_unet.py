#!/usr/bin/env python3
# finetune_and_eval_classification_split_unet.py - Fine-tuning and evaluation script for patch-based UNet classification

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import gc
import glob
import json
import numpy as np
import os
from pathlib import Path
import random
import sys
import time

import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader, get_worker_info

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# local imports
sys.path.append("/home/ads4015/ssl_project/models")
from patch_classification_module_unet import PatchClassificationModuleUNet

sys.path.append("/home/ads4015/ssl_project/data")
from patch_classification_dataset import PatchClassificationDataset

torch.set_float32_matmul_precision("medium")


# --- Helper Functions ---

def _seed_everything(seed):
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)

def _seed_worker(_):
    worker_info = get_worker_info()
    if worker_info is not None:
        base_seed = torch.initial_seed() % 2**31
        np.random.seed(base_seed + worker_info.id)
        random.seed(base_seed + worker_info.id)

def _format_hms(seconds):
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}s"

def list_class_dirs(root_dir):
    return sorted([d for d in root_dir.iterdir() if d.is_dir()])

def discover_samples(root_dir, class_filter=None, exclude=None, channel_substr="ALL", extra_class_globs=None):
    class_dirs = list_class_dirs(root_dir)
    all_names = [d.name for d in class_dirs]

    if class_filter is None or any(str(s).upper() == "ALL" for s in class_filter):
        names = all_names
    else:
        names = [n for n in all_names if n in set(class_filter)]

    if exclude:
        names = [n for n in names if n not in set(exclude)]

    extra_class_globs = extra_class_globs or []
    for spec in extra_class_globs:
        if ":" not in spec:
            print(f'[WARN] Ignoring invalid extra_class_glob spec "{spec}"', flush=True)
            continue
        cname, _ = spec.split(":", 1)
        cname = cname.strip()
        if cname and cname not in names:
            names.append(cname)

    names = sorted(names)
    name_to_idx = {name: idx for idx, name in enumerate(names)}

    samples = []

    subs = None
    s = str(channel_substr).strip() if channel_substr is not None else "ALL"
    if s and s.upper() != "ALL":
        subs = [t.strip().lower() for t in s.split(",") if t.strip()]

    for n in names:
        dd = root_dir / n
        if dd.is_dir():
            for p in sorted(dd.glob("*.nii*")):
                low = p.name.lower()
                if low.endswith("_label.nii") or low.endswith("_label.nii.gz"):
                    continue
                if subs is not None and not any(tok in low for tok in subs):
                    continue
                samples.append({"path": p, "label_idx": name_to_idx[n], "label_name": n})

    seen = {str(rec["path"].resolve()) for rec in samples}

    for spec in extra_class_globs:
        if ":" not in spec:
            continue
        cname, pattern = spec.split(":", 1)
        cname = cname.strip()
        pattern = pattern.strip()
        if not cname or not pattern:
            continue
        if cname not in name_to_idx:
            print(f'[WARN] Extra class glob specifies unknown class name "{cname}", skipping', flush=True)
            continue
        cidx = name_to_idx[cname]
        matches = sorted(glob.glob(pattern))
        if not matches:
            print(f'[WARN] extra_class_glob pattern matched no files for class "{cname}": {pattern}', flush=True)
            continue

        for p in matches:
            p_path = Path(p)
            low = p_path.name.lower()
            if subs is not None and not any(tok in low for tok in subs):
                continue
            resolved_path = str(p_path.resolve())
            if resolved_path in seen:
                continue
            samples.append({"path": p_path, "label_idx": cidx, "label_name": cname})
            seen.add(resolved_path)

    return samples, names

def split_train_val_and_test(samples, mode, seed, train_percent=None, eval_percent=None, train_count=None, eval_count=None):
    rng = random.Random(seed)
    arr = list(samples)
    rng.shuffle(arr)
    n = len(arr)
    if n == 0:
        return [], []

    if mode == "percent":
        assert train_percent is not None, "--train_percent required for percent mode"
        if eval_percent is None:
            eval_percent = 1.0 - train_percent
        n_train = max(1, min(n, int(round(n * train_percent))))
        n_eval = max(1, min(n - n_train, int(round(n * eval_percent))))
    elif mode == "count":
        if train_count is None and eval_count is None:
            raise ValueError("Either --train_count or --eval_count must be specified for count mode")
        if train_count is None:
            n_eval = min(n, int(eval_count))
            n_train = max(1, n - n_eval)
        elif eval_count is None:
            n_train = min(n, int(train_count))
            n_eval = max(1, n - n_train)
        else:
            n_train = min(n, int(train_count))
            n_eval = min(max(0, n - n_train), int(eval_count))
    else:
        raise ValueError(f"Invalid mode: {mode}")

    print(f"[INFO] Total samples: {n}, Train: {n_train}, Eval: {n_eval}", flush=True)
    return arr[:n_train], arr[n_train : n_train + n_eval]

def split_train_val(pool, val_percent=0.2, val_count=None, seed=100, min_train=1, min_val=1):
    n = len(pool)
    if n == 0:
        return [], []
    if n < (min_train + min_val):
        print(f"[WARN] Not enough samples for train+val: {n} < {min_train + min_val}", flush=True)
        return [], []

    rng = random.Random(seed + 1)
    arr = list(pool)
    rng.shuffle(arr)

    if val_count is not None:
        n_val = int(val_count)
    else:
        vp = 0.0 if val_percent is None else float(val_percent)
        vp = min(max(0.0, vp), 1.0)
        n_val = int(round(n * vp))

    n_val = max(min_val, min(n_val, n - min_train))
    n_train = n - n_val
    if n_train < min_train or n_val < min_val:
        print(f"[WARN] Not enough samples after clamp: train={n_train} or val={n_val}", flush=True)
        return [], []

    print(f"[INFO] Splitting train+val pool of {n} samples into Train: {n_train}, Val: {n_val}", flush=True)
    return arr[n_val:], arr[:n_val]

def compute_class_weights(train_samples, num_classes, scheme="inverse_freq", beta=0.9999):
    counts = [0] * num_classes
    for rec in train_samples:
        counts[rec["label_idx"]] += 1

    if scheme == "none":
        return [1.0] * num_classes
    elif scheme == "inverse_freq":
        return [1.0 / max(1, c) for c in counts]
    elif scheme == "effective_num":
        ws = []
        for c in counts:
            if c <= 0:
                ws.append(0.0)
            else:
                ws.append((1.0 - beta) / (1.0 - beta ** c))
        return ws
    raise ValueError(f"Invalid class weight scheme: {scheme}")

def metrics_from_counts(cm):
    K = cm.shape[0]
    support = cm.sum(axis=1)
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)
    acc = tp.sum() / max(1, cm.sum())
    macro_f1 = f1.mean() if K > 0 else 0.0
    return acc, macro_f1, prec, rec, f1, support


@dataclass
class RunOutputs:
    best_ckpt_path: str
    preds_csv: Path


def run_once(root_dir, args, device):

    # 1) discover samples
    all_samples, class_names = discover_samples(
        root_dir,
        class_filter=args.subtypes,
        exclude=args.exclude_subtypes,
        channel_substr=args.channel_substr,
        extra_class_globs=args.extra_class_globs,
    )
    K = len(class_names)
    if K == 0:
        print(f"[ERROR] No classes found in {root_dir} after filtering", flush=True)
        return RunOutputs(best_ckpt_path="", preds_csv=Path(""))
    print(f"[INFO] Discovered {len(all_samples)} samples across {K} classes: {class_names}", flush=True)

    # 2) split into train+val pool and test
    if args.fold_json and args.fold_id is not None:
        with open(args.fold_json, "r") as f:
            folds = json.load(f)
        folds_list = folds.get("folds", [])
        if not folds_list or args.fold_id < 0 or args.fold_id >= len(folds_list):
            raise ValueError(f"Invalid fold_id {args.fold_id} for fold_json with {len(folds_list)} folds")
        fold = folds_list[args.fold_id]
        train_val_set = set(map(str, fold["train"]))
        eval_set = set(map(str, fold["test"]))
        path_to_sample = {str(s["path"].resolve()): s for s in all_samples}
        train_pool = [path_to_sample[p] for p in train_val_set if p in path_to_sample]
        test_set = [path_to_sample[p] for p in eval_set if p in path_to_sample]
    else:
        train_pool, test_set = split_train_val_and_test(
            all_samples,
            mode=args.split_mode,
            seed=args.seed,
            train_percent=args.train_percent,
            eval_percent=args.eval_percent,
            train_count=args.train_count,
            eval_count=args.eval_count,
        )

    print(f"[INFO] Total={len(all_samples)} | Train pool={len(train_pool)} | Test={len(test_set)} | Classes={K}", flush=True)

    # 3) split train_pool into train/val
    train_set, val_set = split_train_val(
        train_pool,
        val_percent=args.val_percent,
        val_count=args.val_count,
        seed=args.seed,
        min_train=args.min_finetune_train,
        min_val=args.min_finetune_val,
    )
    if len(train_set) < args.min_finetune_train or len(val_set) < args.min_finetune_val:
        print(f"[ERROR] Not enough samples for fine-tuning: Train={len(train_set)} Val={len(val_set)}", flush=True)
        return RunOutputs(best_ckpt_path="", preds_csv=Path(""))

    # 4) class weights
    class_weights = compute_class_weights(train_set, K, scheme=args.class_weighting, beta=args.beta)
    print(f"[INFO] Class weights ({args.class_weighting}): {class_weights}", flush=True)

    # 5) loaders
    num_workers = min(args.num_workers, os.cpu_count() or args.num_workers)
    dataloader_kwargs = dict(
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=(num_workers > 0),
        worker_init_fn=_seed_worker,
    )

    train_dataset = PatchClassificationDataset(train_set, augment=True, channel_substr=args.channel_substr)
    val_dataset   = PatchClassificationDataset(val_set, augment=False, channel_substr=args.channel_substr)
    test_dataset  = PatchClassificationDataset(test_set, augment=False, channel_substr=args.channel_substr)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs)
    val_loader   = DataLoader(val_dataset, batch_size=1, shuffle=False, **dataloader_kwargs)
    test_loader  = DataLoader(test_dataset, batch_size=1, shuffle=False, **dataloader_kwargs)

    # 6) tags and logging
    if args.fold_json and args.fold_id is not None:
        split_tag = f"cvfold{args.fold_id}_ntr{len(train_pool)}_ntest{len(test_set)}"
    else:
        total = max(1, len(train_pool) + len(test_set))
        tr_pct = args.train_percent if args.train_percent is not None else (len(train_pool) / total)
        eval_pct = args.eval_percent if args.eval_percent is not None else (len(test_set) / total)
        split_tag = f"percent_tr{tr_pct:.2f}_eval{eval_pct:.2f}_ntr{len(train_pool)}_ntest{len(test_set)}"

    tag = f"{args.init_mode}_{split_tag}_fttr{len(train_set)}_ftval{len(val_set)}_seed{args.seed}"
    run_name = f"CLS_UNET_{tag}"

    wandb_logger = WandbLogger(project=args.wandb_project, name=run_name) if args.wandb_project else None

    # 7) model
    model = PatchClassificationModuleUNet(
        num_classes=K,
        lr=args.lr,
        weight_decay=args.weight_decay,
        pretrained_ckpt=args.pretrained_ckpt,
        class_names=class_names,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        linear_probe=args.linear_probe,
        init_mode=args.init_mode,
        in_channels=args.in_channels,
        class_weights=class_weights,
        # UNet arch
        unet_channels=tuple(int(x) for x in args.unet_channels.split(",")),
        unet_strides=tuple(int(x) for x in args.unet_strides.split(",")),
        unet_num_res_units=args.unet_num_res_units,
        unet_norm=args.unet_norm,
        img_size=(96, 96, 96),
    )

    # 8) trainer (checkpoint on val_accuracy)
    ckpt_dir = Path(args.ckpt_dir) / "classification_unet" / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    model_ckpt = ModelCheckpoint(
        monitor="val_accuracy",
        mode="max",
        save_top_k=1,
        dirpath=str(ckpt_dir),
        filename="finetune_cls_unet_best",
    )
    early_stopping = EarlyStopping(monitor="val_accuracy", mode="max", patience=args.early_stopping_patience)

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        logger=wandb_logger,
        callbacks=[model_ckpt, early_stopping],
        log_every_n_steps=11,
        deterministic=True,
    )
    trainer.fit(model, train_loader, val_loader)

    best_ckpt_path = model_ckpt.best_model_path or str(ckpt_dir / "finetune_cls_unet_last.ckpt")
    if not model_ckpt.best_model_path:
        trainer.save_checkpoint(best_ckpt_path)

    print(f"[INFO] Best model checkpoint saved at: {best_ckpt_path}", flush=True)

    # load best model
    # best_model = PatchClassificationModuleUNet.load_from_checkpoint(best_ckpt_path).to(device).eval()
    ckpt = torch.load(best_ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"], strict=False)
    best_model = model.to(device).eval()

    # 9) evaluate on test
    rows = []
    cm = np.zeros((K, K), dtype=np.int64)

    with torch.no_grad():
        for batch in test_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            logits = best_model(x)
            probs = torch.softmax(logits, dim=1)
            pred = torch.argmax(probs, dim=1)

            yi = int(y.item())
            pi = int(pred.item())
            cm[yi, pi] += 1

            rows.append(
                dict(
                    filename=batch["filename"][0],
                    true_label=class_names[yi],
                    predicted_label=class_names[pi],
                    pred_idx=pi,
                    true_idx=yi,
                    prob_pred=float(probs[0, pi].item()),
                )
            )

    acc, macro_f1, prec, rec, f1, support = metrics_from_counts(cm)
    print(f"[INFO] Test Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}", flush=True)

    if wandb_logger:
        wandb_logger.experiment.summary[f"{tag}_test_accuracy"] = acc
        wandb_logger.experiment.summary[f"{tag}_test_macro_f1"] = macro_f1

    # outputs
    output_dir = Path(args.metrics_root) / "classification_unet" / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    preds_csv = output_dir / f"preds_{tag}.csv"
    if rows:
        with open(preds_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"[INFO] Saved predictions CSV at: {preds_csv}", flush=True)
    else:
        with open(preds_csv, "w", newline="") as f:
            pass
        print(f"[INFO] No predictions; created empty CSV at: {preds_csv}", flush=True)

    cm_csv = output_dir / f"confusion_matrix_{tag}.csv"
    with open(cm_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([""] + class_names)
        for i, cname in enumerate(class_names):
            w.writerow([cname] + list(map(int, cm[i].tolist())))
    print(f"[INFO] Saved confusion matrix CSV at: {cm_csv}", flush=True)

    per_class_metrics_csv = output_dir / f"per_class_metrics_{tag}.csv"
    with open(per_class_metrics_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["class_name", "support", "precision", "recall", "f1_score"])
        for i, cname in enumerate(class_names):
            w.writerow([cname, int(support[i]), f"{prec[i]:.6f}", f"{rec[i]:.6f}", f"{f1[i]:.6f}"])
        w.writerow(["__OVERALL__", int(cm.sum()), f"ACC={acc:.6f}", f"MACRO_F1={macro_f1:.6f}", ""])

    # cleanup
    del train_loader, val_loader, test_loader
    gc.collect()

    return RunOutputs(best_ckpt_path=best_ckpt_path, preds_csv=preds_csv)


def parse_args():
    p = argparse.ArgumentParser(description="Finetune and evaluate a multiclass UNet patch classifier")

    # data
    p.add_argument("--root_dir", type=str, required=True)
    p.add_argument("--subtypes", type=str, nargs="*", default=["ALL"])
    p.add_argument("--exclude_subtypes", type=str, nargs="*", default=None)
    p.add_argument("--channel_substr", type=str, default="ALL")
    p.add_argument("--extra_class_globs", type=str, action="append", default=[])

    # split
    p.add_argument("--split_mode", type=str, choices=["percent", "count"], default="percent")
    p.add_argument("--train_percent", type=float, default=0.8)
    p.add_argument("--eval_percent", type=float, default=None)
    p.add_argument("--train_count", type=int, default=None)
    p.add_argument("--eval_count", type=int, default=None)
    p.add_argument("--seed", type=int, default=100)

    # finetune train/val
    p.add_argument("--val_percent", type=float, default=0.2)
    p.add_argument("--val_count", type=int, default=None)
    p.add_argument("--min_finetune_train", type=int, default=1)
    p.add_argument("--min_finetune_val", type=int, default=1)

    # training
    p.add_argument("--pretrained_ckpt", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--freeze_encoder_epochs", type=int, default=0)
    p.add_argument("--linear_probe", action="store_true")
    p.add_argument("--init_mode", type=str, choices=["pretrained", "random"], default="pretrained")
    p.add_argument("--in_channels", type=int, default=1)
    p.add_argument("--max_epochs", type=int, default=1000)
    p.add_argument("--early_stopping_patience", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=1)

    # UNet arch
    p.add_argument("--unet_channels", type=str, default="32,64,128,256")
    p.add_argument("--unet_strides", type=str, default="2,2,2,1")
    p.add_argument("--unet_num_res_units", type=int, default=2)
    p.add_argument("--unet_norm", type=str, default="BATCH")

    # imbalance
    p.add_argument("--class_weighting", type=str, choices=["none", "inverse_freq", "effective_num"], default="inverse_freq")
    p.add_argument("--beta", type=float, default=0.9999)

    # logging/outputs
    p.add_argument("--wandb_project", type=str, default="finetune_classification_unet")
    p.add_argument("--ckpt_dir", type=str, default="/ministorage/adina/finetune/ckpts")
    p.add_argument("--metrics_root", type=str, default="/ministorage/adina/finetune/metrics")

    # CV folds
    p.add_argument("--fold_json", type=str, default=None)
    p.add_argument("--fold_id", type=int, default=None)

    return p.parse_args()


def main():
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    args = parse_args()
    _seed_everything(args.seed)

    root_dir = Path(args.root_dir)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    t0 = time.perf_counter()
    print(f"[INFO] Starting finetuning+eval at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)
    print(f"[INFO] Using device: {device}", flush=True)
    print(f"[INFO] Data root: {root_dir}", flush=True)

    outputs = run_once(root_dir, args, device)
    if outputs.best_ckpt_path:
        print(f"[INFO] Finished. Best checkpoint: {outputs.best_ckpt_path}", flush=True)
        print(f"[INFO] Predictions CSV: {outputs.preds_csv}", flush=True)

    print(f"[INFO] End {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}, Total time: {_format_hms(time.perf_counter() - t0)}", flush=True)


if __name__ == "__main__":
    main()
