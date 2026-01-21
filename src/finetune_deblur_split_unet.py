#!/usr/bin/env python3
"""
/home/ads4015/ssl_project/src/finetune_deblur_split_unet.py

Finetune UNet deblurring model on sharp/blurred nifti patches using CV folds,
mirroring finetune_deblur_split.py (SwinUNETR) but swapping in DeblurModuleUNet.
"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import gc
import json
import nibabel as nib
import numpy as np
import os
from pathlib import Path
import random
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import torch
from torch.utils.data import DataLoader, get_worker_info

from monai.losses import SSIMLoss

# project imports
sys.path.append("/home/ads4015/ssl_project/models")
from deblur_module_unet import DeblurModuleUNet

sys.path.append("/home/ads4015/ssl_project/data")
from nifti_deblur_dataset import NiftiDeblurDataset, discover_nifti_deblur_items

torch.set_float32_matmul_precision("medium")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _seed_everything(seed):
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seed_worker(worker_id):
    worker_info = get_worker_info()
    if worker_info is not None:
        base_seed = torch.initial_seed() % 2**31
        np.random.seed(base_seed + worker_info.id)
        random.seed(base_seed + worker_info.id)


def _list_subtypes(root):
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def _save_nifti(vol, ref_nifti_path, output_path):
    v = vol.detach().cpu().numpy()
    if v.ndim == 4:
        v = v[0]
    v = np.clip(v.astype(np.float32), 0.0, 1.0)

    ref = nib.load(str(ref_nifti_path))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(v, affine=ref.affine, header=ref.header), str(output_path))


def _calculate_psnr(pred, target, eps=1e-8):
    mse = float(np.mean((pred - target) ** 2)) + eps
    return 10.0 * np.log10(1.0 / mse)


@torch.no_grad()
def _calculate_ssim_3d(pred_t: torch.Tensor, target_t: torch.Tensor) -> float:
    """
    pred_t / target_t: torch tensors on CPU or GPU, shape (D,H,W) or (1,1,D,H,W)
    Values assumed in [0,1]. Returns scalar SSIM in [0,1].
    Uses MONAI SSIMLoss: SSIM = 1 - SSIMLoss
    """
    if pred_t.ndim == 3:
        pred_t = pred_t[None, None, ...]
    if target_t.ndim == 3:
        target_t = target_t[None, None, ...]
    pred_t = pred_t.float()
    target_t = target_t.float()
    ssim_loss_fn = SSIMLoss(spatial_dims=3, data_range=1.0)
    ssim_loss = ssim_loss_fn(pred_t, target_t)
    ssim = 1.0 - ssim_loss
    return float(ssim.detach().cpu().item())


@dataclass
class RunOutputs:
    best_ckpt: str
    metrics_csv: Path
    preds_dir: Path


def run_one_subtype(subtype_dir, blurred_root, args, device):
    subtype = subtype_dir.name
    print(f"[INFO] Starting finetuning for subtype: {subtype}", flush=True)

    all_items = discover_nifti_deblur_items(
        sharp_class_dir=subtype_dir,
        blurred_root=blurred_root,
        channel_substr=args.channel_substr,
    )

    if len(all_items) == 0:
        print(f"[WARN] No paired items found for subtype: {subtype}. Skipping.", flush=True)
        return RunOutputs("", Path(""), Path(""))

    # ----- fold split -----
    if args.folds_json and args.fold_id is not None:
        with open(args.folds_json, "r") as f:
            j = json.load(f)

        folds = j["folds"] if "folds" in j else j.get(subtype, {}).get("folds", [])
        if not folds or args.fold_id < 0 or args.fold_id >= len(folds):
            raise ValueError(f"Invalid fold_id {args.fold_id} for folds JSON with {len(folds)} folds for {subtype}.")

        fold = folds[args.fold_id]
        train_paths = fold.get("train", [])
        test_paths = fold.get("test", fold.get("eval", []))

        if args.train_limit is not None and args.train_limit >= 0:
            train_paths = train_paths[: int(args.train_limit)]

        train_set = set(str(p) for p in train_paths)
        test_set = set(str(p) for p in test_paths)

        path_to_item = {str(item.sharp_image.resolve()): item for item in all_items}
        train_items = [path_to_item[p] for p in train_set if p in path_to_item]
        test_items = [path_to_item[p] for p in test_set if p in path_to_item]
    else:
        n = len(all_items)
        k = int(0.8 * n)
        train_items = all_items[:k]
        test_items = all_items[k:]

    # ----- finetune train/val split inside train pool -----
    val_count = args.val_count if args.val_count is not None else max(1, int(round(args.val_fraction * len(train_items))))

    rng = random.Random(args.seed + 1)
    train_items_shuffled = list(train_items)
    rng.shuffle(train_items_shuffled)

    ft_val_items = train_items_shuffled[:val_count]
    ft_train_items = train_items_shuffled[val_count:]

    if len(ft_train_items) < args.min_finetune_train or len(ft_val_items) < args.min_finetune_val:
        print(
            f"[WARN] Insufficient finetune train/val items for subtype: {subtype}. "
            f"{len(ft_train_items)}/{args.min_finetune_train} train and "
            f"{len(ft_val_items)}/{args.min_finetune_val} val items. Skipping.",
            flush=True,
        )
        return RunOutputs("", Path(""), Path(""))

    print(
        f"[INFO] Finetune train/val/test split for subtype {subtype}: "
        f"{len(ft_train_items)} / {len(ft_val_items)} / {len(test_items)} items.",
        flush=True,
    )

    ds_train = NiftiDeblurDataset(ft_train_items)
    ds_val = NiftiDeblurDataset(ft_val_items)
    ds_test = NiftiDeblurDataset(test_items)

    loader_kw = dict(
        num_workers=min(args.num_workers, os.cpu_count() or args.num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        worker_init_fn=_seed_worker,
    )
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, **loader_kw)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, **loader_kw)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, **loader_kw)

    # ----- tag + output dirs (match Swin script conventions) -----
    fold_tag = f"cvfold{args.fold_id}" if (args.folds_json is not None and args.fold_id is not None) else "nofold"
    limit_tag = f"trlm{args.train_limit}" if (args.train_limit is not None and args.train_limit >= 0) else "trlimALL"
    tag = (
        f"{fold_tag}_{limit_tag}_fttr{len(ft_train_items)}_ftval{len(ft_val_items)}_"
        f"tst{len(test_items)}_{subtype}_seed{args.seed}"
    )

    ckpt_dir = Path(args.ckpt_dir) / subtype / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"deblur_unet_{subtype}_{tag}"
    wandb_logger = WandbLogger(project=args.wandb_project, name=run_name)

    # ----- model -----
    model = DeblurModuleUNet(
        pretrained_ckpt_path=args.pretrained_ckpt_path,
        lr=args.lr,
        encoder_lr_mult=args.encoder_lr_mult,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        freeze_bn_stats=args.freeze_bn_stats,
        weight_decay=args.weight_decay,
        channels=tuple(int(x) for x in args.unet_channels.split(",")),
        strides=tuple(int(x) for x in args.unet_strides.split(",")),
        num_res_units=args.unet_num_res_units,
        norm=args.unet_norm,
        edge_weight=args.edge_weight,
        highfreq_weight=args.highfreq_weight,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best_ckpt",
        monitor="val_total_loss",
        mode="min",
        save_top_k=1,
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_total_loss",
        patience=args.early_stopping_patience,
        mode="min",
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        precision="bf16-mixed" if torch.cuda.is_available() else 32,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=1,
        deterministic=True,
    )

    trainer.fit(model, dl_train, dl_val)

    best_ckpt_path = checkpoint_callback.best_model_path or str(ckpt_dir / "best_ckpt.ckpt")
    if not checkpoint_callback.best_model_path:
        trainer.save_checkpoint(best_ckpt_path)
    print(f"[INFO] Best checkpoint saved at: {best_ckpt_path}", flush=True)

    # ----- evaluation -----
    print(f"[INFO] Starting evaluation on test set for subtype: {subtype}", flush=True)

    preds_dir = Path(args.preds_root) / subtype / tag / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best_model = DeblurModuleUNet.load_from_checkpoint(best_ckpt_path).to(device).eval()

    with torch.no_grad():
        for batch in dl_test:
            blurred_img = batch["input_vol"].to(device)
            sharp_img = batch["target_vol"].to(device)

            residual = best_model(blurred_img)
            output_deblurred_pred = torch.clamp(blurred_img + residual, 0.0, 1.0)

            fname = Path(batch["filename"][0])
            out_path = preds_dir / (fname.stem.replace(".nii", "") + "_deblur_pred.nii.gz")
            ref_nifti_path = subtype_dir / fname
            _save_nifti(output_deblurred_pred[0, 0], ref_nifti_path, out_path)

            psnr = _calculate_psnr(
                pred=output_deblurred_pred[0, 0].detach().cpu().numpy(),
                target=sharp_img[0, 0].detach().cpu().numpy(),
            )
            ssim = _calculate_ssim_3d(
                pred_t=output_deblurred_pred[0, 0],
                target_t=sharp_img[0, 0],
            )
            rows.append(
                {"subtype": subtype, 
                "filename": fname.name, 
                "psnr": f"{psnr:.4f}", 
                "ssim": f"{ssim:.4f}", 
                "pred_path": str(out_path)}
            )

            

    metrics_csv_path = preds_dir / "metrics_test.csv"
    with open(metrics_csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["subtype", "filename", "psnr", "ssim", "pred_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"[INFO] Test metrics saved at: {metrics_csv_path}", flush=True)

    if wandb_logger and rows:
        mean_psnr = float(np.mean([float(r["psnr"]) for r in rows]))
        wandb_logger.experiment.summary[f"{subtype}/{tag}/mean_test_psnr"] = mean_psnr
        print(f"[INFO] Mean test PSNR for subtype {subtype}: {mean_psnr:.4f}", flush=True)

    del dl_train, dl_val, dl_test
    del ds_train, ds_val, ds_test
    gc.collect()

    return RunOutputs(best_ckpt=best_ckpt_path, metrics_csv=metrics_csv_path, preds_dir=preds_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Finetune UNet Deblur Model on Nifti Data per Subtype")

    # data
    p.add_argument("--data_root", type=str, required=True, help="Root directory containing sharp subdirectories.")
    p.add_argument("--blur_root", type=str, required=True, help="Root directory containing blurred images.")
    p.add_argument("--subtypes", nargs="*", default=["ALL"], help='Use "ALL" to process all subtypes.')
    p.add_argument("--exclude_subtypes", nargs="*", default=[])
    p.add_argument("--channel_substr", type=str, default="ALL")
    p.add_argument("--folds_json", type=str, default=None)
    p.add_argument("--fold_id", type=int, default=None)
    p.add_argument("--train_limit", type=int, default=None)

    # finetune split
    p.add_argument("--val_fraction", type=float, default=0.2)
    p.add_argument("--val_count", type=int, default=None)
    p.add_argument("--min_finetune_train", type=int, default=1)
    p.add_argument("--min_finetune_val", type=int, default=1)

    # pretrained init
    p.add_argument("--pretrained_ckpt_path", type=str, default=None, help="Path to pretrained UNet checkpoint.")

    # training hyperparams
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--encoder_lr_mult", type=float, default=0.2)
    p.add_argument("--weight_decay", type=float, default=1e-5)
    p.add_argument("--freeze_encoder_epochs", type=int, default=0)
    p.add_argument("--freeze_bn_stats", type=int, default=0)
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--early_stopping_patience", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=1)
    p.add_argument("--seed", type=int, default=100)

    # UNet arch
    p.add_argument("--unet_channels", type=str, default="32,64,128,256")
    p.add_argument("--unet_strides", type=str, default="2,2,2,1")
    p.add_argument("--unet_num_res_units", type=int, default=2)
    p.add_argument("--unet_norm", type=str, default="BATCH")

    # loss weights
    p.add_argument("--edge_weight", type=float, default=0.1)
    p.add_argument("--highfreq_weight", type=float, default=0.1)

    # outputs/logging
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--preds_root", type=str, required=True)
    p.add_argument("--wandb_project", type=str, default="deblur_finetune_unet")

    return p.parse_args()


def main():
    args = parse_args()
    _seed_everything(args.seed)

    data_root = Path(args.data_root)
    blurred_root = Path(args.blur_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root not found: {data_root}")
    if not blurred_root.exists():
        raise FileNotFoundError(f"Blur root not found: {blurred_root}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if any(s.upper() == "ALL" for s in args.subtypes):
        subtypes = [s for s in _list_subtypes(data_root) if s not in set(args.exclude_subtypes)]
    else:
        subtypes = args.subtypes

    print(f"[INFO] Subtypes to process: {subtypes}", flush=True)
    print(f"[INFO] Using device: {device}", flush=True)
    print(f"[INFO] Starting UNet deblur finetuning at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)

    for subtype in subtypes:
        subtype_dir = data_root / subtype
        if not subtype_dir.exists():
            print(f"[WARN] Missing subtype dir: {subtype_dir}. Skipping.", flush=True)
            continue

        outputs = run_one_subtype(subtype_dir, blurred_root, args, device)
        if outputs.best_ckpt:
            print(
                f"[INFO] Completed finetuning for subtype: {subtype}. "
                f"Best checkpoint: {outputs.best_ckpt} "
                f"Metrics: {outputs.metrics_csv} "
                f"Preds: {outputs.preds_dir}",
                flush=True,
            )

    print(f"[INFO] Finished all finetuning at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", flush=True)


if __name__ == "__main__":
    main()
