# /home/ads4015/ssl_project/src/finetune_inpaint_split_unet.py
# Finetune UNet-backbone inpainting model (same dataset/splits as Swin version)

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
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, get_worker_info

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from monai.losses import SSIMLoss

sys.path.append("/home/ads4015/ssl_project/models")
from inpaint_module_unet import InpaintModuleUNet

sys.path.append("/home/ads4015/ssl_project/data")
from nifti_inpaint_dataset import NiftiInpaintDataset, discover_nifti_inpaint_items

torch.set_float32_matmul_precision("medium")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _seed_everything(seed):
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)


def _seed_worker(worker_id):
    worker_info = get_worker_info()
    if worker_info is not None:
        base_seed = torch.initial_seed() % 2**31
        np.random.seed(base_seed + worker_info.id)


def _list_subtypes(root):
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


def _parse_fixed_size_arg(size_str):
    if isinstance(size_str, (int, tuple, list)):
        return size_str
    s = str(size_str).strip()
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
        if len(parts) != 3:
            raise ValueError(f'Invalid fixed size: {size_str} (expected "16" or "16,16,16")')
        return tuple(int(p) for p in parts)
    return int(s)


def _save_nifti(vol, ref_nifti_path, out_path):
    vol_np = vol.detach().cpu().numpy().astype(np.float32)
    vol_np = np.clip(vol_np, 0.0, 1.0)
    try:
        ref = nib.load(str(ref_nifti_path))
        affine, header = ref.affine, ref.header
    except Exception:
        affine, header = np.eye(4), nib.Nifti1Header()
    nib.save(nib.Nifti1Image(vol_np, affine, header), str(out_path))


def _calc_psnr(pred, target, eps=1e-8):
    mse = np.mean((pred - target) ** 2) + eps
    return 10.0 * np.log10(1.0 / mse)


def _calc_ssim_torch(pred_t: torch.Tensor, target_t: torch.Tensor) -> float:
    """
    pred_t, target_t: torch tensors shaped (B,1,D,H,W), values in [0,1]
    Returns scalar SSIM in [0,1] (higher is better).
    """
    # SSIMLoss returns a loss (lower is better). In your LightningModule you log:
    #   ssim = 1 - loss_ssim
    # We mirror that convention here.
    ssim_loss_fn = SSIMLoss(spatial_dims=3, data_range=1.0)
    with torch.no_grad():
        loss = ssim_loss_fn(pred_t, target_t)
        ssim = (1.0 - loss).detach().float().cpu().item()
    return float(ssim)


def _feather_mask(mask, radius=1):
    m = mask
    for _ in range(max(1, int(radius))):
        m = F.avg_pool3d(m, kernel_size=3, stride=1, padding=1)
    return m.clamp(0.0, 1.0)


@dataclass
class RunOutputs:
    best_ckpt: str
    metrics_csv: Path
    preds_dir: Path


def run_one_subtype(subdir, args, device):
    subtype = subdir.name
    print(f"[INFO] Starting finetuning for subtype: {subtype}")

    all_items = discover_nifti_inpaint_items(subdir, channel_substr=args.channel_substr)

    # splits (same logic as Swin script)
    if args.folds_json and args.fold_id is not None:
        with open(args.folds_json, "r") as f:
            j = json.load(f)
        entry = j.get(subtype, {})
        folds = entry.get("folds", [])
        if not folds or args.fold_id < 0 or args.fold_id >= len(folds):
            raise ValueError(f"Invalid fold_id {args.fold_id} for subtype {subtype}")
        fold = folds[args.fold_id]
        train_set = set(map(str, fold.get("train", [])))
        test_list = fold.get("test", fold.get("eval", []))
        test_set = set(map(str, test_list))

        if args.train_limit is not None and args.train_limit >= 0:
            train_list = [p for p in fold.get("train", [])][: int(args.train_limit)]
            train_set = set(map(str, train_list))

        amap = {str(item.image.resolve()): item for item in all_items}
        train_items = [amap[p] for p in train_set if p in amap]
        test_items = [amap[p] for p in test_set if p in amap]
    else:
        n = len(all_items)
        k = int(0.8 * n)
        train_items, test_items = all_items[:k], all_items[k:]

    val_count = args.val_count if args.val_count is not None else max(1, int(round(args.val_percent * max(1, len(train_items)))))
    rng = random.Random(args.seed + 1)
    train_items_shuffled = list(train_items)
    rng.shuffle(train_items_shuffled)
    ft_val_items = train_items_shuffled[:val_count]
    ft_train_items = train_items_shuffled[val_count:]

    if len(ft_train_items) < args.min_finetune_train or len(ft_val_items) < args.min_finetune_val:
        print(f"[WARN] Skipping {subtype} insufficient finetune data: {len(ft_train_items)} train, {len(ft_val_items)} val")
        return RunOutputs("", Path(""), Path(""))

    print(f"[INFO] Finetune train/val/test sizes for {subtype}: {len(ft_train_items)}/{len(ft_val_items)}/{len(test_items)}")

    default_captions = {
        "amyloid_plaque_patches": "Bright, compact fluorescent deposits marking extracellular amyloid plaques in a cleared mouse brain.",
        "c_fos_positive_patches": "Small, sharply defined glowing nuclei indicating activity-dependent c-Fos expression in cleared tissue.",
        "cell_nucleus_patches": "Numerous small, round fluorescent dots—each dot a cell nucleus—forming a dense speckled 3D pattern.",
        "vessels_patches": "Interconnected tubular fluorescent strands tracing blood vessels from large channels to fine capillaries.",
    }

    ds_train = NiftiInpaintDataset(
        ft_train_items,
        captions_json=args.captions_json,
        default_caption_by_subtype=default_captions,
        mask_mode=args.mask_mode,
        mask_ratio=args.mask_ratio,
        mask_fixed_size=args.mask_fixed_size,
        num_mask_blocks=args.num_mask_blocks,
        augment=False,
        seed=args.seed + 1,
    )
    ds_val = NiftiInpaintDataset(
        ft_val_items,
        captions_json=args.captions_json,
        default_caption_by_subtype=default_captions,
        mask_mode=args.mask_mode,
        mask_ratio=args.mask_ratio,
        mask_fixed_size=args.mask_fixed_size,
        num_mask_blocks=args.num_mask_blocks,
        augment=False,
        seed=args.seed + 2,
    )
    ds_test = NiftiInpaintDataset(
        test_items,
        captions_json=args.captions_json,
        default_caption_by_subtype=default_captions,
        mask_mode=args.mask_mode,
        mask_ratio=args.mask_ratio_test,
        mask_fixed_size=args.mask_fixed_size_test,
        num_mask_blocks=args.num_mask_blocks_test,
        augment=False,
        seed=args.seed + 3,
    )

    loader_kw = dict(
        num_workers=min(args.num_workers, os.cpu_count() or args.num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        worker_init_fn=_seed_worker,
    )
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, **loader_kw)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, **loader_kw)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, **loader_kw)

    fold_tag = f"cvfold{args.fold_id}" if (args.folds_json is not None and args.fold_id is not None) else "nosplit"
    limit_tag = (f"trlm{args.train_limit}" if (args.train_limit is not None and args.train_limit >= 0) else "trlimALL")
    tag = f"{fold_tag}_{limit_tag}_fttr{len(ft_train_items)}_ftval{len(ft_val_items)}_tst{len(test_items)}_seed{args.seed}"
    run_name = f"inpaint_unet_{subtype}_{tag}"

    ckpt_dir = Path(args.ckpt_dir) / subtype / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    wandb_logger = WandbLogger(project=args.wandb_project, name=run_name)

    # ---- build module ----
    model = InpaintModuleUNet(
        pretrained_ckpt_path=args.pretrained_ckpt_path,
        lr=args.lr,
        encoder_lr_mult=args.encoder_lr_mult,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        l1_weight_masked=args.l1_weight_masked,
        l1_weight_global=args.l1_weight_global,
        weight_decay=args.weight_decay,
        text_cond=not args.disable_text_cond,
        text_dim=args.text_dim,
        text_backend=args.text_backend,
        clip_ckpt=args.clip_ckpt,
        unet_channels=args.unet_channels,
        unet_strides=args.unet_strides,
        unet_num_res_units=args.unet_num_res_units,
        unet_norm=args.unet_norm,
    )

    # ---- log text conditioning status ----
    if args.disable_text_cond:
        print("[INFO] Text conditioning: DISABLED", flush=True)
    else:
        print(
            f"[INFO] Text conditioning: ENABLED | backend={args.text_backend} | "
            f"text_dim={args.text_dim} | clip_ckpt={args.clip_ckpt}",
            flush=True,
        )

    # NOTE: we no longer rebuild the backbone here.
    # The architecture is created inside InpaintModuleUNet.__init__ from the args above,
    # so load_from_checkpoint() reconstructs the exact same UNet later.

    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename="best_ckpt",
        save_top_k=1,
        monitor="val_loss",
        mode="min",
    )
    early_stopping_callback = EarlyStopping(
        monitor="val_loss",
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

    best_ckpt = checkpoint_callback.best_model_path or str(ckpt_dir / "best_ckpt.ckpt")
    if not checkpoint_callback.best_model_path:
        trainer.save_checkpoint(best_ckpt)
    print(f"[INFO] Best checkpoint saved at: {best_ckpt}")

    # ---- test evaluation (save preds, compute PSNR in masked region) ----
    preds_root = Path(args.preds_root) if args.preds_root else Path(args.ckpt_dir)
    preds_dir = preds_root / subtype / tag / "preds"
    preds_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    best_model = InpaintModuleUNet.load_from_checkpoint(best_ckpt).to(device).eval()

    print(
        f"[INFO] Loaded model text_cond={best_model.text_cond} "
        f"text_backend={getattr(best_model, '_text_backend', None)}",
        flush=True,
    )

    with torch.no_grad():
        for batch in dl_test:
            masked_vol = batch["masked_vol"].to(device)
            mask = batch["mask"].to(device)
            target_vol = batch["target_vol"].to(device)

            t_emb = None
            if not args.disable_text_cond and "text" in batch:
                t_emb = best_model.text_encoder(batch["text"]).to(device)

            pred = torch.sigmoid(best_model(masked_vol, mask, t_emb))

            fname = Path(batch["filename"][0])

            mask_path = preds_dir / (fname.stem.replace(".nii", "") + "_mask.nii.gz")
            _save_nifti(mask[0, 0], ref_nifti_path=(subdir / fname), out_path=mask_path)

            masked_input_path = preds_dir / (fname.stem.replace(".nii", "") + "_masked_input.nii.gz")
            _save_nifti(masked_vol[0, 0], ref_nifti_path=(subdir / fname), out_path=masked_input_path)

            pred_path = preds_dir / (fname.stem.replace(".nii", "") + "_inpaint_pred.nii.gz")
            _save_nifti(pred[0, 0], ref_nifti_path=(subdir / fname), out_path=pred_path)

            if getattr(args, "feather_radius", 0) and args.feather_radius > 0:
                soft_mask = _feather_mask(mask, radius=args.feather_radius)
            else:
                soft_mask = mask

            composite = (masked_vol * (1 - soft_mask) + pred * soft_mask).clamp(0, 1)
            comp_path = preds_dir / (fname.stem.replace(".nii", "") + "_inpaint_composite.nii.gz")
            _save_nifti(composite[0, 0], ref_nifti_path=(subdir / fname), out_path=comp_path)

            # ---- metrics in masked region ----
            pred_mask_t = (pred * mask).clamp(0, 1)
            target_mask_t = (target_vol * mask).clamp(0, 1)

            psnr = _calc_psnr(
                pred_mask_t.detach().cpu().numpy(),
                target_mask_t.detach().cpu().numpy(),
            )
            ssim = _calc_ssim_torch(pred_mask_t, target_mask_t)

            rows.append(
                dict(
                    subtype=subtype,
                    filename=fname.name,
                    psnr_masked=f"{psnr:.4f}",
                    ssim_masked=f"{ssim:.4f}",
                    mask_path=str(mask_path),
                    masked_input_path=str(masked_input_path),
                    pred_path=str(pred_path),
                    composite_path=str(comp_path),
                )
            )

    metrics_csv = preds_dir / "metrics_test.csv"
    with open(metrics_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "subtype",
                "filename",
                "psnr_masked",
                "ssim_masked",
                "mask_path",
                "masked_input_path",
                "pred_path",
                "composite_path",
            ],
        )
        w.writeheader()
        w.writerows(rows)

    if wandb_logger and rows:
        mean_psnr = float(np.mean([float(r["psnr_masked"]) for r in rows]))
        mean_ssim = float(np.mean([float(r["ssim_masked"]) for r in rows]))
        wandb_logger.experiment.summary[f"{subtype}/{tag}/test_mean_psnr_masked"] = mean_psnr
        wandb_logger.experiment.summary[f"{subtype}/{tag}/test_mean_ssim_masked"] = mean_ssim

    del dl_train, dl_val, dl_test
    del ds_train, ds_val, ds_test
    gc.collect()

    return RunOutputs(best_ckpt=best_ckpt, metrics_csv=metrics_csv, preds_dir=preds_dir)


def parse_args():
    p = argparse.ArgumentParser(description="Finetune UNet inpainting model on NIfTI datasets with splits.")

    # data
    p.add_argument("--data_root", type=str, required=True)
    p.add_argument("--subtypes", nargs="*", default=["ALL"])
    p.add_argument("--exclude_subtypes", nargs="*", default=[])
    p.add_argument("--channel_substr", default="ALL")
    p.add_argument("--captions_json", type=str, default=None)

    # cross validation
    p.add_argument("--folds_json", type=str, default=None)
    p.add_argument("--fold_id", type=int, default=None)
    p.add_argument("--train_limit", type=int, default=None)

    # finetune train/val split
    p.add_argument("--val_percent", type=float, default=0.2)
    p.add_argument("--val_count", type=int, default=None)
    p.add_argument("--min_finetune_train", type=int, default=1)
    p.add_argument("--min_finetune_val", type=int, default=1)

    # model (inpaint)
    p.add_argument("--pretrained_ckpt_path", type=str, default=None)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--encoder_lr_mult", type=float, default=0.05)
    p.add_argument("--freeze_encoder_epochs", type=int, default=5)
    p.add_argument("--l1_weight_masked", type=float, default=1.0)
    p.add_argument("--l1_weight_global", type=float, default=0.1)
    p.add_argument("--weight_decay", type=float, default=1e-5)

    # masking
    p.add_argument("--mask_mode", type=str, default="ratio", choices=["ratio", "fixed_size"])
    p.add_argument("--mask_ratio", type=float, default=0.3)
    p.add_argument("--mask_ratio_test", type=float, default=0.3)
    p.add_argument("--mask_fixed_size", type=str, default="16")
    p.add_argument("--mask_fixed_size_test", type=str, default="16")
    p.add_argument("--num_mask_blocks", type=int, default=1)
    p.add_argument("--num_mask_blocks_test", type=int, default=1)

    # text conditioning
    p.add_argument("--disable_text_cond", action="store_true")
    p.add_argument("--text_dim", type=int, default=512)
    p.add_argument("--text_backend", type=str, default="dummy", choices=["dummy", "clip"])
    p.add_argument("--clip_ckpt", type=str, default=None)

    # UNet architecture sweep args
    p.add_argument("--unet_channels", type=str, default="32,64,128,256,512")
    p.add_argument("--unet_strides", type=str, default="2,2,2,1")
    p.add_argument("--unet_num_res_units", type=int, default=2)
    p.add_argument("--unet_norm", type=str, default="BATCH")

    # training
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--early_stopping_patience", type=int, default=50)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=100)

    # outputs
    p.add_argument("--wandb_project", type=str, default="selma3d_inpaint_unet")
    p.add_argument("--ckpt_dir", type=str, required=True)
    p.add_argument("--preds_root", type=str, required=True)
    p.add_argument("--feather_radius", type=int, default=0)

    return p.parse_args()


def main():
    args = parse_args()
    args.mask_fixed_size = _parse_fixed_size_arg(args.mask_fixed_size)
    args.mask_fixed_size_test = _parse_fixed_size_arg(args.mask_fixed_size_test)

    _seed_everything(args.seed)
    root = Path(args.data_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if any(s.upper() == "ALL" for s in args.subtypes):
        subtypes = [s for s in _list_subtypes(root) if s not in set(args.exclude_subtypes)]
    else:
        subtypes = args.subtypes

    print(f"[INFO] Subtypes to process: {subtypes}")
    print(f"[INFO] Device: {device}")
    print(f"[INFO] Starting finetuning at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    for subtype in subtypes:
        subdir = root / subtype
        if not subdir.exists():
            print(f"[WARN] Missing subtype dir, skipping: {subdir}")
            continue
        out = run_one_subtype(subdir, args, device)
        if out.best_ckpt:
            print(f"[INFO] Finished {subtype}: best={out.best_ckpt} metrics={out.metrics_csv} preds={out.preds_dir}")

    print(f"[INFO] Done at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
