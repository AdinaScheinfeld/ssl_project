# finetune_and_inference_split_unet.py - A script to finetune a model and perform inference in separate steps

# --- Setup ---

# imports
import argparse
import csv
import gc
from dataclasses import dataclass
from datetime import datetime
import json
import nibabel as nib
import numpy as np
import os
from pathlib import Path
import random
import sys
import time

import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info
import torch.multiprocessing as mp

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# local imports
sys.path.append('/home/ads4015/ssl_project/models')
from binary_segmentation_module_unet import BinarySegmentationModuleUnet

sys.path.append('/home/ads4015/ssl_project/data')
from nifti_pair_dataset import NiftiPairDataset

# set matmul precision
torch.set_float32_matmul_precision('medium')


# --- Functions ---


# *** Data Handling ***

# function to return all immediate subfolder names under root
def list_available_subtypes(root):
    return sorted([d.name for d in root.iterdir() if d.is_dir()])


# function to set seed
def _seed_everything(seed):
    pl.seed_everything(seed, workers=True)
    random.seed(seed)
    np.random.seed(seed)


# function to seed worker
def _seed_worker(_):
    info = get_worker_info()
    if info is not None:
        base_seed = torch.initial_seed() % 2**31
        random.seed(base_seed + info.id)
        np.random.seed(base_seed + info.id)


# dataclass to hold image-label pair paths
@dataclass
class Pair:
    image: Path
    label: Path


# function to get all image-label pairs in a class folder
def discover_pairs(class_dir, channel_substr='ch0'):

    # normalize channel filters
    substrings = None
    if channel_substr:
        s = str(channel_substr).strip()
        if s and s.upper() != 'ALL':
            substrings = [t.strip().lower() for t in s.split(',') if t.strip()]

    # list of pairs
    pairs = []

    # iterate over all files in class_dir
    for p in sorted(class_dir.glob('*.nii*')):

        lower = p.name.lower()

        # find images - images have channel_substr in their name and do not have '_label'
        if lower.endswith('_label.nii') or lower.endswith('_label.nii.gz'):
            continue

        # channel filter
        if substrings is not None and not any(sub in lower for sub in substrings):
            continue

        # construct label path by inserting '_label' before file extension
        suffix = ''.join(p.suffixes)  # handles .nii and .nii.gz
        base = p.name[:-len(suffix)]
        label = p.with_name(f'{base}_label{suffix}')

        # if label exists, add to pairs
        if label.exists():
            pairs.append(Pair(image=p, label=label))

    # return the list of pairs
    return pairs


# function to split image-label pairs into train, val, test sets
def split_pairs(pairs, mode, seed, train_percent=None, eval_percent=None, train_count=None, eval_count=None):

    # shuffle
    pairs = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(pairs)
    n = len(pairs)
    if n == 0:
        return [], []
    
    # split by percent
    if mode == 'percent':
        if train_percent is None:
            raise ValueError('--train_percent must be specified in percent mode')
        if eval_percent is None:
            eval_percent = 1 - train_percent
        if not (0.0 <= train_percent <= 1.0) or not (0.0 <= eval_percent <= 1.0):
            raise ValueError('train_percent must be in [0, 1] and eval_percent must be in [0, 1]')
        if train_percent + eval_percent > 1.0 + 1e-6:
            raise ValueError('train_percent + eval_percent must be <= 1')
        
        n_train = max(0, min(n, int(round(n * (train_percent)))))
        n_eval = max(0, min(n - n_train, int(round(n * (eval_percent)))))

    # split by count
    elif mode == 'count':
        if train_count is None and eval_count is None:
            raise ValueError('at least one of --train_count or --eval_count must be specified in count mode')
        if train_count is None:
            n_eval = min(n, int(eval_count))
            n_train = max(0, n - n_eval)
        elif eval_count is None:
            n_train = min(n, int(train_count))
            n_eval = max(0, n - n_train)
        else:
            n_train = min(n, int(train_count))
            n_eval = min(max(0, n - n_train), int(eval_count))
    else:
        raise ValueError('mode must be one of "percent" or "count"')
    
    train_pairs = pairs[:n_train]
    eval_pairs = pairs[n_train:n_train + n_eval]
    return train_pairs, eval_pairs


# *** Inference/Metrics ***

# predict logits
@torch.no_grad()
def predict_logits(model, x):
    model.eval()
    device = next(model.parameters()).device
    x = x.to(device)
    return model(x)

# predict binary mask
def dice_at_threshold(pred, target, threshold=0.5, epsilon=1e-8):

    if target.device != pred.device or target.dtype != pred.dtype:
        target = target.to(pred.device).to(pred.dtype)
    
    # pred, target: expected shapes (B, 1, D, H, W) or (1, 1, D, H, W)
    probs = torch.sigmoid(pred)
    pred_bin = (probs >= threshold).to(pred.dtype)
    intersection = (pred_bin * target).sum()
    denom = pred_bin.sum() + target.sum() + epsilon
    return float((2.0 * intersection / denom).item())


# function to save predictions as NIfTI files
def save_pred_nii(mask_bin, like_path, out_path):

    # mask_bin: expected shape (1, 1, D, H, W)
    vol = mask_bin.squeeze().detach().cpu().numpy().astype(np.uint8)
    try:
        like = nib.load(str(like_path))
        affine, header = like.affine, like.header
    except Exception:
        affine, header = np.eye(4), nib.Nifti1Header()
    nib.save(nib.Nifti1Image(vol, affine, header), str(out_path))

# function to save raw probability predictions as NIfTI files
def save_prob_nii(probs, like_path, out_path):

    # probs: expected shape (1, 1, D, H, W)
    vol = probs.squeeze().detach().cpu().numpy().astype(np.float32)
    try:
        like = nib.load(str(like_path))
        affine, header = like.affine, like.header
    except Exception:
        affine, header = np.eye(4), nib.Nifti1Header()
    nib.save(nib.Nifti1Image(vol, affine, header), str(out_path))


# *** Run per subtype ***

# function to get tags
def exp_tag(args, n_train, n_eval):

    # build tag for when using cv
    if getattr(args, 'folds_json', None) is not None and getattr(args, 'fold_id', None) is not None:
        return f'cvfold{args.fold_id}_ntr{n_train}_nev{n_eval}'

    # build tag for normal splits
    if args.mode == 'count':
        return f'count_tr{n_train}_ev{n_eval}'
    else:
        # tolerate None for train_percent or eval_percent by deriving from actual split sizes
        total = max(1, (n_train + n_eval))
        tr_pct = args.train_percent if args.train_percent is not None else (n_train / total)
        ev_pct = args.eval_percent if args.eval_percent is not None else (n_eval / total)
        return f'percent_tr{tr_pct:.2f}_ev{ev_pct:.2f}_ntr{n_train}_nev{n_eval}'
    

# functon to create suffix
def build_pred_suffix(tag):
    return f'_{tag}'


# function to format seconds as Hh MMm SSs
def _format_hms(seconds):
    seconds = int(round(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f'{h}:{m:02d}:{s:02d}s'


# function to split finetune training data into train and val sets
def split_train_val(pairs, val_percent=0.2, val_count=None, seed=100, min_train=1, min_val=1):

    # get pairs
    pairs = list(pairs)
    n = len(pairs)
    if len(pairs) == 0:
        return [], []
    
    # check minimums
    if n < (min_train + min_val):
        print(f'[WARN] Not enough samples ({n}) to satisfy min_train ({min_train}) + min_val ({min_val}), skipping finetuning')
        return [], []
    
    # shuffle
    rng = random.Random(seed + 1)  # different seed from main split
    rng.shuffle(pairs)

    # split
    if val_count is not None:
        n_val = int(val_count)

    else:
        # default to 20% val from finetune train set
        val_percent = 0.0 if val_percent is None else float(val_percent)
        val_percent = min(max(0.0, val_percent), 1.0)
        n_val = int(round(n * val_percent))

    # clamp n_val to ensure at least min_train and min_val
    n_val = max(min_val, min(n_val, n - min_train))
    n_train = n - n_val

    # final safety check
    if n_train < min_train or n_val < min_val:
        print(f'[WARN] After clamping, not enough samples ({n}) to satisfy min_train ({min_train}) + min_val ({min_val}), skipping finetuning')
        return [], []
    
    # split
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    return train_pairs, val_pairs


@dataclass
class RunOutputs:
    best_ckpt: str
    metrics_csv: Path
    preds_dir: Path


# function to run finetuning and inference for one subtype
def run_for_subtype(subtype_dir, args, device):

    # get pairs
    subtype = subtype_dir.name
    all_pairs = discover_pairs(subtype_dir, channel_substr=args.channel_substr)

    # use specified folds (for cross validation) if available
    use_folds = (args.folds_json is not None and args.fold_id is not None)
    if use_folds:
        with open(args.folds_json, 'r') as f:
            folds = json.load(f)
        entry = folds.get(subtype, {})
        fold_list = entry.get('folds', [])
        if not fold_list or args.fold_id < 0 or args.fold_id >= len(fold_list):
            raise ValueError(f'Invalid fold_id {args.fold_id} for subtype {subtype} with folds: {fold_list}')
        
        fold = fold_list[args.fold_id]
        train_set = set(map(str, fold.get('train', [])))
        eval_set = set(map(str, fold.get('eval', [])))
        _map = {str(p.image): p for p in all_pairs}
        train_pairs = [_map[s] for s in train_set if s in _map]
        eval_pairs = [_map[s] for s in eval_set if s in _map]

        # cap train pool
        if args.train_limit is not None and args.train_limit >= 0:
            train_pairs = train_pairs[:min(len(train_pairs), int(args.train_limit))]


        # debugging
        print(f"[DEBUG] fold_id={args.fold_id} | #all_pairs={len(all_pairs)}", flush=True)
        print(f"[DEBUG] first_pair_example={all_pairs[0].image if all_pairs else 'NONE'}", flush=True)
        print(f"[DEBUG] fold keys={list(fold.keys())}", flush=True)
        print(f"[DEBUG] sample fold train[0:1]={fold.get('train', [])[:1]}", flush=True)
        print(f"[DEBUG] sample fold eval[0:1]={fold.get('eval', [])[:1]}", flush=True)

    # otherwise, do random split
    else:
        train_pairs, eval_pairs = split_pairs(
            all_pairs,
            mode=args.mode,
            seed=args.seed,
            train_percent=args.train_percent,
            eval_percent=args.eval_percent,
            train_count=args.train_count,
            eval_count=args.eval_count,
        )

    print(f'[INFO] {subtype}: Found {len(all_pairs)} pairs -> {len(train_pairs)} train, {len(eval_pairs)} test', flush=True)
    if len(eval_pairs) == 0:
        print(f'[WARN] {subtype}: Skipping due to no train or eval data', flush=True)
        return RunOutputs(best_ckpt='', metrics_csv=Path(''), preds_dir=Path(''))


    # dataset and dataloaders

    # split finetune pool into train/val sets
    train_core, val_pairs = split_train_val(
        train_pairs, 
        val_percent=args.val_percent, 
        val_count=args.val_count, 
        seed=args.seed,
        min_train=args.min_finetune_train,
        min_val=args.min_finetune_eval
    )

    # ensure sufficient data
    if len(train_core) < args.min_finetune_train or len(val_pairs) < args.min_finetune_eval:
        print(f'[WARN] {subtype}: Not enough finetune data after train/val split (train: {len(train_core)}, val: {len(val_pairs)}), skipping finetuning', flush=True)
        return RunOutputs(best_ckpt='', metrics_csv=Path(''), preds_dir=Path(''))

    # test set (never seen during model selection)
    test_dataset = NiftiPairDataset(eval_pairs, augment=False)
    num_workers = min(args.num_workers, os.cpu_count() or args.num_workers)
    # persistent_workers=False helps prevent leaked semaphores on teardown
    loader_kw = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=False, worker_init_fn=_seed_worker)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, **loader_kw)

    train_dataset = NiftiPairDataset(train_core, augment=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **loader_kw)
    
    val_dataset = NiftiPairDataset(val_pairs, augment=False)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, **loader_kw)

    # tagging and logging
    fold_tag = (f'fold{args.fold_id}' if use_folds else 'nofold')
    limit_tag = (f'trlim{args.train_limit}' if (args.train_limit is not None and args.train_limit >= 0) else 'trlimALL')
    split_tag = exp_tag(args, n_train=len(train_pairs), n_eval=len(eval_pairs))
    tag = f'{split_tag}_fttr{len(train_core)}_ftval{len(val_pairs)}_{fold_tag}_{limit_tag}_seed{args.seed}'
    run_name = f'{subtype}_{tag}_seed{args.seed}'
    wandb_logger = WandbLogger(project=args.wandb_project, name=run_name) if args.wandb_project else None

    # model
    model = BinarySegmentationModuleUnet(
        pretrained_ckpt=args.pretrained_ckpt,
        lr=args.lr,
        unet_channels=args.unet_channels,
        unet_strides=args.unet_strides,
        unet_num_res_units=args.unet_num_res_units,
        unet_norm=args.unet_norm,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        encoder_lr_mult=args.encoder_lr_mult,
        loss_name=args.loss_name
    )

    # checkpoint directory
    ckpt_dir = Path(args.ckpt_dir) / subtype / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # train
    # model checkpoint callback
    model_ckpt = ModelCheckpoint(
        monitor='val_dice_050', 
        mode='max', 
        save_top_k=1, 
        save_last=True,
        dirpath=str(ckpt_dir), 
        filename='finetune_split_best'
    )

    # early stopping callback
    early_stopping = EarlyStopping(
        monitor='val_dice_050', mode='max', patience=args.early_stopping_patience
    )

    # trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices = 1,
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        logger=wandb_logger,
        callbacks=[model_ckpt, early_stopping],
        log_every_n_steps=1,
        deterministic=True
    )
    trainer.fit(model, train_loader, val_loader)

    # --- resolve checkpoint paths ---
    best_ckpt = model_ckpt.best_model_path or (model.best_ckpt or '')
    last_ckpt = str(ckpt_dir / 'last.ckpt')  # produced by save_last=True

    # safety: ensure last exists
    if not os.path.exists(last_ckpt):
        trainer.save_checkpoint(last_ckpt)

    # safety: if best didn't get produced for some reason, fall back to last
    if not best_ckpt or not os.path.exists(best_ckpt):
        best_ckpt = last_ckpt
        print(f'[WARN] {subtype}: Best checkpoint not found, falling back to last checkpoint: {best_ckpt}', flush=True)

    # choose checkpoint for inference
    infer_ckpt = best_ckpt if args.infer_ckpt == 'best' else last_ckpt
    print(f'[INFO] {subtype}: Using infer_ckpt={infer_ckpt} (infer_ckpt flag = {args.infer_ckpt})', flush=True)

    # load chosen model for eval/inference
    infer_model = BinarySegmentationModuleUnet.load_from_checkpoint(infer_ckpt).to(device).eval()


    # choose output root (pred_root if provided, else fallback to data root)
    out_root = Path(args.preds_root) if getattr(args, 'preds_root', None) else Path(args.root)
    # pretty name for subtype folder in outputs
    pretty_subtype = getattr(args, 'preds_subtype', None) or subtype
    preds_dir = out_root / pretty_subtype / tag / 'preds'
    preds_dir.mkdir(parents=True, exist_ok=True)
    pred_suffix = build_pred_suffix(tag)

    # eval on test set
    rows = []
    for batch in test_loader:
        x = batch['image'] # (1, 1, D, H, W)
        y = batch['label'] # (1, 1, D, H, W)
        fname = Path(batch['filename'][0])

        # logits and probabilities
        logits = predict_logits(infer_model, x)  # (1, 1, D, H, W)
        probs = torch.sigmoid(logits) # (1, 1, D, H, W)
        dice_050 = dice_at_threshold(logits, y, threshold=0.5)

        # save preds (binary mask at 0.5 threshold and raw probabilities)
        mask_bin = (probs >= 0.5).to(torch.uint8)
        base_stem = fname.stem.replace('.nii', '').replace('.gz', '')
        mask_pred_path = preds_dir / f'{base_stem}_pred{pred_suffix}.nii.gz'
        prob_pred_path = preds_dir / f'{base_stem}_prob{pred_suffix}.nii.gz'
        save_pred_nii(mask_bin, like_path=fname, out_path=mask_pred_path)
        save_prob_nii(probs, like_path=fname, out_path=prob_pred_path)

        # log
        rows.append({
            'subtype': subtype,
            'image_path': str(fname),
            'filename': str(fname.name),
            'dice_050': f'{dice_050:.6f}',
            'pred_path': str(mask_pred_path),
            'prob_path': str(prob_pred_path)
        })
        print(f'[INFO] {subtype}: Eval {fname.name} -> Dice@0.5: {dice_050:.6f}', flush=True)

    # compute mean and append an average line at the bottom of the csv
    metrics_csv = preds_dir / f'metrics_test{pred_suffix}.csv'
    if rows:
        mean_dice = float(np.mean([float(r['dice_050']) for r in rows if r['filename'] != 'MEAN']))
        rows.append({'subtype': subtype,
                     'filename': 'MEAN',
                     'image_path': '',
                     'dice_050': f'{mean_dice:.6f}',
                     'pred_path': '',
                     'prob_path': ''})
        print(f'[INFO] {subtype}: Eval mean Dice@0.5 over {len(rows)-1} samples: {mean_dice:.6f}', flush=True)
        if wandb_logger:
            wandb_logger.experiment.summary[f'{subtype}/{tag}/test_mean_dice_050'] = mean_dice

    # save metrics CSV
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subtype', 'filename', 'image_path', 'dice_050', 'pred_path', 'prob_path'])
        writer.writeheader()
        writer.writerows(rows)

    # summary
    if rows:
        mean_dice = float(np.mean([float(r['dice_050']) for r in rows if r['filename'] != 'MEAN']))
        print(f'[INFO] {subtype}: Eval mean Dice@0.5 over {len(rows)-1} samples: {mean_dice:.6f}', flush=True)
        if wandb_logger:
            wandb_logger.experiment.summary[f'{subtype}/{tag}/test_mean_dice_050'] = mean_dice

    # cleanup
    del train_loader, val_loader, test_loader
    del train_dataset, val_dataset, test_dataset
    gc.collect()

    # return outputs
    return RunOutputs(best_ckpt=best_ckpt, metrics_csv=metrics_csv, preds_dir=preds_dir)


# *** CLI ***

def parse_args():

    # parser
    parser = argparse.ArgumentParser(description='Finetune a model and perform inference in separate steps')

    # data
    parser.add_argument('--root', type=str, required=True, help='Root directory containing subtype subfolders with nifti pairs (ex: amyloid_plaque_patches, ...)')
    parser.add_argument('--subtypes', nargs='*', default=['ALL'], help='Subtype folder to process; use "ALL" to process all subfolders (default: ALL)')
    parser.add_argument('--exclude_subtypes', nargs='*', default=[], help='Subtype folder names to exclude when using ALL (default: none)')
    parser.add_argument('--channel_substr', type=str, default='ch0', help='Substring to identify image channels: substring (ex: "ch0"), comma-sep list (ex: "ch0,ch1"), or "ALL" to include all channels (default: ch0)')

    # split
    parser.add_argument('--mode', type=str, choices=['percent', 'count'], default='percent', help='Data splitting mode: "percent" to specify percentages, "count" to specify exact counts')
    parser.add_argument('--train_percent', type=float, default=None, help='Fraction of data to use for training (only in percent mode, default: None)')
    parser.add_argument('--eval_percent', type=float, default=None, help='Fraction of data to use for eval/validation (only in percent mode, default: None, uses remaining data)')
    parser.add_argument('--train_count', type=int, default=None, help='Exact number of samples to use for training (only in count mode, default: None)')
    parser.add_argument('--eval_count', type=int, default=None, help='Exact number of samples to use for eval/validation (only in count mode, default: None, uses remaining data)')
    parser.add_argument('--seed', type=int, default=100, help='Random seed for data shuffling (default: 100)')

    # validation split within the finetune pool
    parser.add_argument('--val_percent', type=float, default=0.2, help='Fraction of finetune training data to use for validation (default: 0.2); ignored if val_count is set)')
    parser.add_argument('--val_count', type=int, default=None, help='Exact number of samples to use for validation from finetune pool (default: None)')

    # training
    parser.add_argument('--pretrained_ckpt', type=str, default=None, help='Path to pretrained model checkpoint for finetuning (.ckpt)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of training epochs (default: 1000)')
    parser.add_argument('--freeze_encoder_epochs', type=int, default=5, help='Number of initial epochs to freeze the encoder (default: 5)')
    parser.add_argument('--encoder_lr_mult', type=float, default=0.05, help='Learning rate multiplier for encoder layers (default: 0.05)')
    parser.add_argument('--loss_name', type=str, choices=['dicece', 'dicefocal'], default='dicece', help='Loss function to use (default: dicece)')
    parser.add_argument('--early_stopping_patience', type=int, default=45, help='Early stopping patience epochs (default: 45)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader worker processes (default: 4)')
    parser.add_argument('--min_finetune_train', type=int, default=1, help='Minimum number of finetune training samples required to run finetuning (default: 1)')
    parser.add_argument('--min_finetune_eval', type=int, default=1, help='Minimum number of finetune eval samples required to run finetuning (default: 1)')
    parser.add_argument('--unet_channels', type=str, default='32,64,128,256,512', help='Comma-separated list of U-Net channels (default: 32,64,128,256,512)')
    parser.add_argument('--unet_strides', type=str, default='2,2,2,2', help='Comma-separated list of U-Net strides (default: 2,2,2,2)')
    parser.add_argument('--unet_num_res_units', type=int, default=2, help='Number of residual units in each U-Net block (default: 2)')
    parser.add_argument('--unet_norm', type=str, default='INSTANCE', help='Normalization method for U-Net (default: INSTANCE)')

    # logging/output
    parser.add_argument('--wandb_project', type=str, default='finetune', help='Wandb project name for logging (default: finetune)')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Output directory to save checkpoints, predictions, and metrics')

    # cross validation
    parser.add_argument('--preds_root', type=str, required=True, help='Root directory to save predictions')
    parser.add_argument('--preds_subtype', type=str, default=None, help='Pretty name for subtype folder in outputs (default: same as subtype)')
    parser.add_argument('--folds_json', type=str, default=None, help='Path to JSON file defining cross-validation folds (default: None)')
    parser.add_argument('--fold_id', type=int, default=None, help='Fold ID to use from folds_json (0-based index, default: None)')
    parser.add_argument('--train_limit', type=int, default=None, help='Limit the number of training samples to this number (default: None, meaning no limit)')

    # inference checkpoint selection
    parser.add_argument('--infer_ckpt', type=str, choices=['best', 'last'], default='best', help='Which checkpoint to use for inference: "best" (default) or "last"')

    # parse
    args = parser.parse_args()

    # parse unet channels and strides
    args.unet_channels = tuple(int(x) for x in args.unet_channels.split(','))
    args.unet_strides = tuple(int(x) for x in args.unet_strides.split(','))

    # safety check (ensure len of channels = len of strides + 1)
    if len(args.unet_channels) != len(args.unet_strides) + 1:
        raise ValueError(f'Unet config error: number of channels ({len(args.unet_channels)}) must be one more than number of strides ({len(args.unet_strides)})')

    return args


# --- Main ---

# main
def main():

    # set multiprocessing start method
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    # setup
    args = parse_args()
    _seed_everything(args.seed)
    root = Path(args.root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    t0 = time.perf_counter()
    print(f'[INFO] Using device: {device}', flush=True)
    print(f'[INFO] Start at {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")}', flush=True)
    print(f'[INFO] Root: {root}', flush=True)

    # determine subtypes to process
    if any(s.upper() == 'ALL' for s in args.subtypes):
        selected_subtypes = list_available_subtypes(root)
        if args.exclude_subtypes:
            selected_subtypes = [s for s in selected_subtypes if s not in args.exclude_subtypes]
    else:
        selected_subtypes = args.subtypes

    print(f'[INFO] Subtypes to process: {selected_subtypes}', flush=True)
    print(f'[INFO] Mode: {args.mode}', flush=True)

    # process each subtype
    for subtype in selected_subtypes:
        subdir = root / subtype
        if not subdir.exists():
            print(f'[WARN] Subtype directory not found: {subdir}, skipping...', flush=True)
            continue
        out = run_for_subtype(subdir, args, device)
        if out.best_ckpt:
            print(f'[INFO] {subdir.name}: best_ckpt={out.best_ckpt} | metrics={out.metrics_csv} | preds_dir={out.preds_dir}', flush=True)

    print(f'[INFO] Finished at {datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")}', flush=True)
    dt = time.perf_counter() - t0
    print(f'[INFO] Total runtime: {_format_hms(dt)} ({dt:.2f} seconds)', flush=True)


# --- Entry Point ---

if __name__ == '__main__':
    main()







