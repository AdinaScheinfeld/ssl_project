# finetune_and_inference_split.py - A script to finetune a model and perform inference in separate steps

# --- Setup ---

# imports
import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import nibabel as nib
import numpy as np
import os
from pathlib import Path
import random
import sys

import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info
import torch.multiprocessing as mp

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

# local imports
sys.path.append('/home/ads4015/ssl_project/models')
from binary_segmentation_module import BinarySegmentationModule

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

    # list of pairs
    pairs = []

    # iterate over all files in class_dir
    for p in sorted(class_dir.glob('*.nii*')):

        name = p.name
        lower = name.lower()

        # find images - images have channel_substr in their name and do not have '_label'
        if lower.endswith('_label.nii') or lower.endswith('_label.nii.gz'):
            continue
        if channel_substr not in lower:
            continue

        # construct label path by inserting '_label' before file extension
        suffix = ''.join(p.suffixes)  # handles .nii and .nii.gz
        base = name[:-len(suffix)]
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
        if train_percent + eval_percent > 1:
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


# *** Run per subtype ***

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
    train_pairs, eval_pairs = split_pairs(
        all_pairs,
        mode=args.mode,
        seed=args.seed,
        train_percent=args.train_percent,
        eval_percent=args.eval_percent,
        train_count=args.train_count,
        eval_count=args.eval_count,
    )

    print(f'[INFO] {subtype}: Found {len(all_pairs)} pairs -> {len(train_pairs)} train, {len(eval_pairs)} eval', flush=True)
    if len(train_pairs) == 0 or len(eval_pairs) == 0:
        print(f'[WARN] {subtype}: Skipping due to no train or eval data', flush=True)
        return RunOutputs(best_ckpt='', metrics_csv=Path(''), preds_dir=Path(''))

    # create datasets and dataloaders
    train_dataset = NiftiPairDataset(train_pairs, augment=True)
    eval_dataset = NiftiPairDataset(eval_pairs, augment=False)

    num_workers = min(args.num_workers, os.cpu_count() or args.num_workers)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=False, 
                              worker_init_fn=_seed_worker)
    eval_loader = DataLoader(eval_dataset, batch_size=1, shuffle=False, 
                             num_workers=num_workers, pin_memory=True, persistent_workers=False, 
                             worker_init_fn=_seed_worker)
    
    # model and logger
    run_name = f'{subtype}_seed{args.seed}_{args.mode}'
    effective_eval = args.eval_percent if args.eval_percent is not None else (1.0 - args.train_percent)
    if args.mode == 'percent':
        run_name += f'_tr{args.train_percent:.2f}_ev{effective_eval:.2f}'
    else:
        run_name += f'_tr{args.train_count}_ev{args.eval_count}'

    # logger
    wandb_logger = WandbLogger(project=args.wandb_project, name=run_name) if args.wandb_project else None

    # model
    model = BinarySegmentationModule(
        pretrained_ckpt=args.pretrained_ckpt,
        lr=args.lr,
        feature_size=args.feature_size,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        encoder_lr_mult=args.encoder_lr_mult,
        loss_name=args.loss_name
    )

    # checkpoint directory
    ckpt_dir = Path(args.output_dir) / subtype / 'checkpoints'
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # model checkpoint callback
    model_ckpt = ModelCheckpoint(
        monitor='val_dice_050', mode='max', save_top_k=1, dirpath=str(ckpt_dir), filename='finetune_split_best'
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
    )
    trainer.fit(model, train_loader, eval_loader)

    # resolve best checkpoint path or fallback to last ckpt
    best_ckpt = model_ckpt.best_model_path or (model.best_ckpt or '')
    if not best_ckpt:
        best_ckpt = str(ckpt_dir / 'finetune_split_last.ckpt')
        trainer.save_checkpoint(best_ckpt)

    # load best model for eval
    best_model = BinarySegmentationModule.load_from_checkpoint(best_ckpt)
    best_model.eval().to(device)

    # eval loop with saving preds
    preds_dir = Path(args.output_dir) / subtype / 'predictions'
    preds_dir.mkdir(parents=True, exist_ok=True)

    # prepare output
    rows = []
    for batch in eval_loader:
        x = batch['image'] # (1, 1, D, H, W)
        y = batch['label'] # (1, 1, D, H, W)
        fname = Path(batch['filename'][0])
        logits = predict_logits(best_model, x)
        dice_050 = dice_at_threshold(logits, y, threshold=0.5)

        # save pred
        mask_bin = (torch.sigmoid(logits) >= 0.5).to(torch.uint8)
        out_path = preds_dir / (fname.stem.replace('.nii', '') + '_pred.nii.gz')
        save_pred_nii(mask_bin, fname, out_path)
        rows.append({'subtype': subtype, 'filename': str(fname.name), 'dice_050': f'{dice_050:.6f}', 'pred_path': str(out_path)})
        print(f'[INFO] {subtype}: Eval {fname.name} -> Dice@0.5: {dice_050:.6f}', flush=True)

    # save metrics CSV
    metrics_csv = Path(args.output_dir) / subtype / f'eval_metrics_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['subtype', 'filename', 'dice_050', 'pred_path'])
        writer.writeheader()
        writer.writerows(rows)

    # summary
    if rows:
        mean_dice = float(np.mean([float(r['dice_050']) for r in rows]))
        print(f'[INFO] {subtype}: Eval mean Dice@0.5 over {len(rows)} samples: {mean_dice:.6f}', flush=True)
        if wandb_logger:
            wandb_logger.experiment.summary[f'{subtype}_eval_mean_dice_050'] = mean_dice

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
    parser.add_argument('--channel_substr', type=str, default='ch0', help='Substring to identify image channels (default: ch0)')

    # split
    parser.add_argument('--mode', type=str, choices=['percent', 'count'], required=True, help='Data splitting mode: "percent" to specify percentages, "count" to specify exact counts')
    parser.add_argument('--train_percent', type=float, default=None, help='Fraction of data to use for training (only in percent mode, default: None)')
    parser.add_argument('--eval_percent', type=float, default=None, help='Fraction of data to use for eval/validation (only in percent mode, default: None, uses remaining data)')
    parser.add_argument('--train_count', type=int, default=None, help='Exact number of samples to use for training (only in count mode, default: None)')
    parser.add_argument('--eval_count', type=int, default=None, help='Exact number of samples to use for eval/validation (only in count mode, default: None, uses remaining data)')
    parser.add_argument('--seed', type=int, default=100, help='Random seed for data shuffling (default: 100)')

    # training
    parser.add_argument('--pretrained_ckpt', type=str, default=None, help='Path to pretrained model checkpoint for finetuning (.ckpt)')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training (default: 4)')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate (default: 1e-4)')
    parser.add_argument('--feature_size', type=int, default=24, help='Feature size (default: 24)')
    parser.add_argument('--max_epochs', type=int, default=1000, help='Maximum number of training epochs (default: 1000)')
    parser.add_argument('--freeze_encoder_epochs', type=int, default=5, help='Number of initial epochs to freeze the encoder (default: 5)')
    parser.add_argument('--encoder_lr_mult', type=float, default=0.05, help='Learning rate multiplier for encoder layers (default: 0.05)')
    parser.add_argument('--loss_name', type=str, choices=['dicece', 'dicefocal'], default='dicece', help='Loss function to use (default: dicece)')
    parser.add_argument('--early_stopping_patience', type=int, default=45, help='Early stopping patience epochs (default: 45)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of DataLoader worker processes (default: 4)')

    # logging/output
    parser.add_argument('--wandb_project', type=str, default='finetune', help='Wandb project name for logging (default: finetune)')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory to save checkpoints, predictions, and metrics')

    # parse
    args = parser.parse_args()

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
    print(f'[INFO] Using device: {device}', flush=True)
    print(f'[INFO] Start at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)
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

    print(f'[INFO] Finished at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)


# --- Entry Point ---

if __name__ == '__main__':
    main()



