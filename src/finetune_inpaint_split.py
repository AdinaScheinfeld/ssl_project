# finetune_inpaint_split.py - Finetunes text-conditioned inpainting model and evaluates on validation set

# --- Setup ---

# imports
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
import sys
import time

import torch
from torch.utils.data import DataLoader, get_worker_info

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

sys.path.append('/home/ads4015/ssl_project/models')
from inpaint_module import InpaintModule

sys.path.append('/home/ads4015/ssl_project/data')
from nifti_inpaint_dataset import NiftiInpaintDataset, discover_nifti_inpaint_items

torch.set_float32_matmul_precision('medium')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false') # disable tokenizer parallelism warnings


# --- Helper Functions ---

# function to seed everything
def _seed_everything(seed):
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)

# function to seed worker deterministically based on id
def _seed_worker(worker_id):
    worker_info = get_worker_info()
    if worker_info is not None:
        base_seed = torch.initial_seed() % 2**31
        np.random.seed(base_seed + worker_info.id)

# function to return immediate subfolder names under data root
def _list_subtypes(root):
    return sorted([d.name for d in root.iterdir() if d.is_dir()])

# cv folds loader to load train/test splits from json file
def _split_from_folds(subtype, fold_json, fold_id, train_limit=None):

    # load folds json
    with open(fold_json, 'r') as f:
        j = json.load(f)
    
    # get subtype entry
    entry = j.get(subtype, {})
    folds = entry.get('folds', [])
    if not folds or fold_id < 0 or fold_id >= len(folds):
        raise ValueError(f"Invalid fold_id {fold_id} for subtype {subtype}")
    
    # get train/test lists
    fold = folds[fold_id]
    train_list = [Path(p) for p in fold['train']]
    test_list = [Path(p) for p in fold.get('test', fold.get('eval', []))] # support both "test" and "eval" key names

    # apply train limit if specified
    if train_limit is not None and train_limit >= 0:
        train_list = train_list[:min(len(train_list), int(train_limit))]

    return train_list, test_list

# *** I/O helpers ***

# function to save preds as nifti
def _save_nifti(vol, ref_nifti_path, out_path):

    # get volume
    vol_np = vol.detach().cpu().numpy().astype(np.float32)
    vol_np = np.clip(vol_np, 0.0, 1.0) # clamp to [0,1]

    try:
        # load reference nifti for affine
        ref_nifti = nib.load(str(ref_nifti_path))
        affine, header = ref_nifti.affine, ref_nifti.header
    except Exception:
        affine, header = np.eye(4), nib.Nifti1Header()

    # create nifti and save
    nib.save(nib.Nifti1Image(vol_np, affine, header), str(out_path))

# function to calculate psnr between two volumes
def _calc_psnr(pred, target, eps=1e-8):

    mse = np.mean((pred - target) ** 2) + eps
    psnr = 10.0 * np.log10(1.0 / mse)
    return psnr


# --- Dataclass ---

@dataclass
class RunOutputs:
    best_ckpt: str # path to best checkpoint (.ckpt)
    metrics_csv: Path # path to metrics csv file
    preds_dir: Path # path to predictions output directory


# --- Train/Eval ---

# function to run training and evaluation for one subtype
def run_one_subtype(subdir, args, device):

    # get subtype name
    subtype = subdir.name
    print(f'[INFO] Starting finetuning for subtype: {subtype}')

    # discover items
    all_items = discover_nifti_inpaint_items(subdir, channel_substr=args.channel_substr)

    # get train/test splits

    # prefer folds json if provided
    if args.folds_json and args.fold_id is not None:

        # build maps to select items by path
        with open(args.folds_json, 'r') as f:
            j = json.load(f)
        entry = j.get(subtype, {})
        folds = entry.get('folds', [])
        if not folds or args.fold_id < 0 or args.fold_id >= len(folds):
            raise ValueError(f"Invalid fold_id {args.fold_id} for subtype {subtype}")
        fold = folds[args.fold_id]
        train_set = set(map(str, fold.get('train', [])))
        # support both "test" and "eval" key names
        test_list = fold.get('test', fold.get('eval', []))
        test_set = set(map(str, test_list))

        # cap train size deterministically if specified
        if args.train_limit is not None and args.train_limit >= 0:
            train_list = [p for p in fold.get('train', [])][: int(args.train_limit)]
            train_set = set(map(str, train_list))

        # map paths to discovered items
        amap = {str(item.image.resolve()): item for item in all_items}
        train_items = [amap[p] for p in train_set if p in amap]
        test_items = [amap[p] for p in test_set if p in amap]

    # else, use simple 80/20 split
    else:
        n = len(all_items)
        k = int(0.8 * n)
        train_items = all_items[:k]
        test_items = all_items[k:]

    # split finetune training pool into train/val for model selection
    val_count = args.val_count if args.val_count is not None else max(1, int(round(args.val_percent * max(1, len(train_items)))))
    ft_val_items = train_items[:val_count]
    ft_train_items = train_items[val_count:]

    # ensure minimum sizes
    if len(ft_train_items) < args.min_finetune_train or len(ft_val_items) < args.min_finetune_val:
        print(f'[WARN] Skipping subtype {subtype} due to insufficient finetune data: {len(ft_train_items)}/{args.min_finetune_train} train, {len(ft_val_items)}/{args.min_finetune_val} val')
        return RunOutputs('', Path(''), Path(''))
    
    print(f'[INFO] Finetune train/val/test sizes for subtype {subtype}: {len(ft_train_items)}/{len(ft_val_items)}/{len(test_items)} (train/val/test)')

    # get default captions from folder names
    default_captions = {
        'amyloid_plaque_patches': 'Bright, compact fluorescent deposits marking extracellular amyloid plaques in a cleared mouse brain.',
        'c_fos_positive_patches': 'Small, sharply defined glowing nuclei indicating activity-dependent c-Fos expression in cleared tissue.',
        'cell_nucleus_patches': 'Numerous small, round fluorescent dots—each dot a cell nucleus—forming a dense speckled 3D pattern.',
        'vessels_patches': 'Interconnected tubular fluorescent strands tracing blood vessels from large channels to fine capillaries.'
    }

    # build datasets and dataloaders
    ds_train = NiftiInpaintDataset(ft_train_items, captions_json=args.captions_json, default_caption_by_subtype=default_captions, mask_ratio=args.mask_ratio, augment=True, seed=args.seed+1)
    ds_val = NiftiInpaintDataset(ft_val_items, captions_json=args.captions_json, default_caption_by_subtype=default_captions, mask_ratio=args.mask_ratio, augment=False, seed=args.seed+2)
    ds_test = NiftiInpaintDataset(test_items, captions_json=args.captions_json, default_caption_by_subtype=default_captions, mask_ratio=args.mask_ratio_test, augment=False, seed=args.seed+3)

    loader_kw = dict(num_workers=min(args.num_workers, os.cpu_count() or args.num_workers),
                     pin_memory=torch.cuda.is_available(),
                     persistent_workers=False,
                     worker_init_fn=_seed_worker)
    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, **loader_kw)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, **loader_kw)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, **loader_kw)

    # compose descriptive experiment tag for output and wandb
    fold_tag = f'cvfold{args.fold_id}' if (args.folds_json is not None and args.fold_id is not None) else 'nosplit'
    limit_tag = (f'trlim{args.train_limit}' if (args.train_limit is not None and args.train_limit >= 0) else 'trlimALL')
    tag = f'{fold_tag}_{limit_tag}_fttr{len(ft_train_items)}_ftval{len(ft_val_items)}_tst{len(test_items)}_seed{args.seed}'
    run_name = f'inpaint_{subtype}_{tag}'

    # create checkpoint/output dirs
    ckpt_dir = Path(args.ckpt_dir) / subtype / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # wandb logger
    wandb_logger = WandbLogger(project=args.wandb_project, name=run_name)

    # instantiate lightning module
    model = InpaintModule(
        pretrained_ckpt_path=args.pretrained_ckpt_path,
        lr=args.lr,
        feature_size=args.feature_size,
        encoder_lr_mult=args.encoder_lr_mult,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        l1_weight_masked=args.l1_weight_masked,
        l1_weight_global=args.l1_weight_global,
        text_cond=not args.disable_text_cond,
        text_dim=args.text_dim,
        text_backend=args.text_backend,
        clip_ckpt=args.clip_ckpt,
        weight_decay=args.weight_decay
    )

    # callbacks: model checkpointing and early stopping
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename='best_ckpt',
        save_top_k=1,
        monitor='val_loss',
        mode='min'
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_loss',
        patience=args.early_stopping_patience,
        mode='min'
    )

    # trainer
    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        precision='bf16-mixed' if torch.cuda.is_available() else 32,
        logger=wandb_logger,
        callbacks=[checkpoint_callback, early_stopping_callback],
        log_every_n_steps=1,
        deterministic=True
    )

    # fit (train + val)
    trainer.fit(model, dl_train, dl_val)

    # resolve best checkpoint path
    best_ckpt = checkpoint_callback.best_model_path or (str(ckpt_dir / 'best_ckpt.ckpt'))
    if not checkpoint_callback.best_model_path:
        trainer.save_checkpoint(best_ckpt)
    print(f'[INFO] Best checkpoint saved at: {best_ckpt}')

    # test loop (write preds and calculate metrics)
    print(f'[INFO] Starting test evaluation for subtype {subtype}...')

    # create preds output dir
    preds_root = Path(args.preds_root) if args.preds_root else Path(args.ckpt_dir)
    preds_dir = preds_root / subtype / tag / 'preds'
    preds_dir.mkdir(parents=True, exist_ok=True)

    # list to hold metrics
    rows = []

    # reload best checkpoint
    best_model = InpaintModule.load_from_checkpoint(best_ckpt).to(device).eval()

    # iterate over test set
    with torch.no_grad():

        for batch in dl_test:

            # get data
            masked_vol = batch['masked_vol'].to(device) # (1, 1, D, H, W), original image with masked region zeroed out
            mask = batch['mask'].to(device)               # (1, 1, D, H, W)
            target_vol = batch['target_vol'].to(device)           # (1, 1, D, H, W)

            # text conditioning from dataset (folder name or captions json)
            t_emb = None
            if not args.disable_text_cond and 'text' in batch:
                t_emb = best_model.text_encoder(batch['text']).to(device) # (1, text_dim)

            # forward pass
            pred = best_model(masked_vol, mask, t_emb)    # (1, 1, D, H, W)
            pred = torch.sigmoid(pred) # (1, 1, D, H, W), clamp to [0, 1]

            # save prediction as nifti
            fname = Path(batch['filename'][0])
            output_path = preds_dir / (fname.stem.replace('.nii', '') + '_inpaint_pred.nii.gz')
            _save_nifti(pred[0,0], ref_nifti_path=(subdir / fname), out_path=output_path)

            # compose prediction into original volume
            # composite = original_unmasked + pred_in_hole
            composite = (masked_vol + pred * mask).clamp(0, 1)
            output_composite_path = preds_dir / (fname.stem.replace('.nii', '') + '_inpaint_composite.nii.gz')
            _save_nifti(composite[0,0], ref_nifti_path=(subdir / fname), out_path=output_composite_path)

            # compute psnr (on masked region only)
            pred_mask = (pred * mask).detach().cpu().numpy() # predicted values in masked region, already clamped to [0,1] (sigmoid)
            target_mask = (target_vol * mask).detach().cpu().numpy()
            psnr = _calc_psnr(pred_mask, target_mask)
            rows.append({'subtype': subtype, 
                         'filename': fname.name, 
                         'psnr_masked': f'{psnr:.4f}', 
                         'pred_path': str(output_path),
                         'composite_path': str(output_composite_path)}
                         )
            
    # write metrics to csv
    metrics_csv = preds_dir / 'metrics_test.csv'
    with open(metrics_csv, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['subtype', 'filename', 'psnr_masked', 'pred_path', 'composite_path'])
        w.writeheader()
        w.writerows(rows)

    print(f'[INFO] Test metrics saved at: {metrics_csv}')

    # add psnr to wandb summary to compare across runs
    if wandb_logger and rows:
        metric = float(np.mean([float(r['psnr_masked']) for r in rows]))
        wandb_logger.experiment.summary[f'{subtype}/{tag}/test_mean_psnr_masked'] = metric

    # free dataloaders/datasets to avoid leaked semaphores
    del dl_train, dl_val, dl_test
    del ds_train, ds_val, ds_test
    gc.collect()

    # return outputs
    return RunOutputs(best_ckpt=best_ckpt, metrics_csv=metrics_csv, preds_dir=preds_dir)


# --- CLI ---

# parse arguments
def parse_args():

    parser = argparse.ArgumentParser(description='Finetune text-conditioned inpainting model on Nifti datasets with splits.')

    # data
    parser.add_argument('--data_root', type=str, required=True, help='Root directory containing subtype subfolders with Nifti files.')
    parser.add_argument('--subtypes', nargs='*', default=['ALL'], help='List of subtype subfolder names to process. Use "ALL" to process all subfolders under data_root (Default: ALL).')
    parser.add_argument('--exclude_subtypes', nargs='*', default=[], help='List of subtype subfolder names to exclude from processing (Default: []).')
    parser.add_argument('--channel_substr', default='ALL', help='Substring to filter Nifti files by channel in their filenames (Default: ALL).')
    parser.add_argument('--captions_json', type=str, default=None, help='Optional path to JSON file mapping Nifti filenames to text captions for conditioning (Default: None).')

    # cross validation splits
    parser.add_argument('--folds_json', type=str, default=None, help='Path to JSON file defining cross-validation folds for subtypes (Default: None).')
    parser.add_argument('--fold_id', type=int, default=None, help='Fold ID to use from folds_json (Default: None).')
    parser.add_argument('--train_limit', type=int, default=None, help='Optional limit on number of training samples to use per subtype (Default: None for all).')

    # finetune train/val split within training set
    parser.add_argument('--val_percent', type=float, default=0.2, help='Percentage of finetune training set to use for validation (Default: 0.2).')
    parser.add_argument('--val_count', type=int, default=None, help='Optional fixed number of validation samples for finetune (overrides val_percent) (Default: None).')
    parser.add_argument('--min_finetune_train', type=int, default=1, help='Minimum number of finetune training samples required to run finetuning (Default: 1).')
    parser.add_argument('--min_finetune_val', type=int, default=1, help='Minimum number of finetune validation samples required to run finetuning (Default: 1).')

    # model
    parser.add_argument('--pretrained_ckpt_path', type=str, default=None, help='Path to pretrained model checkpoint (Default: None).')
    parser.add_argument('--batch_size', type=int, default=2, help='Finetune training batch size (Default: 2).')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for finetuning (Default: 1e-4).')
    parser.add_argument('--feature_size', type=int, default=32, help='Feature size for model (Default: 32).')
    parser.add_argument('--encoder_lr_mult', type=float, default=0.05, help='Learning rate multiplier for encoder during finetuning (Default: 0.05).')
    parser.add_argument('--freeze_encoder_epochs', type=int, default=5, help='Number of initial epochs to freeze encoder during finetuning (Default: 5).')
    parser.add_argument('--l1_weight_masked', type=float, default=1.0, help='L1 loss weight for masked region (Default: 1.0).')
    parser.add_argument('--l1_weight_global', type=float, default=0.1, help='L1 loss weight for global image (Default: 0.1).')
    parser.add_argument('--mask_ratio', type=float, default=0.3, help='Mask ratio for training/validation datasets (Default: 0.3).')
    parser.add_argument('--mask_ratio_test', type=float, default=0.3, help='Mask ratio for test dataset (Default: 0.3).')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Optimizer weight decay (Default: 1e-5).')
    parser.add_argument('--disable_text_cond', action='store_true', help='Disable text conditioning during finetuning (Default: False).')
    parser.add_argument('--text_dim', type=int, default=512, help='Text dimension for model (Default: 512).')
    parser.add_argument('--text_backend', type=str, default='clip', choices=['clip', 'dummy'], help='Text encoder backend to use (Default: clip).')
    parser.add_argument('--clip_ckpt', type=str, default=None, help='Optional path to CLIP checkpoint for text encoder (Default: None).')

    # training
    parser.add_argument('--max_epochs', type=int, default=500, help='Maximum number of finetuning epochs (Default: 500).')
    parser.add_argument('--early_stopping_patience', type=int, default=50, help='Number of epochs with no improvement after which training will be stopped (Default: 50).')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers (Default: 4).')
    parser.add_argument('--seed', type=int, default=100, help='Random seed for reproducibility (Default: 100).')

    # output
    parser.add_argument('--wandb_project', type=str, default='inpaint_finetune', help='Weights & Biases project name for logging (Default: inpaint_finetune).')
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Directory to save finetuned model checkpoints.')
    parser.add_argument('--preds_root', type=str, required=True, help='Root directory for saving model predictions.')

    args = parser.parse_args()
    return args


# --- Main ---

# main function
def main():

    # parse args
    args = parse_args()
    _seed_everything(args.seed)
    root = Path(args.data_root)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # resolve subtypes to process
    if any(s.upper() == 'ALL' for s in args.subtypes):
        subtypes = [s for s in _list_subtypes(root) if s not in set(args.exclude_subtypes)]
    else:
        subtypes = args.subtypes

    print(f'[INFO] Subtypes to process: {subtypes}')
    print(f'[INFO] Device: {device}')
    print(f'[INFO] Starting finetuning at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    # iterate over subtypes
    for subtype in subtypes:
        subdir = root / subtype
        if not subdir.exists():
            print(f'[WARN] Subtype directory does not exist, skipping: {subdir}')
            continue
        output = run_one_subtype(subdir, args, device)
        if output.best_ckpt:
            print(f'[INFO] Finished finetuning for subtype {subtype}. Best checkpoint: {output.best_ckpt}, Metrics CSV: {output.metrics_csv}, Predictions Dir: {output.preds_dir}')

    print(f'[INFO] Finetuning completed at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')


# entry point
if __name__ == '__main__':
    main()








