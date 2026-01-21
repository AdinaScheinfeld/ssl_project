# /home/ads4015/ssl_project/src/finetune_deblur_split.py - Script that finetunes a deblurring model on sharp/blurred nifti patches using cv folds

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
import random
import sys

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

import torch
from torch.utils.data import DataLoader, get_worker_info

from monai.losses import SSIMLoss

# project imports
sys.path.append('/home/ads4015/ssl_project/models')
from deblur_module import DeblurModule

sys.path.append('/home/ads4015/ssl_project/data')
from nifti_deblur_dataset import NiftiDeblurDataset, discover_nifti_deblur_items

torch.set_float32_matmul_precision('medium')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')


# --- Helper Functions ---

# function to set random seeds for reproducibility
def _seed_everything(seed):
    pl.seed_everything(seed, workers=True)
    np.random.seed(seed)

# function to seed dataloader workers
def _seed_worker(worker_id):
    worker_info = get_worker_info()
    if worker_info is not None:
        base_seed = torch.initial_seed() % 2**31
        np.random.seed(base_seed + worker_info.id)

# function to get list of subfolders under data root
def _list_subtypes(root):
    return sorted([d.name for d in root.iterdir() if d.is_dir()])

# function to save nifti image
def _save_nifti(vol, ref_nifti_path, output_path):

    # convert tensor to numpy
    v = vol.detach().cpu().numpy()
    if v.ndim == 4:
        v = v[0]

    # clip to [0, 1]
    v = np.clip(v.astype(np.float32), 0.0, 1.0)

    # load reference nifti for affine and header
    ref_nifti = nib.load(str(ref_nifti_path))

    # create new nifti image
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(v, affine=ref_nifti.affine, header=ref_nifti.header), str(output_path))

# function to calculate psnr
def _calculate_psnr(pred, target, eps=1e-8):
    mse = float(np.mean((pred - target) ** 2)) + eps
    return 10.0 * np.log10(1.0 / mse)

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


# --- Dataclass ---

# dataclass to hold run outputs
@dataclass
class RunOutputs:
    best_ckpt: str
    metrics_csv: Path
    preds_dir: Path


# --- Train/Eval Per Subtype ---

# function to finetune deblur model on given subtype
def run_one_subtype(subtype_dir, blurred_root, args, device):

    # get subtype
    subtype = subtype_dir.name
    print(f'[INFO] Starting finetuning for subtype: {subtype}', flush=True)

    # discover deblur items for this subtype
    all_items = discover_nifti_deblur_items(
        sharp_class_dir=subtype_dir,
        blurred_root=blurred_root,
        channel_substr=args.channel_substr
    )

    # ensure items found
    if len(all_items) == 0:
        print(f'[WARN] No paired items found for subtype: {subtype}. Skipping.', flush=True)
        return RunOutputs("", Path(""), Path(""))

    # split items into train/val/test based on cv fold
    if args.folds_json and args.fold_id is not None:
        with open(args.folds_json, 'r') as f:
            j = json.load(f)

        if 'folds' in j:
            folds = j['folds']
        else:
            entry = j.get(subtype, {})
            folds = entry.get('folds', [])
            
        if not folds or args.fold_id < 0 or args.fold_id >= len(folds):
            raise ValueError(f'Invalid fold_id {args.fold_id} for folds JSON with {len(folds)} folds for {subtype}.')
        
        # get train/test paths for this fold
        fold = folds[args.fold_id]
        train_paths = fold.get('train', [])
        test_paths = fold.get('test', fold.get('eval', []))

        # train limit if specified
        if args.train_limit is not None and args.train_limit >=0:
            train_paths = train_paths[: int(args.train_limit)]

        # split items
        train_set = set(str(p) for p in train_paths)
        test_set = set(str(p) for p in test_paths)

        # map from sharp image path string to DeblurItem
        path_to_item = {str(item.sharp_image.resolve()): item for item in all_items}
        train_items = [path_to_item[p] for p in train_set if p in path_to_item]
        test_items = [path_to_item[p] for p in test_set if p in path_to_item]

    # if no folds json provided, use 80/20 train/test split
    else:
        n = len(all_items)
        k = int(0.8 * n)
        train_items = all_items[:k]
        test_items = all_items[k:]

    # split train pool into finetune train/val sets
    val_count = (args.val_count if args.val_count is not None else max(1, int(round(args.val_fraction * len(train_items)))))

    # shuffle train items
    rng = random.Random(args.seed + 1)
    train_items_shuffled = list(train_items)
    rng.shuffle(train_items_shuffled)

    ft_val_items = train_items_shuffled[:val_count]
    ft_train_items = train_items_shuffled[val_count:]

    # skip if insufficient items
    if len(ft_train_items) < args.min_finetune_train or len(ft_val_items) < args.min_finetune_val:
        print(f'[WARN] Insufficient finetune train/val items for subtype: {subtype}. '
              f'{len(ft_train_items)}/{args.min_finetune_train} train and '
              f'{len(ft_val_items)}/{args.min_finetune_val} val items. Skipping.', flush=True
        )
        return RunOutputs("", Path(""), Path(""))
    print(f'[INFO] Finetune train/val/test split for subtype {subtype}: '
          f'{len(ft_train_items)} / {len(ft_val_items)} / {len(test_items)} items.', flush=True)
    
    # build datasets and dataloaders
    ds_train = NiftiDeblurDataset(ft_train_items)
    ds_val = NiftiDeblurDataset(ft_val_items)
    ds_test = NiftiDeblurDataset(test_items)

    loader_kw = dict(
        num_workers=min(args.num_workers, os.cpu_count() or args.num_workers),
        pin_memory=torch.cuda.is_available(),
        persistent_workers=False,
        worker_init_fn=_seed_worker
    )

    dl_train = DataLoader(ds_train, batch_size=args.batch_size, shuffle=True, **loader_kw)
    dl_val = DataLoader(ds_val, batch_size=1, shuffle=False, **loader_kw)
    dl_test = DataLoader(ds_test, batch_size=1, shuffle=False, **loader_kw)

    # compose experiment tag for ckpt, preds, wandb
    fold_tag = (f'cvfold{args.fold_id}' if (args.folds_json is not None and args.fold_id is not None) else 'nofold')
    limit_tag = (f'trlim{args.train_limit}' if args.train_limit is not None and args.train_limit >=0 else 'trlimALL')
    tag = f'{fold_tag}_{limit_tag}_fttr{len(ft_train_items)}_ftval{len(ft_val_items)}_tst{len(test_items)}_{subtype}_seed{args.seed}'

    # create checkpoint directory
    ckpt_dir = Path(args.ckpt_dir) / subtype / tag
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # wandb logger
    run_name = f'deblur_{subtype}_{tag}'
    wandb_logger = WandbLogger(project=args.wandb_project, name=run_name)

    # instantiate deblur lightning module
    model = DeblurModule(
        pretrained_ckpt_path=args.pretrained_ckpt_path,
        lr=args.lr,
        feature_size=args.feature_size,
        encoder_lr_mult=args.encoder_lr_mult,
        freeze_encoder_epochs=args.freeze_encoder_epochs,
        weight_decay=args.weight_decay
    )

    # callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename='best_ckpt',
        monitor='val_total_loss',
        mode='min',
        save_top_k=1
    )
    early_stopping_callback = EarlyStopping(
        monitor='val_total_loss',
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

    # train
    trainer.fit(model, dl_train, dl_val)

    # save best checkpoint
    best_ckpt_path = checkpoint_callback.best_model_path or str(ckpt_dir / 'best_ckpt.ckpt')
    if not checkpoint_callback.best_model_path:
        trainer.save_checkpoint(best_ckpt_path)
    print(f'[INFO] Best checkpoint saved at: {best_ckpt_path}', flush=True)


    # evaluate on test set
    print(f'[INFO] Starting evaluation on test set for subtype: {subtype}', flush=True)
    
    # create preds output directory
    preds_dir = Path(args.preds_root) / subtype / tag / 'preds'
    preds_dir.mkdir(parents=True, exist_ok=True)

    # list to hold metrics
    rows = []

    # load best checkpoint
    best_model = DeblurModule.load_from_checkpoint(best_ckpt_path).to(device).eval()

    # iterate over test items
    with torch.no_grad():
        for batch in dl_test:

            # get data
            blurred_img = batch['input_vol'].to(device)
            sharp_img = batch['target_vol'].to(device)
            
            # forward pass
            residual = best_model(blurred_img)
            output_deblurred_pred = torch.clamp(blurred_img + residual, 0.0, 1.0) # [1, 1, D, H, W]

            # save prediction nifti
            fname = Path(batch['filename'][0])
            output_pred_path = preds_dir / (fname.stem.replace('.nii', '') + '_deblur_pred.nii.gz')
            ref_nifti_path = subtype_dir / fname
            _save_nifti(output_deblurred_pred[0, 0], ref_nifti_path, output_pred_path)

            # calculate psnr over whole volume
            psnr = _calculate_psnr(
                pred=output_deblurred_pred[0, 0].detach().cpu().numpy(),
                target=sharp_img[0, 0].detach().cpu().numpy()
            )

            ssim = _calculate_ssim_3d(
                pred_t=output_deblurred_pred[0, 0],
                target_t=sharp_img[0, 0],
            )

            # append metrics row
            rows.append(
                {
                    "subtype": subtype,
                    "filename": fname.name,
                    "psnr": f"{psnr:.4f}",
                    "ssim": f"{ssim:.4f}",
                    "pred_path": str(output_pred_path),
                }
            )

    # save metrics to csv
    metrics_csv_path = preds_dir / 'metrics_test.csv'
    with open(metrics_csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["subtype", "filename", "psnr", "ssim", "pred_path"])
        writer.writeheader()
        writer.writerows(rows)

    print(f'[INFO] Test metrics saved at: {metrics_csv_path}', flush=True)

    # add mean psnr to wandb summary for quick comparison
    if wandb_logger and rows:
        mean_psnr = float(np.mean([float(r['psnr']) for r in rows]))
        wandb_logger.experiment.summary[f'{subtype}/{tag}/mean_test_psnr'] = mean_psnr
        print(f'[INFO] Mean test PSNR for subtype {subtype}: {mean_psnr:.4f}', flush=True)

    # cleanup (to avoid leaked semaphores)
    del dl_train, dl_val, dl_test
    del ds_train, ds_val, ds_test
    gc.collect()

    return RunOutputs(
        best_ckpt=best_ckpt_path,
        metrics_csv=metrics_csv_path,
        preds_dir=preds_dir
    )


# --- Main ---

# arg parser
def parse_args():

    parser = argparse.ArgumentParser(description='Finetune Deblur Model on Nifti Data per Subtype')

    # data args
    parser.add_argument('--data_root', type=str, required=True, help='Root directory containing sharp subdirectories.')
    parser.add_argument('--blur_root', type=str, required=True, help='Root directory containing blurred images.')
    parser.add_argument('--subtypes', nargs='*', default=['ALL'], help='List of subtypes to process. Use "ALL" to process all subtypes found under data_root. Default: ALL')
    parser.add_argument('--exclude_subtypes', nargs='*', default=[], help='List of subtypes to exclude from processing.')
    parser.add_argument('--channel_substr', type=str, default='ALL', help='Substring filter for channel selection in filenames. Default: ALL (no filtering).')
    parser.add_argument('--folds_json', type=str, default=None, help='Path to JSON file defining CV folds.')
    parser.add_argument('--fold_id', type=int, default=None, help='Fold ID to use from folds JSON.')
    parser.add_argument('--train_limit', type=int, default=None, help='Limit on number of training items to use per subtype.')

    # finetuning args
    parser.add_argument('--val_fraction', type=float, default=0.2, help='Fraction of training data to use for validation.')
    parser.add_argument('--val_count', type=int, default=None, help='Number of validation items to use (overrides val_fraction).')
    parser.add_argument('--min_finetune_train', type=int, default=1, help='Minimum number of finetune training items required to run. Default: 1')
    parser.add_argument('--min_finetune_val', type=int, default=1, help='Minimum number of finetune validation items required to run. Default: 1')
    parser.add_argument('--pretrained_ckpt_path', type=str, default=None, help='Path to pretrained checkpoint to initialize SwinUNETR encoder. Default: None (random init)')

    # training hyperparameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for finetuning.')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for finetuning.')
    parser.add_argument('--feature_size', type=int, default=24, help='Feature size for deblur model.')
    parser.add_argument('--encoder_lr_mult', type=float, default=0.1, help='Learning rate multiplier for encoder layers.')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay for optimizer.')
    parser.add_argument('--freeze_encoder_epochs', type=int, default=5, help='Number of epochs to freeze encoder during finetuning. Default: 0 (no freezing)')
    parser.add_argument('--max_epochs', type=int, default=500, help='Maximum number of finetuning epochs.')
    parser.add_argument('--early_stopping_patience', type=int, default=50, help='Early stopping patience epochs.')  
    parser.add_argument('--num_workers', type=int, default=1, help='Number of dataloader workers.')
    parser.add_argument('--seed', type=int, default=100, help='Random seed for reproducibility.')

    # output args
    parser.add_argument('--ckpt_dir', type=str, required=True, help='Directory to save finetuned checkpoints.')
    parser.add_argument('--preds_root', type=str, required=True, help='Directory to save predictions.')
    parser.add_argument('--wandb_project', type=str, default='deblur_finetune', help='Weights & Biases project name.')

    return parser.parse_args()

# main function
def main():

    # parse args
    args = parse_args()
    _seed_everything(args.seed)

    # get data
    data_root = Path(args.data_root)
    blurred_root = Path(args.blur_root)
    if not data_root.exists():
        raise FileNotFoundError(f'Data root directory not found: {data_root}')
    if not blurred_root.exists():
        raise FileNotFoundError(f'Blurred root directory not found: {blurred_root}')

    # device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # determine subtypes to process
    if any(s.upper() == 'ALL' for s in args.subtypes):
        subtypes = [s for s in _list_subtypes(data_root) if s not in set(args.exclude_subtypes)]
    else:
        subtypes = args.subtypes

    print(f'[INFO] Subtypes to process: {subtypes}', flush=True)
    print(f'[INFO] Using device: {device}', flush=True)
    print(f'[INFO] Starting deblur finetuning at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

    # iterate over subtypes
    for subtype in subtypes:
        subtype_dir = data_root / subtype
        if not subtype_dir.exists():
            print(f'[WARN] Subtype directory not found: {subtype_dir}. Skipping.', flush=True)
            continue

        # run finetuning for this subtype
        outputs = run_one_subtype(subtype_dir, blurred_root, args, device)
        if outputs.best_ckpt:
            print(f'[INFO] Completed finetuning for subtype: {subtype}. ' 
                  f'Best checkpoint saved at: {outputs.best_ckpt} '
                  f'Metrics csv: {outputs.metrics_csv} '
                  f'Predictions saved at: {outputs.preds_dir}', flush=True)
            
    print(f'[INFO] Finished all finetuning at {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', flush=True)

if __name__ == '__main__':
    main()





























