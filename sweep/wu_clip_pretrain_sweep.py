# wu_clip_pretrain.py - Wu Data Pretraining

# --- Setup ---

# imports
import argparse
from datetime import datetime
from math import ceil
import os
import sys
import wandb
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

# get ibot pretraining module
sys.path.append('/home/ads4015/ssl_project/sweep/')
from ibot_clip_pretrain_module_sweep import IBOTCLIPPretrainModuleSweep

# get data module
sys.path.append('/home/ads4015/ssl_project/data/')
from wu_clip_data_module import WuCLIPDataModule


os.environ['TOKENIZERS_PARALLELISM'] = 'false'

# set matmul precision for better performance on tensor core gpus
torch.set_float32_matmul_precision('medium')


# load config from yaml file
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)
    

# function to compute per device batch size based on global batch size and number of devices
def compute_per_device_batch_size(cfg, world_override=None):

    data = cfg['data']

    # prefer global batch size, if set
    global_batch_size = data.get('global_batch_size')
    if not global_batch_size:
        return data['batch_size'] # if global batch size not set, return batch size per gpu from config
    
    # if global batch size is set, get distributed computing config values
    dist = cfg.get('dist', {})

    if world_override is None:
        configured_world = int(dist.get('devices', 1)) * int(dist.get('num_nodes', 1)) # total number of gpus across all nodes
        env_world = int(os.environ.get('WORLD_SIZE', configured_world)) # get world size from environment variable, default to configured world size
        world = max(env_world, 1)
    else:
        world = max(int(world_override), 1)

    accumulate_grad_batches = int(dist.get('accumulate_grad_batches', 1)) # gradient accumulation
    per_dev = global_batch_size // (world * accumulate_grad_batches) # per device batch size

    # ensure correct dimensions and divisibility
    if per_dev < 1:
        raise ValueError(f'Global batch size {global_batch_size} is too small for {world} devices with accumulate_grad_batches={accumulate_grad_batches}. Set a larger global batch size.', flush=True)
    if global_batch_size % (world * accumulate_grad_batches) != 0:
        print(f'[WARNING] global_batch_size={global_batch_size} not divisible by world*accumulate_grad_batches={world*accumulate_grad_batches}.'
              f'Using per-device batch_size={per_dev} and effective global={per_dev*world*accumulate_grad_batches}.', flush=True)
        
    return per_dev


# function to determine whether using 1 or more gpus
def resolve_dist(config):

    dist = config.get('dist', {})
    use_multi = bool(dist.get('multi_gpu', False)) # use multi-gpu if set to True in config

    # # get values
    # devices = int(dist.get('devices', 1)) # number of gpus per node
    # num_nodes = int(dist.get('num_nodes', 1)) # number of nodes
    # strategy = 'auto'

    # get values if using multiple gpus
    if use_multi:
        # devices = int(os.environ.get('SLURM_NTASKS_PER_NODE', devices)) # number of gpus per node from slurm env
        # num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', os.environ.get('SLURM_NNODES', num_nodes))) # number of nodes from slurm env
        # strategy = DDPStrategy(find_unused_parameters=True)
        # world = max(1, devices * num_nodes) # total number of gpus across all nodes

        requested = int(dist.get('devices', 1))
        local_gpus = torch.cuda.device_count()
        devices = max(1, min(requested, local_gpus))
        num_nodes = 1 # lock to 1 node only
        strategy = DDPStrategy(find_unused_parameters=True)
        world = devices # local world size
    
    # if using single gpu, set devices to 1
    else:
        devices, num_nodes, world = 1, 1, 1
        strategy = 'auto'

    # return values
    return devices, num_nodes, strategy, world


# wandb sweep helper functions

# function to get sweep defaults and save as dict
def _get_sweep_defaults(cfg: dict):

    # create a dict to hold sweep defaults from config
    d = {}

    # data section
    d['global_batch_size'] = cfg['data'].get('global_batch_size', 'batch_size') # use global batch size, if set, otherwise use batch size

    # model section (optimizer + architecture + temperatures)
    for k in ['feature_size', 'clip_temperature', 'ema_decay', 'embed_dim', 'lr', 'mask_patch_size', 
              'mask_ratio', 'mask_ratio_warmup', 'temp_student', 'temp_teacher', 'warmup_epochs']:
        d[k] = cfg['model'][k]

    # loss weights section
    for k in ['align_weight', 'clip_weight', 'distill_weight', 'reconstruction_weight']:
        d[k] = cfg['loss_weights'][k]

    # set logit defaults (0.0 means equal weights after softmax normalization)
    d['distill_logit'] = 0.0
    d['reconstruction_logit'] = 0.0
    d['align_logit'] = 0.0
    d['clip_logit'] = 0.0

    # return the dict
    return d


# function to override config with sweep parameters
def _apply_sweep_overrides(cfg: dict, sweep_cfg):

    # data section
    if 'global_batch_size' in sweep_cfg:
        cfg['data']['global_batch_size'] = int(sweep_cfg['global_batch_size'])

    # model section
    for k in ['feature_size', 'clip_temperature', 'ema_decay', 'embed_dim', 'lr', 'mask_patch_size', 
              'mask_ratio', 'mask_ratio_warmup', 'temp_student', 'temp_teacher', 'warmup_epochs']:
        if k in sweep_cfg:
            if k in ['embed_dim', 'feature_size', 'mask_patch_size', 'warmup_epochs']:
                cfg['model'][k] = int(sweep_cfg[k])
            else:
                cfg['model'][k] = float(sweep_cfg[k])
    
    # loss weights section
    for k in ['align_weight', 'clip_weight', 'distill_weight', 'reconstruction_weight']:
        if k in sweep_cfg:
            cfg['loss_weights'][k] = float(sweep_cfg[k])


# determine hardware specific per gpu batch cap (each h100s can have max batch size of 8, each l40 can have max batch size of 2)
def _detect_gpu_type_and_cap():

    if not torch.cuda.is_available():
        return ('CPU', 1) # if no gpu available, return cpu and batch size 1
    
    name = torch.cuda.get_device_name(0).lower() # get name of the first gpu
    if 'h100' in name:
        return ('H100', 8)
    if 'l40' in name:
        return ('L40', 2)
    
    mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3) # get memory of the first gpu in GB
    if mem_gb >= 80:
        return ('large_mem_gpu', 8)
    elif mem_gb >= 40:
        return ('mid_mem_gpu', 2)
    else:
        return ('small_mem_gpu', 1)
    

# function to enforce per device batch size limits based on gpu type
def _enforce_per_device_batch_cap(cfg: dict, world: int, cap_per_device: int):

    # ensure per-device batch size is not larger than cap
    data = cfg['data']
    dist = cfg.get('dist', {})
    accumulate_grad_batches = int(dist.get('accumulate_grad_batches', 1)) # gradient accumulation
    global_batch_size = int(data.get('global_batch_size') or data.get('batch_size', 1)) # use global batch size if set, otherwise use batch size

    if global_batch_size < 1:
        global_batch_size = 1

    per_device_batch_size = global_batch_size // max(1, world * accumulate_grad_batches) # compute per device batch size
    if per_device_batch_size <= cap_per_device:
        return
    
    # lower per device batch size to cap, if necessary
    new_per_device_batch_size = cap_per_device
    new_global_batch_size = max(1, new_per_device_batch_size * max(1, world * accumulate_grad_batches)) # compute new global batch size
    print(f'[WARNING] Requested global_batch_size={global_batch_size} implies per-device={per_device_batch_size} > cap={cap_per_device}. '
          f'Reducing global_batch_size to {new_global_batch_size} (per-device={new_per_device_batch_size}, world={world}, accumulate_grad_batches={accumulate_grad_batches}).', 
          flush=True)
    data['global_batch_size'] = new_global_batch_size # update global batch size in config


# function to normalize loss weights (to make runs comparable)
def _normalize_loss_weights(cfg: dict, sweep_cfg: dict):

    # get logit keys from config
    logit_keys = ['distill_logit', 'reconstruction_logit', 'align_logit', 'clip_logit']
    if not any(k in sweep_cfg for k in logit_keys):
        return # if no logit keys in sweep config, nothing to normalize and use default weights
    
    # get weights from config, default to 0.0 (equeal weights) if not set
    logits = torch.tensor([
        float(sweep_cfg.get('distill_logit', 0.0)),
        float(sweep_cfg.get('reconstruction_logit', 0.0)),
        float(sweep_cfg.get('align_logit', 0.0)),
        float(sweep_cfg.get('clip_logit', 0.0))
    ], dtype=torch.float32)

    # convert to weights using softmax
    weights = torch.softmax(logits, dim=0).tolist()
    cfg['loss_weights']['distill_weight'] = weights[0]
    cfg['loss_weights']['reconstruction_weight'] = weights[1]
    cfg['loss_weights']['align_weight'] = weights[2]
    cfg['loss_weights']['clip_weight'] = weights[3]

    # log the normalized weights
    print(f'[INFO] Using sftmax-normalized loss weights from logits: '
          f'distill={weights[0]:.4f}, reconstruction={weights[1]:.4f}, align={weights[2]:.4f}, clip={weights[3]:.4f}', 
          flush=True)


# --- Main Entry Point --- 

# main
if __name__ == '__main__':

    # load config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config yaml file')
    args = parser.parse_args()
    config = load_config(args.config)

    # set seed for reproducibility
    pl.seed_everything(config['training']['seed'])

    # initialize wandb and overrides
    sweep_defaults = _get_sweep_defaults(config) # get sweep defaults from config
    run = wandb.init(project=config['training']['project_name'], config=sweep_defaults, name=None) # initialize wandb run with sweep defaults

    _apply_sweep_overrides(config, dict(wandb.config)) # apply sweep overrides to config

    # normalize loss weights if logit keys are present in sweep config
    _normalize_loss_weights(config, dict(wandb.config))

    # get values
    devices, num_nodes, strategy, world = resolve_dist(config)

    # detect gpu type and cap per device batch size
    gpu_type, cap_per_device = _detect_gpu_type_and_cap()
    print(f'[INFO] Detected GPU type: {gpu_type} with per-device batch cap: {cap_per_device}', flush=True)
    _enforce_per_device_batch_cap(config, world, cap_per_device) # enforce per device batch size cap based on gpu type

    # get config values for distributed training
    dist_cfg = config.get('dist', {})
    accelerator = dist_cfg.get('accelerator', 'gpu') # default to gpu if cfg not set
    precision = dist_cfg.get('precision', '32-true') # default to float32 if not set (runs all computation in full float32 instead of using mixed precision)
    accumulate_grad_batches = int(dist_cfg.get('accumulate_grad_batches', 1)) # default to 1 if not set (no gradient accumulation)
    sync_batchnorm = bool(dist_cfg.get('sync_batchnorm', False)) and world > 1 # synchronize batch norm across gpus (default to False if not set)
    deterministic = bool(dist_cfg.get('deterministic', False)) # set to True for reproducibility (may slow down training)

    # compute per-gpu batch size from config
    per_device_batch_size = compute_per_device_batch_size(config, world_override=world)

    # wandb logger
    wandb_logger = WandbLogger(project=config['training']['project_name'])

    # initialize data module
    datamodule = WuCLIPDataModule(
        data_dir=config['data']['data_dir'],
        batch_size=per_device_batch_size, # computed using compute_per_device_batch_size() function and global batch size
        train_frac=config['data']['train_frac'],
        seed=config['training']['seed'],
        data_subset_frac=config['data']['data_subset_frac'],
        text_prompts=config['data']['text_prompts'],
        use_sub_patches=config['data'].get('use_sub_patches', False),
        base_patch_size=config['data'].get('base_patch_size', 96),
        sub_patch_size=config['data'].get('sub_patch_size', 64)
    )
    datamodule.setup()

    # adaptive logging interval
    n_train_batches = ceil(len(datamodule.train_ds) / datamodule.batch_size)
    log_every = max(1, n_train_batches // 5) # log every 5% of training batches
    print(f'[INFO] Setting log_every_n_steps to {log_every} based on {n_train_batches} training batches', flush=True)

    # initialize model
    model = IBOTCLIPPretrainModuleSweep(config)

    # callbacks
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=config['training']['patience'], mode='min'),

        # callback for checkpointing
        ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            save_top_k=1,
            filename=config['model']['save_filename'],
            dirpath=config['model']['save_dirpath'],
            verbose=True
        )
    ]

    # pytorch lightning trainer
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        logger=wandb_logger,
        accelerator=accelerator,
        devices=devices,
        num_nodes=num_nodes,
        strategy=strategy,
        precision=precision,
        accumulate_grad_batches=accumulate_grad_batches,
        sync_batchnorm=sync_batchnorm,
        deterministic=deterministic,
        log_every_n_steps=log_every,
        callbacks=callbacks
    )

    print(f'[DEBUG] GPUs visible={torch.cuda.device_count()}, devices={devices}, num_nodes={num_nodes}, strategy={strategy}', flush=True)

    # start training
    trainer.fit(model, datamodule=datamodule)

    # log best results
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_val_loss = trainer.checkpoint_callback.best_model_score
    wandb_logger.experiment.log({
        'checkpoint_best_val_loss': best_val_loss.item() if best_val_loss else None,
        'checkpoint_best_model_path': best_model_path
    })

    print(f'[INFO] Best model saved to : {best_model_path}', flush=True)
    print(f'[INFO] Best val loss: {best_val_loss}', flush=True)

    # log job end time and total time
    start_epoch = os.getenv('START_EPOCH')
    if start_epoch is not None:
        start_dt = datetime.fromtimestamp(int(start_epoch))
        end_dt = datetime.now()
        total_runtime = end_dt - start_dt
        print(f'[INFO] Total runtime: {total_runtime}', flush=True)

        # log to wandb
        wandb_logger.experiment.log({
            'job_runtime_seconds': total_runtime.total_seconds(),
            'job_runtime': str(total_runtime),
            'job_start_time': start_dt.isoformat(),
            'job_end_time': end_dt.isoformat()
        })
    else:
        print('[WARNING] START_EPOCH not set. Add `export START_EPOCH="$(date +%s)"` to your job script to enable runtime logging.', flush=True)

    # log best val loss
    if best_val_loss is not None:
        wandb_logger.experiment.summary['best_val_loss'] = float(best_val_loss.item())
    else:
        wandb_logger.experiment.summary['best_val_loss'] = None

    # record final epoc val loss
    last_val = None
    try:
        last_val = float(model.trainer.callback_metrics.get('val_loss').item())
    except Exception:
        pass
    wandb_logger.experiment.summary['final_val_loss'] = last_val
    wandb_logger.experiment.summeary['val_loss'] = last_val

    run.finish() # finish the wandb run


        
    




















