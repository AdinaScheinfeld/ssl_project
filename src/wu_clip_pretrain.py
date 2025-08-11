# Wu Data Pretraining

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
sys.path.append('/home/ads4015/ssl_project/models')
from ibot_clip_pretrain_module import IBOTCLIPPretrainModule

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

    # get values
    devices = int(dist.get('devices', 1)) # number of gpus per node
    num_nodes = int(dist.get('num_nodes', 1)) # number of nodes
    strategy = 'auto'

    # get values if using multiple gpus
    if use_multi:
        devices = int(os.environ.get('SLURM_NTASKS_PER_NODE', devices)) # number of gpus per node from slurm env
        num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', os.environ.get('SLURM_NNODES', num_nodes))) # number of nodes from slurm env
        strategy = DDPStrategy(find_unused_parameters=True)
        world = max(1, devices * num_nodes) # total number of gpus across all nodes
    
    # if using single gpu, set devices to 1
    else:
        devices, num_nodes, world = 1, 1, 1
        strategy = 'auto'

    # return values
    return devices, num_nodes, strategy, world

    

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

    # get values
    devices, num_nodes, strategy, world = resolve_dist(config)

    # get config values for distributed training
    dist_cfg = config.get('dist', {})
    accelerator = dist_cfg.get('accelerator', 'gpu') # default to gpu if cfg not set
    # devices = dist_cfg.get('devices', 1) # number of gpus per node, default to 1 if not set
    # num_nodes = dist_cfg.get('num_nodes', 1) # number of nodes, default to 1 if not set
    # strategy = dist_cfg.get('strategy', 'ddp') if int(devices) > 1 or int(num_nodes) > 1 else 'auto' # use 'auto' for single gpu training
    precision = dist_cfg.get('precision', '32-true') # default to float32 if not set (runs all computation in full float32 instead of using mixed precision)
    accumulate_grad_batches = int(dist_cfg.get('accumulate_grad_batches', 1)) # default to 1 if not set (no gradient accumulation)
    sync_batchnorm = bool(dist_cfg.get('sync_batchnorm', False)) and world >1 # synchronize batch norm across gpus (default to False if not set)
    deterministic = bool(dist_cfg.get('deterministic', False)) # set to True for reproducibility (may slow down training)

    # # if slurm is spawning ranks, align devices with n-tasks-per-node to satisfy lightning
    # slurm_ntasks_per_node = os.environ.get('SLURM_NTASKS_PER_NODE')
    # if slurm_ntasks_per_node:
    #     try:
    #         devices = int(slurm_ntasks_per_node)
    #     except ValueError:
    #         pass

    # strategy = DDPStrategy(find_unused_parameters=True)

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
    model = IBOTCLIPPretrainModule(config)

    # callbacks
    callbacks = [
        # callback for early stopping
        # early stopping is triggered when loss does not decrease for `patience` consecutive epochs
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
        # 'best_train_loss': best_train_loss,
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


        
    




















