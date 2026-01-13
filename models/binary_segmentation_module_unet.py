# binary_segmentation_module_unet.py - Binary segmentation module for finetuning model using Selma data

# --- Setup ---

# imports
import os
import wandb

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from monai.data.meta_tensor import MetaTensor
from monai.losses import DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric
from monai.networks.nets import Unet

import torch
from torch.serialization import add_safe_globals


# --- Model ---

class BinarySegmentationModuleUnet(pl.LightningModule):

    # init
    def __init__(
        self, 
        pretrained_ckpt=None, 
        lr=1e-4, 
        freeze_encoder_epochs=0, 
        encoder_lr_mult=0.5, 
        loss_name='dicece',
        unet_channels=(32, 64, 128, 256, 512),
        unet_strides=(2, 2, 2, 2),
        unet_num_res_units=2,
        unet_norm='INSTANCE'
        ):

        super().__init__()
        self.save_hyperparameters()

        self.freeze_encoder_epochs = freeze_encoder_epochs
        self.encoder_frozen = freeze_encoder_epochs > 0

        # trackers/counters
        self.logged_images = 0 # counter for number of logged images in validation step
        self.best_train_loss = float('inf')
        self.best_val_loss = float('inf')
        self.best_ckpt = None

        # define model
        self.model = Unet(
            spatial_dims=3,
            in_channels=1,
            out_channels=1,
            channels=tuple(unet_channels),
            strides=tuple(unet_strides),
            num_res_units=int(unet_num_res_units),
            norm=str(unet_norm)
        )

        # learning rate
        self.lr = lr
        self.encoder_lr_mult = float(encoder_lr_mult)
        self.loss_name = str(loss_name).lower()

        # scheduler
        self._total_steps = None
        self._warmup_steps = None

        # loss function and metric
        if self.loss_name == 'dicefocal':
            self.loss_fn = DiceFocalLoss(sigmoid=True, lambda_dice=0.5, lambda_focal=0.5, alpha=0.25, gamma=2.0) # alpha (class balance) and gamma (focusing) for focal
            print(f'[INFO] Using Dice + Focal loss for finetuning', flush=True)
        else:
            self.loss_fn = DiceCELoss(sigmoid=True)
            print(f'[INFO] Using Dice + CE loss for finetuning', flush=True)
        self.val_dice = DiceMetric(include_background=False, reduction='mean')

        # freeze encoder during warmup (freezes everything except final segmentation conv)
        if self.encoder_frozen:
            print(f'[INFO] Freezing encoder for first {self.freeze_encoder_epochs} epochs', flush=True)
            for name, param in self.model.named_parameters():
                
                # keep segmentation head trainable (last conv layer)
                if not name.startswith('model.2.0.conv.'):
                    param.requires_grad = False
                    print(f'  [DEBUG] Freezing parameter: {name}', flush=True)

        # load pretrained weights if provided
        if pretrained_ckpt:
            print(f"[INFO] Loading checkpoint from: {pretrained_ckpt}", flush=True)
            add_safe_globals([MetaTensor])
            ckpt = torch.load(pretrained_ckpt, weights_only=False, map_location='cpu')
            state_dict = ckpt.get('state_dict', ckpt)

            print(f'[DEBUG] ckpt key samples: {list(state_dict.keys())[:5]}', flush=True)
            print(f'[DEBUG] ckpt has model.*: {any(k.startswith("model.") for k in state_dict.keys())}', flush=True)
            print(f'[DEBUG] ckpt has student_encoder.*: {any(k.startswith("student_encoder.") for k in state_dict.keys())}', flush=True)

            # normlize  prefixes commonly added by lightning
            POSSIBLE_WRAPPERS = ['module.', 'student_model.', 'student.', 'teacher_model.', 'net.', 'encoder.', 'backbone.']

            # function to strip prefix
            def strip_prefix(k):
                changed = True
                while changed:
                    changed = False
                    for p in POSSIBLE_WRAPPERS:
                        if k.startswith(p):
                            k = k[len(p):]
                            changed = True
                return k

            # mapping
            mapped = {}
            for k, v in state_dict.items():
                k2 = strip_prefix(k)

                # if checkpoint has unet keys (like model.0.conv)
                if k2.startswith('model.'):
                    mapped[k2] = v
                    continue

                # if checkpoint has student_encoder keys
                if k2.startswith('student_encoder.'):
                    rest = k2[len('student_encoder.'):]
                    if rest.startswith('model.'):
                        mapped_k = rest # already has model. prefix
                    else:
                        mapped_k = 'model.' + rest # add model. prefix
                    mapped[mapped_k] = v
                    continue
                    

            # look at keys that are loaded
            if mapped:
                sample_keys = list(mapped.keys())[:5]
                # print(f'[DEBUG] Sample encoder keys: {sample_keys}', flush=True)
            else:
                print(f'[DEBUG] No encoder keys found in checkpoint.', flush=True)

            # filter before loading state dict
            model_sd = self.model.state_dict()
            safe_mapped, dropped = {}, []

            for k, v in mapped.items():
                if k not in model_sd:
                    dropped.append((k, 'not in model'))
                    continue
                
                if v.shape != model_sd[k].shape:
                    dropped.append((k, f'ckpt {tuple(v.shape)} vs model {tuple(model_sd[k].shape)}'))
                    continue
                safe_mapped[k] = v

            incompatible = self.model.load_state_dict(safe_mapped, strict=False)

            # inspect loading results
            if incompatible.missing_keys:
                print(f'[DEBUG] Missing keys after loading: {incompatible.missing_keys}', flush=True)
            if incompatible.unexpected_keys:
                print(f'[DEBUG] Unexpected keys after loading: {incompatible.unexpected_keys}', flush=True)

            print(f'[INFO] Loaded pretrained encoder with kept={len(safe_mapped)}, dropped={len(dropped)}, missing={len(incompatible.missing_keys)}, unexpected={len(incompatible.unexpected_keys)}', flush=True)

            if dropped:
                print('[WARN] Dropped due to shape mismatch:', flush=True)
                for k, msg in dropped[:10]:
                    print(' ', k, '->', msg, flush=True)
                if len(dropped) > 10:
                    print(f' ... (+{len(dropped)-10} more)', flush=True)

        # indicate if no pretrained checkpoint provided
        else:
            print(f'[WARN] No pretrained checkpoint provided, training from scratch.', flush=True)

    # forward pass
    def forward(self, x):
        return self.model(x)


    # on train epoch start
    def on_train_epoch_start(self):

        # unfreeze encoder after specified number of epochs
        if self.encoder_frozen and self.current_epoch >= self.freeze_encoder_epochs:
            print(f'[INFO] Unfreezing encoder at epoch {self.current_epoch}', flush=True)
            for name, param in self.model.named_parameters():
                param.requires_grad = True
            self.encoder_frozen = False

    
    # training step
    def training_step(self, batch, batch_idx):
        assert batch['image'].shape[1] == 1, f'Expected image to have 1 channel, but got {batch["image"].shape[1]} channels'
        logits = self.forward(batch['image'])
        train_loss = self.loss_fn(logits, batch['label'].float()) # compute loss
        batch_size = batch['image'].shape[0]
        self.log('train_loss', train_loss, on_step=True, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.best_train_loss = min(self.best_train_loss, float(train_loss.detach().cpu()))
        return train_loss
    
    # val step
    def validation_step(self, batch, batch_idx):

        assert batch['image'].shape[1] == 1, f'Expected image to have 1 channel, but got {batch["image"].shape[1]} channels'
        
        logits = self.forward(batch['image'])
        val_loss = self.loss_fn(logits, batch['label'].float()) # compute loss
        batch_size = batch['image'].shape[0]
        self.log('val_loss', val_loss, on_epoch=True, prog_bar=True, batch_size=batch_size)
        self.best_val_loss = min(self.best_val_loss, val_loss.item())

        # thresholded dice at 0.5
        probs = torch.sigmoid(logits)
        binary_preds = (probs > 0.5).float()
        self.val_dice(y_pred=binary_preds, y=batch['label'].float())

        # stash a few batches for threshold sweep at epoch end
        if not hasattr(self, '_val_store'):
            self._val_store = []

        # take up to 2 samples from first few batches
        if len(self._val_store) < 4:
            num_to_take = min(2, probs.shape[0])
            self._val_store.append((
                probs[:num_to_take].detach().to('cpu', dtype=torch.float32), 
                batch['label'][:num_to_take].detach().to('cpu', dtype=torch.float32)
            ))

        # log some images to wandb
        if self.logged_images < 5:

            # prepare tensors for visualization (detach metatensor, move to cpu, cast to float)
            preds = torch.sigmoid(logits.detach().to(dtype=torch.float32)) > 0.5
            preds = preds.to(dtype=torch.float32)
            images = batch['image'][:, 0:1] # use only first channel for visualization
            labels = batch['label']
    
            # drop metatensor wrapper if present
            if isinstance(images, MetaTensor):
                images = images.as_tensor()
            if isinstance(labels, MetaTensor):
                labels = labels.as_tensor()
            if isinstance(preds, MetaTensor):
                preds = preds.as_tensor()

            # move to cpu float32 before numpy
            images = images.detach().to(device='cpu', dtype=torch.float32, copy=True)
            labels = labels.detach().to(device='cpu', dtype=torch.float32, copy=True)
            preds = preds.detach().to(device='cpu', dtype=torch.float32, copy=True)

            num_to_log = min(5 - self.logged_images, images.shape[0])

            for i in range(num_to_log):

                # log center slice
                img_np = images[i, 0].numpy()
                lbl_np = labels[i].numpy().squeeze()
                pred_np = preds[i, 0].numpy()
                mid = img_np.shape[0] // 2

                # print(f'Image: {img_np.shape}, Label: {lbl_np.shape}, Pred: {pred_np.shape}', flush=True)

                self.val_table.add_data(
                    batch['filename'][i],
                    wandb.Image(img_np[mid]),
                    wandb.Image(lbl_np[mid]),
                    wandb.Image(pred_np[mid])
                )
                self.logged_images += 1

        return val_loss
    
    # on validation epoch start
    def on_validation_epoch_start(self):
        
        # create counter and wandb table to log images after validation epoch
        self.logged_images = 0
        self.val_table = wandb.Table(columns=['Filename', 'Image', 'Label', 'Prediction'])
        self._val_store = [] # reset stored val batches for threshold sweep
    

    # on validation epoch end
    def on_validation_epoch_end(self):
        
        # log val image table to wandb
        if self.logged_images > 0:
            self.logger.experiment.log({f'val_examples_{self.current_epoch}': self.val_table})

        # thresholded dice at 0.5
        try:
            dice = float(self.val_dice.aggregate().item())
        except Exception:
            dice = None
        self.val_dice.reset()
        if dice is not None:
            self.log('val_dice_050', dice, prog_bar=True)

        # sweep thresholds on stored batches
        if self._val_store:
            with torch.no_grad():
                probs_cpu = torch.cat([p for p, _ in self._val_store], dim=0)
                labels_cpu = torch.cat([y for _, y in self._val_store], dim=0)
                eps = torch.finfo(probs_cpu.dtype).eps
                best_dice, best_t = 0.0, 0.5
                for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
                    preds = (probs_cpu > t).float()
                    intersection = (preds * labels_cpu).sum()
                    denom = preds.sum() + labels_cpu.sum() + eps
                    dice_t = (2.0 * intersection / denom).item()
                    if dice_t > best_dice:
                        best_dice, best_t = dice_t, t
                self.log('val_dice_best_t', best_dice, prog_bar=False)
                self.log('val_best_t', float(best_t), prog_bar=False)

        # update best val loss from aggregated epoch metric
        mv = self.trainer.callback_metrics.get('val_loss')
        if mv is not None:
            try:
                vv = float(mv.detach().cpu())
            except Exception:
                vv = float(mv)
            if vv < self.best_val_loss:
                self.best_val_loss = vv

    
    # on train epoch end
    def on_train_epoch_end(self):

        # update best train loss from aggregated epoch metric
        m = (self.trainer.callback_metrics.get('train_loss_epoch') or self.trainer.callback_metrics.get('train_loss'))
        if m is not None:
            try:
                v = float(m.detach().cpu())
            except Exception:
                v = float(m)
            if v < self.best_train_loss:
                self.best_train_loss = v


    # configure optimizers
    def configure_optimizers(self):

        print(f'[INFO] Configuring optimizers with encoder_lr_mult={self.encoder_lr_mult}', flush=True)
        
        # split params into backbone and head
        backbone_params, head_params = [], []
        for name, param in self.model.named_parameters():
            if name.startswith('model.2.0.conv.'): # final conv layer as head
                head_params.append(param)
            else:
                backbone_params.append(param)

        # guard against no params in head
        if len(head_params) == 0:
            raise RuntimeError(f'No head params match prefix "model.2.0.conv." for optimizer configuration.', flush=True)

        base_lr = float(self.lr)
        optimizer = torch.optim.AdamW(
            [
                {'params': backbone_params, 'lr': base_lr * self.encoder_lr_mult}, # lower LR for backbone (learns more gently)
                {'params': head_params, 'lr': base_lr} # higher LR for head
            ],
            weight_decay=1e-4,
            betas=(0.9, 0.999)
        )

        # use cosine scheduler with linear warmup
        def _lr_lambda(step):

            # guard for first calls before on_fit_start populates total steps
            total = self._total_steps or 1
            warmup = self._warmup_steps or 1
            step = max(0, step)
            if step < warmup:
                return float(step + 1) / float(warmup) # linear warmup
            
            # cosine over remaining steps
            t = (step - warmup) / float(max(1, total - warmup))

            # 0.5 * (1 + cos(pi * t)) in [1..0]
            return 0.5 * (1.0 + torch.cos(torch.pi * torch.tensor(t))).item()
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step', # called after each optimizer step
                'frequency': 1
            }
        }
    

    # on train start
    def on_train_start(self):

        # print first step lrs for encoder and decoder
        optimizer = self.trainer.optimizers[0]
        encoder_lr = optimizer.param_groups[0]['lr']
        decoder_lr = optimizer.param_groups[1]['lr']
        print(f'[INFO] Starting training with encoder_lr={encoder_lr}, decoder_lr={decoder_lr}', flush=True)


    # on fit start
    def on_fit_start(self):

        # compute total training steps for schedulers that run "per step"
        try:
            self._total_steps = int(self.trainer.estimated_stepping_batches)
        except Exception:
            
            # rough fallback estimate
            steps_per_epoch = max(1, len(self.trainer.datamodule.train_dataloader()) if self.trainer.datamodule else len(self.trainer.fit_loop._combined_loader))
            self._total_steps = steps_per_epoch * max(1, self.trainer.max_epochs)

        # warmup (~3% of total steps)
        self._warmup_steps = max(1, int(0.03 * self._total_steps))
        print(f'[INFO] LR schedule: total_steps={self._total_steps}, warmup_steps={self._warmup_steps}', flush=True)

    
    # after complete training
    def on_fit_end(self):

        # find best checkpoint from callbacks
        for cb in self.trainer.callbacks:
            if isinstance(cb, pl.callbacks.ModelCheckpoint):
                monitor_name = getattr(cb, 'monitor', None)
                self.best_ckpt = getattr(cb, 'best_model_path', None)
                best_val_score = getattr(cb, 'best_model_score', None)
                break

        # prefer checkpoint's epoch aggregated val for consistency with saved best
        if 'best_val_loss' in self.__dict__ and best_val_score is not None:
            best_val_loss_out = float(best_val_score.cpu())
        else:
            best_val_loss_out = self.best_val_loss

        payload = {
            'best_train_loss': self.best_train_loss,
            'best_val_monitored': float(best_val_score.cpu()) if best_val_score is not None else None,
            'monitor': monitor_name
        }
        if self.best_ckpt:
            payload['best_model_path'] = self.best_ckpt


        # print best metrics
        print(f'[INFO] Training complete:', flush=True)
        for k, v in payload.items():
            print(f'  {k}: {v}', flush=True)

        # log to wandb summary
        if isinstance(self.logger, WandbLogger):
            self.logger.experiment.summary.update(payload)
    










