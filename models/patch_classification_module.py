# patch_classification_module.py - Module class for multiclass patch-based image classification tasks

# --- Setup ---

# imports
import torch
import torch.nn as nn
from torch.optim import AdamW
from torchmetrics.classification import MulticlassAccuracy

from monai.networks.nets import SwinUNETR

import pytorch_lightning as pl

# --- Helper Functions ---

# function to strip module prefixes from state dict
def _strip_module_prefixes(state_dict, prefixes=('model.', 'module.', 'encoder.', 'net.')):

    # dict for new state
    new_state_dict = {}

    # iterate over items
    for key, value in state_dict.items():

        # strip prefixes
        new_key = key
        for prefix in prefixes:
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix):]

        # add to new state dict
        new_state_dict[new_key] = value

    return new_state_dict

# function to keep only encoder keys (not decoder or head)
def _keep_encoder_keys(state_dict):
    return {k: v for k, v in state_dict.items() if k.startswith('swinViT.')}

# function to load swinViT weights into SwinUNETR instance
def _load_backbone_encoder_from_ckpt(model, ckpt_path):

    # load checkpoint
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('state_dict', ckpt)

    # strip prefixes
    state_dict = _strip_module_prefixes(state_dict)

    # keep only encoder keys
    state_dict = _keep_encoder_keys(state_dict)

    # load state dict into model and check for missing/unexpected keys and count how many keys were loaded
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    loaded = sum(1 for k in state_dict.keys() if k in model.state_dict())

    # return info
    return loaded, missing, unexpected


# --- Module Class ---

# multiclass patch classification module on top of monai SwinUNETR encoder
class PatchClassificationModule(pl.LightningModule):

    # init
    def __init__(
            self,
            num_classes, # number of target classes (i.e. length of subtype folders)
            lr=1e-4,
            weight_decay=1e-5,
            pretrained_ckpt=None, # path to pretrained checkpoint to load encoder weights from, only used if init_mode is 'pretrained'
            feature_size=24, # must match the feature size used during pretraining
            class_names=None,
            freeze_encoder_epochs=0, # number of epochs to freeze encoder weights for
            linear_probe=False, # if True, only train classification head (i.e. freeze encoder for entire training)
            init_mode='pretrained', # 'pretrained' or 'random' initialization
            in_channels=1, # number of input image channels (1 for grayscale, 3 for RGB)
            class_weights=None, # optional list of class weights for loss function
    ):
        
        super().__init__()
        self.save_hyperparameters()

        # create swin unetr model
        self.backbone = SwinUNETR(
            img_size=(96, 96, 96),
            in_channels=in_channels,
            out_channels=2, # decoder out channels (not used)
            feature_size=feature_size,
            use_checkpoint=False,
        )

        # build pooling and head after know true encoder channel dim
        self.pool = nn.AdaptiveAvgPool3d(1) # global average pooling (C, D, H, W) -> (C)

        # infer deepest channel dim
        with torch.no_grad():
            dummy_input = torch.zeros((1, in_channels, 96, 96, 96))
            feats = self.backbone.swinViT(dummy_input)
            deepest = feats[-1] if isinstance(feats, (list, tuple)) else feats
            encoder_channel_dim = deepest.shape[1]
        self.head = nn.Linear(encoder_channel_dim, num_classes) # classification head

        # use class weights in loss if provided
        if class_weights is not None:
            class_weights_tensor = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer('class_weights', class_weights_tensor)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss() # cross-entropy loss for multiclass classification

        self.class_names = class_names or [str(i) for i in range(num_classes)] # labels corresponding to each class index

        self._freeze_encoder_epochs = int(freeze_encoder_epochs) # number of epochs to freeze encoder weights for (warm start)
        self._linear_probe = bool(linear_probe) # whether to use linear probing (i.e. freeze encoder and only train classifier head)

        # initialize weights
        if init_mode not in ('pretrained', 'random'):
            raise ValueError(f"Invalid init_mode '{init_mode}', must be 'pretrained' or 'random'")

        # load pretrained weights if specified
        if init_mode == 'pretrained':
            if not pretrained_ckpt:
                print('[WARN] init_mode is "pretrained" but no pretrained_ckpt path provided, using random initialization instead', flush=True)

            else:
                try:
                    loaded, missing, unexpected = _load_backbone_encoder_from_ckpt(self.backbone, pretrained_ckpt)
                    print(f'[INFO] Loaded encoder from {pretrained_ckpt} (loaded={loaded}, missing={missing}, unexpected={unexpected})', flush=True)
                except Exception as e:
                    print(f'[ERROR] Failed to load pretrained encoder from {pretrained_ckpt}, using random initialization instead', flush=True)
            
        else:
            print('[INFO] Using random initialization for all model weights', flush=True)

        # torchmetrics
        self.train_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=num_classes)
        self.test_accuracy = MulticlassAccuracy(num_classes=num_classes)

    # *** Freezing Policy ***

    # function to set requires_grad for encoder parameters
    def _set_encoder_requires_grad(self, flag):
        for p in self.backbone.swinViT.parameters():
            p.requires_grad = flag

    # on train start
    def on_train_start(self):
        
        # determine if using linear probing
        if self._linear_probe:
            print('[INFO] Using linear probing: freezing encoder weights for entire training', flush=True)
            self._set_encoder_requires_grad(False)

        # otherwise freeze encoder for specified number of epochs
        elif self._freeze_encoder_epochs > 0 and self.current_epoch < self._freeze_encoder_epochs:
            print(f'[INFO] Freezing encoder weights for epoch {self.current_epoch} / {self._freeze_encoder_epochs}.', flush=True)
            self._set_encoder_requires_grad(False)

    # on train epoch start
    def on_train_epoch_start(self):

        # ignore if using linear probing
        if self._linear_probe:
            return
        
        # unfreeze encoder if past freeze epochs
        if self._freeze_encoder_epochs > 0:
            still_freeze = self.current_epoch < self._freeze_encoder_epochs
            self._set_encoder_requires_grad(not still_freeze)
            if still_freeze:
                print(f'[INFO] Freezing encoder weights for epoch {self.current_epoch} / {self._freeze_encoder_epochs}.', flush=True)

    # *** Forward Pass ***

    # function to run the swin transformer and take the deepest features
    def _encode(self, x):
        feats = self.backbone.swinViT(x) # get features from swinViT encoder (deepest is last)
        if isinstance(feats, (list, tuple)):
            z = feats[-1] # take deepest features
        else:
            z = feats
        return z

    # forward
    def forward(self, x):
        z = self._encode(x) # encode input [B, C, D, H, W] -> [B, C', D', H', W']
        z = self.pool(z) # global average pool to [B, C', 1, 1, 1]
        z = torch.flatten(z, 1) # flatten to [B, C']
        logits = self.head(z) # classification head [B, num_classes]
        return logits
    
    # *** Steps ***

    # shared step function
    def _step(self, batch, stage):

        # get data, logits, loss, and preds
        x, y = batch['image'], batch['label']
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        # log metrics for train
        if stage == 'train':
            self.train_accuracy.update(preds, y)
            self.log(f'train_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'train_accuracy', self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        
        # log metrics for val
        elif stage == 'val':
            self.val_accuracy.update(preds, y)
            self.log(f'val_loss', loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
            self.log(f'val_accuracy', self.val_accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)

        # log metrics for test
        elif stage == 'test':
            self.test_accuracy.update(preds, y)
            self.log(f'test_loss', loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log(f'test_accuracy', self.test_accuracy, prog_bar=True, on_step=False, on_epoch=True)

        return loss
    
    # training step
    def training_step(self, batch, batch_idx):
        return self._step(batch, stage='train')

   # validation step
    def validation_step(self, batch, batch_idx):
        return self._step(batch, stage='val')

    # test step
    def test_step(self, batch, batch_idx):
        return self._step(batch, stage='test')
    
    # *** Optimizer ***

    # configure optimizers
    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)









