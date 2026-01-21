# patch_classification_module_unet.py - UNet-backbone multiclass patch classification

import torch
import torch.nn as nn
from typing import Optional
from torch.optim import AdamW
from torchmetrics.classification import MulticlassAccuracy
import pytorch_lightning as pl

from monai.networks.nets import Unet


# -------------------------
# checkpoint loading utils
# -------------------------

def _strip_module_prefixes(state_dict, prefixes=("student_encoder.", "ema_student_encoder.", "model.", "module.", "net.", "encoder.")):
    new_sd = {}
    for k, v in state_dict.items():
        nk = k
        for p in prefixes:
            if nk.startswith(p):
                nk = nk[len(p):]
        new_sd[nk] = v
    return new_sd


def _load_unet_backbone_from_ckpt(unet: Unet, ckpt_path: str):
    """
    Your UNet pretraining checkpoints store weights as: student_encoder.<rest>
    In your segmentation finetune you mapped: student_encoder.<rest> -> model.<rest>
    We try that mapping first, then fallback to direct <rest>.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)

    # (1) keep only student_encoder.* keys
    student = {k: v for k, v in sd.items() if k.startswith("student_encoder.")}
    if not student:
        # fallback: maybe you saved weights without that prefix
        student = sd

    # strip known wrappers AFTER filtering
    student = _strip_module_prefixes(student, prefixes=("student_encoder.", "ema_student_encoder.", "module.", "net.", "encoder."))

    model_sd = unet.state_dict()

    # Attempt A: "model.<rest>" (matches your segmentation loader convention)
    mappedA = {f"model.{k}": v for k, v in student.items()}
    safeA = {k: v for k, v in mappedA.items() if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape)}
    loadA = len(safeA)

    # Attempt B: "<rest>" directly
    safeB = {k: v for k, v in student.items() if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape)}
    loadB = len(safeB)

    if loadA >= loadB:
        safe = safeA
        strategy = "model.<rest>"
    else:
        safe = safeB
        strategy = "<rest>"

    missing, unexpected = unet.load_state_dict(safe, strict=False)
    loaded = sum(1 for k in safe.keys() if k in model_sd)

    print(
        f"[INFO] UNet load ({strategy}): loaded={loaded} "
        f"missing={len(missing)} unexpected={len(unexpected)} total_ckpt_keys={len(student)}",
        flush=True,
    )
    if missing:
        print(f"[DEBUG] Missing keys (first 30): {missing[:30]}", flush=True)
    if unexpected:
        print(f"[DEBUG] Unexpected keys (first 30): {unexpected[:30]}", flush=True)

    return loaded, missing, unexpected


# -------------------------
# module
# -------------------------

class PatchClassificationModuleUNet(pl.LightningModule):
    """
    UNet backbone -> bottleneck feature map -> GAP -> Linear head
    Mirrors your SwinUNETR PatchClassificationModule API.
    """

    def __init__(
        self,
        num_classes: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
        pretrained_ckpt: Optional[str] = None,
        class_names=None,
        freeze_encoder_epochs: int = 0,
        linear_probe: bool = False,
        init_mode: str = "pretrained",
        in_channels: int = 1,
        # UNet arch
        unet_channels=(32, 64, 128, 256),
        unet_strides=(2, 2, 2, 1),
        unet_num_res_units: int = 2,
        unet_norm: str = "BATCH",
        class_weights=None,
        img_size=(96, 96, 96),
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["class_names", "class_weights"])

        self.num_classes = int(num_classes)
        self.class_names = class_names or [str(i) for i in range(self.num_classes)]

        # build UNet backbone (out_channels is irrelevant; we will hook bottleneck features)
        self.backbone = Unet(
            spatial_dims=3,
            in_channels=in_channels,
            out_channels=1,
            channels=tuple(int(x) for x in unet_channels),
            strides=tuple(int(x) for x in unet_strides),
            num_res_units=int(unet_num_res_units),
            norm=str(unet_norm),
        )

        # loss
        if class_weights is not None:
            cw = torch.as_tensor(class_weights, dtype=torch.float32)
            self.register_buffer("class_weights", cw)
            self.criterion = nn.CrossEntropyLoss(weight=self.class_weights)
        else:
            self.criterion = nn.CrossEntropyLoss()

        # freeze policy
        self._freeze_encoder_epochs = int(freeze_encoder_epochs)
        self._linear_probe = bool(linear_probe)

        # init mode
        if init_mode not in ("pretrained", "random"):
            raise ValueError("init_mode must be 'pretrained' or 'random'")

        if init_mode == "pretrained":
            if not pretrained_ckpt:
                print("[WARN] init_mode=pretrained but no pretrained_ckpt provided; using random init.", flush=True)
            else:
                try:
                    _load_unet_backbone_from_ckpt(self.backbone, pretrained_ckpt)
                except Exception as e:
                    print(f"[ERROR] Failed to load UNet pretrained ckpt: {e} (using random init)", flush=True)
        else:
            print("[INFO] Using random initialization for UNet backbone.", flush=True)

        # ---- pick bottleneck feature tensor automatically ----
        # We attach temporary hooks on modules that output 5D tensors and choose:
        # - smallest spatial volume (D*H*W), tie-break by largest C
        self._bottleneck_name = None
        self._bottleneck_channels = None
        self._capture = None

        self._infer_and_set_bottleneck(in_channels=in_channels, img_size=img_size)

        # pooling + head
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Linear(int(self._bottleneck_channels), self.num_classes)

        # metrics
        self.train_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.val_accuracy = MulticlassAccuracy(num_classes=self.num_classes)
        self.test_accuracy = MulticlassAccuracy(num_classes=self.num_classes)

    def _infer_and_set_bottleneck(self, in_channels: int, img_size):
        candidates = []

        def make_hook(name):
            def hook(_m, _inp, out):
                if torch.is_tensor(out) and out.ndim == 5:
                    # out: [B,C,D,H,W]
                    B, C, D, H, W = out.shape
                    spatial = int(D) * int(H) * int(W)
                    candidates.append((spatial, int(C), name))
            return hook

        hooks = []
        for name, m in self.backbone.named_modules():
            hooks.append(m.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            x = torch.zeros((1, in_channels, *img_size), dtype=torch.float32)
            _ = self.backbone(x)

        for h in hooks:
            h.remove()

        if not candidates:
            # fallback: use UNet output (not ideal, but keeps code runnable)
            print("[WARN] No 5D activations captured; falling back to using UNet output as features.", flush=True)
            self._bottleneck_name = "__UNET_OUTPUT__"
            with torch.no_grad():
                x = torch.zeros((1, in_channels, *img_size), dtype=torch.float32)
                y = self.backbone(x)
            self._bottleneck_channels = int(y.shape[1])
            print(f"[INFO] Bottleneck fallback channels={self._bottleneck_channels}", flush=True)
            return

        # choose best candidate
        # smallest spatial, then largest channels
        candidates.sort(key=lambda t: (t[0], -t[1]))
        spatial, C, name = candidates[0]
        self._bottleneck_name = name
        self._bottleneck_channels = C
        print(f"[INFO] UNet bottleneck module='{name}' spatial={spatial} C={C}", flush=True)

        # persistent hook for forward
        self._capture = {"z": None}

        def persistent_hook(_m, _inp, out):
            if torch.is_tensor(out) and out.ndim == 5:
                self._capture["z"] = out

        # attach hook only on the chosen module
        for n, m in self.backbone.named_modules():
            if n == name:
                m.register_forward_hook(persistent_hook)
                break

    # -------------------------
    # freeze policy
    # -------------------------

    def _set_backbone_requires_grad(self, flag: bool):
        for p in self.backbone.parameters():
            p.requires_grad = flag

    def on_train_start(self):
        if self._linear_probe:
            print("[INFO] Linear probe: freezing backbone for entire training.", flush=True)
            self._set_backbone_requires_grad(False)
        elif self._freeze_encoder_epochs > 0 and self.current_epoch < self._freeze_encoder_epochs:
            print(f"[INFO] Freezing backbone for epoch {self.current_epoch}/{self._freeze_encoder_epochs}", flush=True)
            self._set_backbone_requires_grad(False)

    def on_train_epoch_start(self):
        if self._linear_probe:
            return
        if self._freeze_encoder_epochs > 0:
            still_freeze = self.current_epoch < self._freeze_encoder_epochs
            self._set_backbone_requires_grad(not still_freeze)
            if still_freeze:
                print(f"[INFO] Freezing backbone for epoch {self.current_epoch}/{self._freeze_encoder_epochs}", flush=True)

    # -------------------------
    # forward
    # -------------------------

    def _encode(self, x):
        self._capture["z"] = None if self._capture is not None else None
        y = self.backbone(x)

        if self._bottleneck_name == "__UNET_OUTPUT__":
            return y

        z = None if self._capture is None else self._capture.get("z", None)
        if z is None:
            # fallback if hook didn’t fire for some reason
            return y
        return z

    def forward(self, x):
        z = self._encode(x)          # [B,C,D,H,W]
        z = self.pool(z)             # [B,C,1,1,1]
        z = torch.flatten(z, 1)      # [B,C]
        logits = self.head(z)        # [B,K]
        return logits

    # -------------------------
    # steps
    # -------------------------

    def _step(self, batch, stage: str):
        x, y = batch["image"], batch["label"]
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)

        if stage == "train":
            self.train_accuracy.update(preds, y)
            self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("train_accuracy", self.train_accuracy, prog_bar=True, on_step=False, on_epoch=True)
        elif stage == "val":
            self.val_accuracy.update(preds, y)
            self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
            self.log("val_accuracy", self.val_accuracy, prog_bar=True, on_step=False, on_epoch=True, sync_dist=False)
        elif stage == "test":
            self.test_accuracy.update(preds, y)
            self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
            self.log("test_accuracy", self.test_accuracy, prog_bar=True, on_step=False, on_epoch=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    # -------------------------
    # optim
    # -------------------------

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=float(self.hparams.lr), weight_decay=float(self.hparams.weight_decay))




