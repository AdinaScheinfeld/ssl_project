# /home/ads4015/ssl_project/zeroshot_inference/segmentation/zeroshot_segmentation_single_patch.py - Zero-shot segmentation inference on single 3D image patches using Image+CLIP and other models.

#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import sys

# ----------------- Utilities -----------------

def debug(msg):
    print(f"[DEBUG] {msg}", flush=True)

def info(msg):
    print(f"[INFO] {msg}", flush=True)

# ----------------- Data loading -----------------

def is_label_file(p: Path) -> bool:
    n = p.name.lower()
    return n.endswith('_label.nii') or n.endswith('_label.nii.gz') or '_label.nii' in n

def infer_datatype_from_path(img_path: Path) -> str:
    return img_path.parent.name

def load_nii_as_tensor(path: Path):
    info(f"Loading image: {path}")
    nii = nib.load(str(path))
    arr = nii.get_fdata(dtype=np.float32)
    if arr.ndim == 3:
        vol = arr[None, ...]
    elif arr.ndim == 4 and arr.shape[-1] == 1:
        vol = np.transpose(arr, (3, 0, 1, 2))
    else:
        raise ValueError(f"Unsupported shape {arr.shape}")
    lo, hi = np.percentile(vol, (1.0, 99.0))
    if hi > lo:
        vol = np.clip((vol - lo) / (hi - lo), 0, 1)
    else:
        vol = np.zeros_like(vol)
    return torch.from_numpy(vol), nii.affine, nii.header

# ----------------- CLIP prompts -----------------

def fg_bg_prompts(datatype: str):
    mapping = {
        'amyloid_plaque_patches': 'amyloid plaques',
        'c_fos_positive_patches': 'c-Fos positive neurons',
        'cell_nucleus_patches': 'cell nuclei',
        'vessels_patches': 'blood vessels',
    }
    fg = mapping.get(datatype, datatype.replace('_', ' '))
    return (
        f"a 3D light sheet microscopy image patch showing {fg}",
        "a 3D light sheet microscopy image patch of background tissue"
    )

# ----------------- Model loaders -----------------

def load_clip_model(ckpt, device):
    info(f"Loading Image+CLIP model from {ckpt}")
    sys.path.append('/home/ads4015/ssl_project/models')
    from ibot_clip_pretrain_module import IBOTCLIPPretrainModule
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    cfg = state.get('hyper_parameters') or state.get('hparams')
    model = IBOTCLIPPretrainModule(cfg)
    model.load_state_dict(state.get('state_dict', state), strict=False)
    model.to(device).eval()
    info("Image+CLIP model loaded successfully")
    return model

from monai.networks.nets import SwinUNETR

def load_imgonly_encoder(ckpt, device, feature_size, embed_dim):
    info(f"Loading Image-only encoder from {ckpt}")
    enc = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=embed_dim,
                    feature_size=feature_size, use_checkpoint=False)
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    sd = state.get('state_dict', state)
    enc_sd = enc.state_dict()
    mapped = {}
    for k,v in sd.items():
        kk = k
        for p in ['student_encoder.','encoder.','model.','module.']:
            if kk.startswith(p): kk = kk[len(p):]
        if kk in enc_sd and v.shape == enc_sd[kk].shape:
            mapped[kk] = v
    enc.load_state_dict(mapped, strict=False)
    enc.to(device).eval()
    info(f"Image-only encoder loaded with {len(mapped)} weights")
    return enc

def make_random_encoder(device, feature_size, embed_dim):
    info("Initializing random encoder")
    enc = SwinUNETR(img_size=(96,96,96), in_channels=1, out_channels=embed_dim,
                    feature_size=feature_size, use_checkpoint=False)
    enc.to(device).eval()
    return enc

# ----------------- Zero-shot inference -----------------

def energy_map(encoder, x, eps=1e-6):
    feats = encoder(x)                       # (B, C, d', h', w')
    e = feats.norm(dim=1)                    # (B, d', h', w')
    e = F.interpolate(
        e.unsqueeze(1),
        size=x.shape[2:],
        mode='trilinear',
        align_corners=False
    )                                        # (B, 1, D, H, W)

    # ---- CRITICAL FIX ----
    # Normalize PER VOLUME to [0, 1]
    B = e.shape[0]
    e_flat = e.view(B, -1)
    emin = e_flat.min(dim=1).values.view(B, 1, 1, 1, 1)
    emax = e_flat.max(dim=1).values.view(B, 1, 1, 1, 1)
    e = (e - emin) / (emax - emin + eps)

    return e


@torch.no_grad()
def clip_text_map(model, x, datatype):
    feats = model.student_encoder(x)
    feats = F.normalize(feats, dim=1)
    fg, bg = fg_bg_prompts(datatype)
    tfg = F.normalize(model.encode_text([fg]), dim=-1)
    tbg = F.normalize(model.encode_text([bg]), dim=-1)
    sim = (feats * (tfg - tbg)[:,:,None,None,None]).sum(dim=1)
    sim = F.interpolate(sim.unsqueeze(1), size=x.shape[2:], mode='trilinear', align_corners=False)
    return torch.sigmoid(sim * 5.0)

# ----------------- Main -----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', required=True)
    ap.add_argument('--out_root', required=True)
    ap.add_argument('--clip_ckpt')
    ap.add_argument('--imgonly_ckpt')
    ap.add_argument('--models', default='clip_text,clip_notext,imgonly,random')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    info(f"Using device: {device}")

    img_path = Path(args.image)
    datatype = infer_datatype_from_path(img_path)
    x, aff, hdr = load_nii_as_tensor(img_path)
    x = x.unsqueeze(0).to(device)

    clip_model = None
    if 'clip' in args.models:
        clip_model = load_clip_model(args.clip_ckpt, device)

    imgonly_enc = None
    if 'imgonly' in args.models:
        imgonly_enc = load_imgonly_encoder(args.imgonly_ckpt, device, 24, 512)

    rand_enc = None
    if 'random' in args.models:
        rand_enc = make_random_encoder(device, 24, 512)

    with torch.no_grad():
        for tag in args.models.split(','):
            info(f"Running model variant: {tag}")

            if tag == 'clip_text':
                probs = clip_text_map(clip_model, x, datatype)
            elif tag == 'clip_notext':
                probs = energy_map(clip_model.student_encoder, x)
            elif tag == 'imgonly':
                probs = energy_map(imgonly_enc, x)
            elif tag == 'random':
                probs = energy_map(rand_enc, x)
            else:
                continue

            # ---------------- Output paths ----------------
            out_dir = Path(args.out_root) / datatype / tag / 'preds'
            out_dir.mkdir(parents=True, exist_ok=True)

            base = img_path.name
            if base.endswith('.nii.gz'):
                base = base[:-7]
            elif base.endswith('.nii'):
                base = base[:-4]

            prob_path = out_dir / f"{base}_prob.nii.gz"
            pred_path = out_dir / f"{base}_pred_thr0.50.nii.gz"

            # ---------------- Save probability map ----------------
            nib.save(
                nib.Nifti1Image(
                    probs.detach().squeeze().cpu().numpy(),
                    aff,
                    hdr
                ),
                prob_path
            )
            info(f"Saved probability map to {prob_path}")

            # ---------------- Save binary segmentation (thr=0.5) ----------------
            mask = (probs >= 0.5)
            nib.save(
                nib.Nifti1Image(
                    mask.detach().squeeze().cpu().numpy(),
                    aff,
                    hdr
                ),
                pred_path
            )
            info(f"Saved binary mask to {pred_path}")




if __name__ == '__main__':
    main()
