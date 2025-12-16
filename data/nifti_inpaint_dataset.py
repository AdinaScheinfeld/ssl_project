# /home/ads4015/ssl_project/data/nifti_inpaint_dataset.py - A dataset class for handling NIfTI images for inpainting tasks.

# --- Setup ---

# imports
from dataclasses import dataclass
import hashlib
import json
import nibabel as nib
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset


# --- Helper Functions ---

# function to get stable int hashing
def _stable_int_hash(s, mod=2**31):
    h = hashlib.sha1(s.encode('utf-8')).hexdigest()
    return int(h[:8], 16) % mod

# function to load NIfTI image and return as numpy array
def _load_nifti(path):
    nifti = nib.load(str(path)) # load nifti file
    arr = nifti.get_fdata(dtype=np.float32) # get data as float32 numpy array
    return arr, nifti.affine, nifti.header # return array, affine, and header

# function to ensure channel first 3d array
def _ensure_channel_first_3d(arr):

    # (D,H,W) -> (1,D,H,W)
    if arr.ndim == 3:
        return arr[None, ...]
    
    # (D,H,W,1) -> (1,D,H,W)
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return np.transpose(arr, (3,0,1,2))
    
    # if bad shape, raise error
    raise ValueError(f"Unsupported array shape: {arr.shape}")

# function to do per volume normalization to [0, 1] using percentile clipping
def _percentile_norm(vol, low=1.0, high=99.0):

    # compute percentiles
    vmin, vmax = np.percentile(vol, (low, high))

    # normalize to [0, 1]
    if vmax > vmin:
        vol = np.clip((vol - vmin) / (vmax - vmin), 0.0, 1.0)

    else:
        vol = np.zeros_like(vol) # if vmax == vmin, set to zeros

    return vol.astype(np.float32) # return as float32

# function to fill masked region with random noise (to prevent model cheating)
def _fill_masked_region_with_noise(vol, mask, rng):

    # compute mean and stddev of unmasked voxels
    unmasked_voxels = vol[mask == 0.0]
    if unmasked_voxels.size == 0:
        mu, sigma = 0.0, 1.0
    else:
        mu = float(np.mean(unmasked_voxels))
        sigma = float(np.std(unmasked_voxels))

    # generate noise and fill in masked region
    noise = rng.normal(loc=mu, scale=max(1e-6, 0.3*sigma), size=vol.shape).astype(np.float32)
    vol_filled = vol * (1.0 - mask) + noise * mask

    # clamp to [0, 1] and return
    return np.clip(vol_filled, 0.0, 1.0)

# function to create single rectangular block mask of specified ratio of voxels
def _make_block_mask(shape, ratio, rng):

    # get total number of voxels and target number of masked voxels
    D, H, W = shape
    target_voxels = int(ratio * D * H * W)

    # choose block dimensions by taking cube root of target voxels and clamp to valid ranges
    side = max(1, int(round(target_voxels ** (1/3))))

    # randomize aspect within +/- 50% of side length while keeping volume close to target
    for _ in range(8):
        d = max(1, min(D, int(side * rng.uniform(0.5, 1.5))))
        h = max(1, min(H, int(side * rng.uniform(0.5, 1.5))))
        
        # solve for w to get close to target voxels
        w = max(1, min(W, int(round(target_voxels / max(1, d * h)))))

        # ensure volume not out of range
        if d * h * w <= D * H * W:
            break

    # uniformly sample block corner so it lies completely within volume
    z0 = rng.randint(0, max(1, D - d + 1))
    y0 = rng.randint(0, max(1, H - h + 1))
    x0 = rng.randint(0, max(1, W - w + 1))

    # fill in block region with 1s (masked region to be inpainted)
    mask = np.zeros((D, H, W), dtype=np.float32)
    mask[z0:z0 + d, y0:y0 + h, x0:x0 + w] = 1.0
    return mask

# function to return boolean foreground mask excluding padding and very dark background
def _foreground_mask(vol, low_p=2.0, hi_p=99.5, min_frac=0.02):

    # get vol and compute intensity thresholds
    v = vol[0] # remove channel dim
    lo, hi = np.percentile(v, (low_p, hi_p))
    thresh = max(1e-6, lo) # avoid zero threshold
    fg_mask = v > thresh
    
    # ensure minimum fraction of voxels are foreground
    if fg_mask.mean() < min_frac:
        thresh = 0.5 * thresh
        fg_mask = v > thresh

    return fg_mask

# function to sample block in foreground region
def _sample_block_in_foreground(fg_mask, ratio, rng, margin=2, max_tries=50, min_fg_frac=0.5):

    # get shape
    D, H, W = fg_mask.shape
    target_voxels = max(1, int(round(ratio * D * H * W)))
    side = max(1, int(round(target_voxels ** (1/3))))

    # initial dimensions
    d0 = max(1, min(D - 2*margin, int(side * rng.uniform(0.7, 1.3))))
    h0 = max(1, min(H - 2*margin, int(side * rng.uniform(0.7, 1.3))))
    w0 = max(1, min(W - 2*margin, int(round(target_voxels / max(1, d0 * h0)))))

    # check if block fits in foreground
    for _ in range(max_tries):
        d, h, w = d0, h0, w0

        # sample random position within valid range
        z0 = rng.randint(margin, max(margin + 1, D - d - margin + 1))
        y0 = rng.randint(margin, max(margin + 1, H - h - margin + 1))
        x0 = rng.randint(margin, max(margin + 1, W - w - margin + 1))

        # check if block stays in foreground
        sub = fg_mask[z0:z0 + d, y0:y0 + h, x0:x0 + w]
        fg_frac = float(sub.mean())
        if fg_frac >= min_fg_frac:
            m = np.zeros((D, H, W), dtype=np.float32)
            m[z0:z0 + d, y0:y0 + h, x0:x0 + w] = 1.0
            print(f"Block successfully sampled at position: {(z0, y0, x0)} with size: {(d, h, w)} and fg_frac={fg_frac:.4f} (min_fg_frac={min_fg_frac:.4f}).", flush=True)
            return m
        
        # if not, slightly reduce size and try again
        d0 = max(1, int(d0 * 0.9))
        h0 = max(1, int(h0 * 0.9))
        w0 = max(1, int(w0 * 0.9))

        print(f"Retrying block sampling with smaller size: {(d0, h0, w0)}")

    # if all tries fail, return random block mask
    print("Failed to sample block in foreground after max tries, using random block mask.")
    return _make_block_mask((D, H, W), ratio, rng)

# function to sample block in foreground region using specific size (instead of ratio)
def _sample_block_in_foreground_fixed_size(fg_mask, block_size, rng, margin=2, max_tries=50, min_fg_frac=0.5):

    # get shape of foreground mask
    D, H, W = fg_mask.shape

    # block size can be int (cubic side) or (d,h,w) tuple
    if isinstance(block_size, (tuple, list)):
        d0, h0, w0 = block_size
    else:
        d0 = h0 = w0 = int(block_size)

    # clamp to valid range (leave margin from borders)
    d0 = max(1, min(D - 2*margin, int(d0)))
    h0 = max(1, min(H - 2*margin, int(h0)))
    w0 = max(1, min(W - 2*margin, int(w0)))

    # ensure block size is valid
    if d0 <= 0 or h0 <= 0 or w0 <= 0:
        d0 = max(1, D - 2*margin)
        h0 = max(1, H - 2*margin)
        w0 = max(1, W - 2*margin)

    # try to sample block in foreground
    for _ in range(max_tries):
        d, h, w = d0, h0, w0

        # sample random position within valid range
        z0 = rng.randint(margin, max(margin + 1, D - d - margin + 1))
        y0 = rng.randint(margin, max(margin + 1, H - h - margin + 1))
        x0 = rng.randint(margin, max(margin + 1, W - w - margin + 1))

        # check if block stays in foreground
        sub = fg_mask[z0:z0 + d, y0:y0 + h, x0:x0 + w]
        fg_frac = float(sub.mean())
        if fg_frac >= min_fg_frac:
            m = np.zeros((D, H, W), dtype=np.float32)
            m[z0:z0 + d, y0:y0 + h, x0:x0 + w] = 1.0
            print(f"Block successfully sampled at position: {(z0, y0, x0)} with size: {(d, h, w)} and fg_frac={fg_frac:.4f} (min_fg_frac={min_fg_frac:.4f}).", flush=True)
            return m
        
        # if not, slightly reduce size and try again
        d0 = max(1, int(d0 * 0.9))
        h0 = max(1, int(h0 * 0.9))
        w0 = max(1, int(w0 * 0.9))
        print(f"Retrying block sampling with smaller size: {(d0, h0, w0)}")

    # if all tries fail, return centered block
    m = np.zeros((D, H, W), dtype=np.float32)
    d = min(d0, D - 2*margin)
    h = min(h0, H - 2*margin)
    w = min(w0, W - 2*margin)
    z0 = max(margin, (D - d) // 2)
    y0 = max(margin, (H - h) // 2)
    x0 = max(margin, (W - w) // 2)

    sub = fg_mask[z0:z0 + d, y0:y0 + h, x0:x0 + w]
    fg_frac = float(sub.mean())
    print(f"Using centered block at position: {(z0, y0, x0)} with size: {(d, h, w)} and fg_frac={fg_frac:.4f} (min_fg_frac={min_fg_frac:.4f}).", flush=True)
    m[z0:z0 + d, y0:y0 + h, x0:x0 + w] = 1.0
    return m

# --- Dataset Class ---

# single input item in dataset
@dataclass
class InpaintItem:
    image: Path # absolute path to .nii or .nii.gz image file
    subtype: str # folder name used for caption (ex: "amyloid_plaque_patches")

# dataset class for NIfTI inpainting
class NiftiInpaintDataset(Dataset):

    # init
    def __init__(
            self,
            items, # discovered items for a subtype
            captions_json=None, # optional path to captions json file that overrides subtype names
            default_caption_by_subtype=None, # optional dict of default captions by subtype
            mask_mode='ratio', # 'ratio' or 'fixed_size' mode for mask generation
            mask_ratio=0.3, # ratio of voxels to mask for inpainting
            mask_fixed_size=32, # fixed block size (int or (d,h,w) tuple) for 'fixed_size' mode
            num_mask_blocks=1, # number of mask blocks to create per volume
            augment=True, # whether to jitter mask_ratio at train time
            seed=100 # random seed for mask placement
    ):
        
        # store basic config
        self.items = items
        self.augment = augment
        self.mask_mode = str(mask_mode.lower())
        self.mask_ratio = float(mask_ratio)

        # allow scalar or (d,h,w) tuple for fixed size (normalize to 3 tuple of ints)
        if isinstance(mask_fixed_size, (tuple, list)):
            self.mask_fixed_size = tuple(int(s) for s in mask_fixed_size)

        # scalar
        else:
            self.mask_fixed_size = int(mask_fixed_size)
            
        self.num_mask_blocks = max(1, int(num_mask_blocks)) # at least 1 block
        self.base_seed = int(seed)

        # caption sources (1. captions_json, 2. default_caption_by_subtype, 3. subtype name)
        self.captions_map = {}
        if captions_json and Path(captions_json).is_file(): # load captions from json file
            with open(captions_json, 'r') as f:
                raw = json.load(f)

                # normalize keys to basenames so either full path or filename can be used
                for k, v in raw.items():
                    self.captions_map[Path(k).name] = str(v)
        self.default_caption_by_subtype = default_caption_by_subtype or {} # default captions dict

    # length
    def __len__(self):
        return len(self.items)
    
    # caption normalization
    @staticmethod
    def _normalize_prompt_from_subtype(subtype):

        # get cleaned subtype name
        base = subtype
        if base.endswith('_patches'):
            base = base[:-8]

        # pretty format
        if base == 'amyloid_plaque':
            return 'amyloid plaque'
        if base == 'c_fos_positive':
            return 'c-Fos positive'
        if base == 'cell_nucleus':
            return 'cell nucleus'
        if base == 'vessels':
            return 'vessels'
        return base.replace('_', ' ').strip() # generic replacement
    
    # resolve caption for item
    def _resolve_caption(self, path, subtype):

        # 1. per file json override, 2. explicit default map, 3. folder-derived prompt
        return (
            self.captions_map.get(path.name) or
            self.default_caption_by_subtype.get(subtype) or
            NiftiInpaintDataset._normalize_prompt_from_subtype(subtype)
        )
    
    # get item
    def __getitem__(self, idx):

        # get item
        item = self.items[idx]

        # deterministic per-item rng
        uid = f'{item.image.resolve()}'
        item_seed = _stable_int_hash(uid) + self.base_seed
        rng = np.random.RandomState(item_seed)

        # load image and convert to channel first 3d array
        img_arr, affine, header = _load_nifti(item.image)
        vol = _ensure_channel_first_3d(img_arr) # (1,D,H,W)
        vol = _percentile_norm(vol) # per volume normalization to [0, 1]

        # build rectangular hole(s)
        D, H, W = vol.shape[1:] # get spatial shape
        fg_mask = _foreground_mask(vol) # get foreground mask (D,H,W), boolean

        # create mask
        mask3d = np.zeros((D, H, W), dtype=np.float32)

        # sample 1 or more blocks (depending on num_mask_blocks)
        for _ in range(self.num_mask_blocks):

            # ratio mode
            if self.mask_mode == 'ratio':
                ratio = self.mask_ratio
                if self.augment:
                    ratio = float(
                        np.clip(
                            rng.normal(loc=self.mask_ratio, scale=0.10),
                            0.10,
                            0.60
                        )
                    )
                
                # sample block in foreground
                block_mask = _sample_block_in_foreground(fg_mask, ratio, rng, margin=2) # (D,H,W)

            # fixed size mode
            elif self.mask_mode == 'fixed_size':
                
                # if tuple is given, use that as (d,h,w)
                if isinstance(self.mask_fixed_size, (tuple, list)):
                    d0, h0, w0 = [int(s) for s in self.mask_fixed_size]
                    if self.augment:
                        scale = float(
                            np.clip(
                                rng.normal(loc=1.0, scale=0.2), 0.5, 1.5
                            )
                        )
                        d0 = max(1, int(d0 * scale))
                        h0 = max(1, int(h0 * scale))
                        w0 = max(1, int(w0 * scale))
                    block_size = (d0, h0, w0)

                # else use cubic size (int)
                else:
                    side = int(self.mask_fixed_size)
                    if self.augment:
                        side = int(
                            np.clip(
                                rng.normal(loc=side, scale=0.2*side), 4,  min(D, H, W)
                            )
                        )
                    block_size = side

                # sample block in foreground
                block_mask = _sample_block_in_foreground_fixed_size(fg_mask, block_size, rng, margin=2) # (D,H,W)


            # raise error if unknown mode
            else:
                raise ValueError(f"Unknown mask_mode: {self.mask_mode}")
            
            # combine block mask into overall mask
            mask3d = np.maximum(mask3d, block_mask)

        # final mask
        mask = mask3d[None, ...] # (1,D,H,W)

        # fill masked region with noise instead of zeros
        masked_vol = _fill_masked_region_with_noise(vol, mask, rng)

        # return everything needed by model
        return {
            'masked_vol': torch.from_numpy(masked_vol.copy()), # masked input volume
            'mask': torch.from_numpy(mask.copy()), # input mask
            'target_vol': torch.from_numpy(vol.copy()), # target volume to inpaint
            'filename': item.image.name, # filename of input volume for saving outputs
            'text': self._resolve_caption(item.image, item.subtype), # text prompt caption for input volume
            'affine': torch.from_numpy(affine.astype(np.float32)) # affine matrix for saving output nifti
        }


# --- Discovery Function ---

# function to discover NIfTI inpainting dataset items from root data folder
def discover_nifti_inpaint_items(class_dir, channel_substr='ALL'):

    # normalize channel filter tokens
    subtypes = None
    s = str(channel_substr).strip()
    if s and s.upper() != 'ALL':
        subtypes = [sub.strip().lower() for sub in s.split(',') if sub.strip()]

    # discover items
    items = []
    subtype = class_dir.name

    # walk directory for .nii or .nii.gz files
    for path in sorted(class_dir.glob('*.nii*')):
        low = path.name.lower()

        # only accept .nii or .nii.gz files
        if not (low.endswith('.nii') or low.endswith('.nii.gz')):
            continue

        # skip segmentation labels
        if low.endswith('_label.nii') or low.endswith('_label.nii.gz'):
            continue

        # apply channel filter if specified
        if subtypes is not None and not any(tok in low for tok in subtypes):
            continue

        # add item
        items.append(InpaintItem(image=path.resolve(), subtype=subtype))

    return items












