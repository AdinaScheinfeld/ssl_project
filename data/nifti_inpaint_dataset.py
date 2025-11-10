# nifti_inpaint_dataset.py - A dataset class for handling NIfTI images for inpainting tasks.

# --- Setup ---

# imports
from dataclasses import dataclass
import json
import nibabel as nib
import numpy as np
from pathlib import Path

import torch
from torch.utils.data import Dataset


# --- Helper Functions ---

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
            mask_ratio=0.3, # ratio of voxels to mask for inpainting
            augment=True, # whether to jitter mask_ratio at train time
            seed=100 # random seed for mask placement
    ):
        
        # store basic config
        self.items = items
        self.augment = augment
        self.mask_ratio = float(mask_ratio)
        self.rng = np.random.RandomState(seed)

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

        # load image and convert to channel first 3d array
        img_arr, affine, header = _load_nifti(item.image)
        vol = _ensure_channel_first_3d(img_arr) # (1,D,H,W)
        vol = _percentile_norm(vol) # per volume normalization to [0, 1]

        # build rectangular hole
        D, H, W = vol.shape[1:] # get spatial shape
        ratio = self.mask_ratio

        # jitter ratio slightly if augmenting
        if self.augment:
            ratio = float(np.clip(self.rng.normal(loc=self.mask_ratio, scale=0.05), 0.05, 0.7))
        
        # create block mask
        mask3d = _make_block_mask((D, H, W), ratio, self.rng) # (D,H,W)
        mask = mask3d[None, ...] # (1,D,H,W)

        # zero out masked region in input volume
        masked_vol = vol * (1.0 - mask) # (1,D,H,W)

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












