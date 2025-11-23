# /home/ads4015/ssl_project/data/nifti_deblur_dataset.py - Dataset class for loading blurred and sharp nifti image pairs for deblurring tasks

# --- Setup ---

# import
from dataclasses import dataclass
import nibabel as nib
import numpy as np
from pathlib import Path
import torch
from torch.utils.data import Dataset


# --- Helper Functions ---

# function to load nifti image
def _load_nifti(path):
    nifti = nib.load(str(path))
    arr = nifti.get_fdata(dtype=np.float32)
    return arr, nifti.affine, nifti.header

# function to ensure volume has shape (C, D, H, W)
def _ensure_channel_first_3d(arr):

    # add channel dimension if missing
    if arr.ndim == 3:
        return arr[None, ...]
    
    # if channel is last dimension, move to first
    if arr.ndim == 4 and arr.shape[-1] == 1:
        return np.transpose(arr, (3, 0, 1, 2))
    
    # error if shape is unexpected
    raise ValueError(f'Unsupported array shape for 3D image: {arr.shape}')

# function to normalize image to [0, 1] using percentile clipping based on sharp image
def _normalize_image(sharp_img, blurred_img, p_min=1.0, p_max=99.0):
    
    # get percentiles from blurred image
    # will use same percentiles for sharp image
    # getting percentiles from blurred image helps avoid issues with downstream inference when no sharp image is available
    v = blurred_img.reshape(-1)
    vmin, vmax = np.percentile(v, [p_min, p_max])

    # clip and normalize both images
    if vmax > vmin:
        blurred_norm = np.clip((blurred_img - vmin) / (vmax - vmin), 0.0, 1.0)
        sharp_norm = np.clip((sharp_img - vmin) / (vmax - vmin), 0.0, 1.0)
    
    # if vmax == vmin, return zeros
    else:
        blurred_norm = np.zeros_like(blurred_img, dtype=np.float32)
        sharp_norm = np.zeros_like(sharp_img, dtype=np.float32)

    return sharp_norm.astype(np.float32), blurred_norm.astype(np.float32)

# function to discover DeblurItem objects for a given subtype
def discover_nifti_deblur_items(sharp_class_dir, blurred_root, channel_substr='ALL'):

    # get subtype
    subtype = sharp_class_dir.name

    # normalize channel filter tokens
    filters = None
    s = str(channel_substr).strip()
    if s and s.upper() != 'ALL':
        filters = [tok.strip().lower() for tok in s.split(',') if tok.strip()]

    # list to hold discovered items
    items = []

    # iterate over sharp images in subtype directory
    for sharp_path in sorted(sharp_class_dir.glob('*.nii*')):

        # make path lowercase
        sharp_name_lower = sharp_path.name.lower()

        # only accept .nii or .nii.gz files
        if not (sharp_name_lower.endswith('.nii') or sharp_name_lower.endswith('.nii.gz')):
            continue

        # skip files that contain 'label' in filename
        if 'label' in sharp_name_lower:
            continue

        # if filters are provided, check if any filter token is in filename
        if filters is not None and not any(tok in sharp_name_lower for tok in filters):
            continue

        # derive corresponding blurred image path
        rel_path = sharp_path.relative_to(sharp_class_dir.parent)
        blurred_path = blurred_root / rel_path

        # check if blurred image exists
        if not blurred_path.is_file():
            print(f'[WARNING] Missing blurred image for sharp image: {sharp_path}. Skipping.', flush=True)
            continue

        # create DeblurItem and add to list
        item = DeblurItem(
            sharp_image=sharp_path.resolve(),
            blurred_image=blurred_path.resolve(),
            subtype=subtype
        )
        items.append(item)

    return items


# --- Dataclass ---

# dataclass for one sharp/blurred image pair for a given subtype
@dataclass
class DeblurItem:
    sharp_image: Path
    blurred_image: Path
    subtype: str


# --- Dataset Class ---

# dataset class for loading blurred and sharp nifti image pairs for deblurring tasks
class NiftiDeblurDataset(Dataset):

    # init
    def __init__(self, items):

        # list of DeblurItem objects discovered for a subtype
        self.items = items

    # len
    def __len__(self):
        return len(self.items)
    
    # getitem
    def __getitem__(self, idx):

        # get paired item
        item = self.items[idx]

        # load sharp and blurred images
        sharp_arr, sharp_affine, sharp_header = _load_nifti(item.sharp_image)
        blurred_arr, blurred_affine, blurred_header = _load_nifti(item.blurred_image)

        # ensure channel first shape
        sharp_vol = _ensure_channel_first_3d(sharp_arr)
        blurred_vol = _ensure_channel_first_3d(blurred_arr)

        # normalize images to [0, 1] using percentiles from blurred image
        sharp_vol, blurred_vol = _normalize_image(sharp_vol, blurred_vol)

        # convert to torch tensors
        sharp_tensor = torch.from_numpy(sharp_vol.copy()) # (1, D, H, W)
        blurred_tensor = torch.from_numpy(blurred_vol.copy()) # (1, D, H, W)

        # return dict with tensors and metadata
        return {
            'input_vol': blurred_tensor, # blurred input tensor
            'target_vol': sharp_tensor,   # sharp ground truth tensor
            'filename': item.sharp_image.name, # filename of sharp image
            'affine': torch.from_numpy(sharp_affine.astype(np.float32)), # affine of sharp image
        }














