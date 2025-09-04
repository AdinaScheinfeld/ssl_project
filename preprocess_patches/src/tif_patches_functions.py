# tif_patches_functions.py - functions used in tif_stack_get_patches.py and tif_stack_get_patches_multichannel.py


# --- Setup ---

# imports
import argparse
import glob
import hashlib
import nibabel as nib
import numpy as np
import os
import random
import re
import sys
import tifffile as tiff

from skimage.filters import threshold_otsu


# --- Functions ---


# compute global otsu threshold on downsampled image (to increase speed)
def compute_global_otsu_threshold(vol, downsample_factor=4):

    # downsample and threshold
    vol_ds = vol[::downsample_factor, ::downsample_factor, ::downsample_factor]
    vmin, vmax = float(vol_ds.min()), float(vol_ds.max())
    if vmin == vmax:
        return vmin
    return float(threshold_otsu(vol_ds))


# function to sort slices in tiff stack
def make_sort_key(regex_pattern):

    # regex matches
    regex = re.compile(regex_pattern)

    # sort key
    def sort_key(path):
        name = os.path.basename(path)
        m = regex.search(name)
        if not m:
            raise ValueError(f'Filename "{name}" does not match regex "{regex_pattern}"', flush=True)
        return int(m.group(1))

    return sort_key


# function to set seed
def deterministic_seed(folder_path, user_seed=None):

    # hash seed
    base = int(hashlib.md5(folder_path.encode('utf-8')).hexdigest()[:8], 16)
    seed = base if user_seed is None else ((base ^ user_seed) & 0xFFFFFFFF)

    # set seed
    random.seed(seed)
    np.random.seed(seed)
    return seed


# get candidates (patches fully inside volume)
def get_grid_candidates(vol_shape, patch_size, stride=None):

    Z, Y, X = vol_shape
    ps = int(patch_size)
    st = ps if stride is None else int(stride) # stride defaults to patch size for no overlap

    # return (z,y,x) starts for patches fully inside volume
    for z in range(0, Z - ps + 1, st):
        for y in range(0, Y - ps + 1, st):
            for x in range(0, X - ps + 1, st):
                yield (z, y, x)















