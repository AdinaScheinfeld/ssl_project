# Script to split SELMA 3D data into train/val sets and augment training data

# --- Setup ---

# imports
import numpy as np
import os
import random
import sys
import tifffile as tiff

from monai.data import Dataset, DataLoader

from monai.transforms import (
    Compose,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord
)

# get functions from other files
sys.path.append('/home/ads4015/ssl_project/preprocess_patches/src')
from selma3d_visualization_functions import LoadTiffd, get_train_transforms, get_val_transforms


# set seed for reproducibility
random.seed(100)


# --- Functions ---

# function to apply transforms and save output
def save_transformed_patches(samples, transforms, output_dir):

    print('Saving transformed patches...', flush=True)

    # create directory for output
    os.makedirs(output_dir, exist_ok=True)

    # transform patch and save
    for i, patch in enumerate(samples):
        output_path = os.path.join(output_dir, f'{patch["label"]}_{i:05d}.tif')
        transformed_patch = transforms(patch)
        img_np = transformed_patch['image'].squeeze().cpu().numpy() # remove channel
        tiff.imwrite(output_path, img_np.astype(np.float32)) # save as tiff


# --- Get Data ---

print('Getting data...', flush=True)

# define paths per class and create directories
data_root = '/midtier/paetzollab/scratch/ads4015/data_selma3d/small_patches'
output_root = '/midtier/paetzollab/scratch/ads4015/data_selma3d/lsm_fm'
train_dir = os.path.join(output_root, 'train')
val_dir = os.path.join(output_root, 'val')
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# define list of datatypes and weights
data_types = ['ab_plaque', 'cfos', 'nucleus', 'vessel_eb', 'vessel_wga']
data_weights = {'ab_plaque': 5, 'cfos': 1, 'nucleus': 5, 'vessel_eb': 2, 'vessel_wga': 2} # oversample datatypes with fewer images

# collect files per class
train_samples, val_samples = [], []
train_prefixes, val_prefixes = {}, {}

# loop through datatypes
for dtype in data_types:

    # get all files from dtype
    folder = os.path.join(data_root, dtype)
    print('folder', folder, flush=True)
    all_files = sorted([os.path.join(folder, file) for file in os.listdir(folder) if file.endswith('.tiff')])
    print('all_files', all_files, flush=True)

    # group patches by image (using prefix)
    grouped = {}
    for file in all_files:
        prefix = '_'.join(os.path.basename(file).split('_')[:-1])
        grouped.setdefault(prefix, []).append(file)

    # turn groups into list and shuffle
    groups = list(grouped.values())
    random.shuffle(groups)

    # split image groups into train and val
    split_idx = max(1, int(0.8 * len(groups)))
    train_groups = groups[:split_idx]
    val_groups = groups[split_idx:]

    # store image image prefixes for logging
    train_prefixes[dtype] = [os.path.basename(g[0]).rsplit('_', 1)[0] for g in train_groups]
    val_prefixes[dtype] = [os.path.basename(g[0]).rsplit('_', 1)[0] for g in val_groups]

    # sampling: take all or subset of patches based on class size
    for group in train_groups:
        files = group 
        if data_weights[dtype] > 1:
            train_samples += [{'image': file, 'label': dtype} for file in files for i in range(data_weights[dtype])]
        else:
            subset = random.sample(files, max(1, len(files) // 2))
            train_samples += [{'image': file, 'label': dtype} for file in subset]

    for group in val_groups:
        val_samples += [{'image': file, 'label': dtype} for file in group]


# --- Transforms ---

# get train and val transforms
train_transforms = get_train_transforms()
val_transforms = get_val_transforms()


# --- Save ---

# process and save training and validation sets
save_transformed_patches(train_samples, train_transforms, train_dir)
save_transformed_patches(val_samples, val_transforms, val_dir)

# logging
print(f'Num train samples: {len(train_samples)}; Num val samples: {len(val_samples)}', flush=True)

print('\nImages used for training:', flush=True)
for k, v in train_prefixes.items():
    print(f'  {k}: {v}', flush=True)

print('\nImage prefixes used for validation:', flush=True)
for k, v in val_prefixes.items():
    print(f'  {k}: {v}', flush=True)



    










