# Python script to extract patches from SELMA3D dataset to finetune foundation model

# --- Setup ---

# imports
import argparse
import nibabel as nib
import numpy as np
import os
from pathlib import Path
import sys
import torch
from tqdm import tqdm
import yaml


# --- Config ---

# map brain structure class to corresponding image and label paths
DATA_CLASSES = {
    'brain_amyloid_plaque_patches': {
        'image': 'brain_amyloid_plaque_patches/AD_plaques/raw',
        'label': 'annotation_brain_amyloid_plaque/AD_plaques/gt'
    },
    'brain_c_fos_positive_patches': {
        'image': 'brain_c_fos_positive_patches/cFos-Active_Neurons/raw',
        'label': 'annotation_brain_c_fos_positive/cFos-Active_Neurons/gt'
    },
    'brain_cell_nucleus_patches': {
        'image': 'brain_cell_nucleus_patches/shannel_cells/raw',
        'label': 'annotation_brain_cell_nucleus/shannel_cells/gt'
    },
    'brain_vessels_patches': {
        'image': 'brain_vessels_patches/VessAP_vessel/raw',
        'label': 'annotation_brain_vessels/VessAP_vessel/gt'
    }
}

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default='config.yaml', help='Path to the config file')
parser.add_argument('--class_name', type=str, choices=list(DATA_CLASSES.keys()), required=True, help='Name of the class to process')
args = parser.parse_args()

# load config from yaml
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# extract config values from loaded yaml config file
INPUT_ROOT = Path(config['input_root'])
OUTPUT_ROOT = Path(config['output_root'])
PATCH_SIZE = tuple(config['patch_size'])
PATCHES_PER_CLASS = config['patches_per_class']


# --- Helper Functions ---

# function to load nifti file from given path
def load_nifti(path):
    return nib.load(str(path))


# function to extract volume of specified size from specified 3d position
def extract_patch(volume, start, size):
    z, y, x = start # unpack start position
    dz, dy, dx = size # unpack patch size
    return volume[z:z+dz, y:y+dy, x:x+dx] # return patch extracted from volume


# function to save patches as .pt and .nii.gz files
def save_patch_with_label(patch, label_patch, save_dir, patch_id, vol_id, channel, affine):

    # create save directory if it doesn't exist
    save_dir.mkdir(parents=True, exist_ok=True)

    # define filename
    filename_pt = f'patch_{patch_id:03d}_vol{vol_id:03d}_ch{channel}.pt'
    filename_nii = f'patch_{patch_id:03d}_vol{vol_id:03d}_ch{channel}.nii.gz'
    filename_lbl_nii = f'patch_{patch_id:03d}_vol{vol_id:03d}_ch{channel}_label.nii.gz'

    # save patch as .pt file
    torch.save({
        'image': torch.tensor(patch).float(),
        'label': torch.tensor(label_patch).long()
    }, save_dir / filename_pt)

    # save patch as nifti
    nib.save(nib.Nifti1Image(patch.astype(np.float32), affine), save_dir / filename_nii)
    nib.save(nib.Nifti1Image(label_patch.astype(np.uint8), affine), save_dir / filename_lbl_nii)

    # return .pt and .nii.gz filenames
    return filename_pt, filename_nii, filename_lbl_nii


# function to pad volume to minimum size
def pad_to_min_size(volume, target_size):

    # create list to hold padding widths
    pad_widths = []

    # calculate padding widths for each spatial dimension
    for dim, target in zip(volume.shape[-3:], target_size):
        total_pad = max(0, target - dim) # calculate total padding needed
        pad_before = total_pad // 2 # calculate padding before
        pad_after = total_pad - pad_before # calculate padding after
        pad_widths.append((pad_before, pad_after)) # append padding widths to list
    pad_widths = [(0, 0)] * (volume.ndim - 3) + pad_widths # add no padding to non-spatial dimensions (but ensure correct shape for adding)

    # return padded volume
    return np.pad(volume, pad_widths, mode='constant', constant_values=0)


# function to get start positions centered on foreground voxels
## FIX - function only returns coords that keep patch fully inside volume (can edit this to return padded patches that go out of bounds if they meet a foreground threshold)
def get_centered_coords(fg_mask, shape, patch_size):

    # get patch size
    dz, dy, dx = patch_size

    # get foreground coordinates
    valid_coords = np.argwhere(fg_mask)

    # shuffle valid coords
    np.random.shuffle(valid_coords)

    # filter valid coords to only those that keep patch fully inside volume
    for zc, yc, xc in valid_coords:
        z = zc - dz // 2
        y = yc - dy // 2
        x = xc - dx // 2
        if z >= 0 and y >= 0 and x >= 0 and z + dz <= shape[0] and y + dy <= shape[1] and x + dx <= shape[2]:
            yield (z, y, x) # return iterator


# --- Main function ---

# function to extract patches from a class
def process_class(class_folder, paths):

    # define counter to keep track of number of patches extracted
    patch_id_counter = 0

    # get input, label, and output directories
    input_dir = INPUT_ROOT / paths['image']
    label_dir = INPUT_ROOT / paths['label']
    output_dir = OUTPUT_ROOT / class_folder.split('_', 1)[-1]

    # get image files
    image_files = sorted(input_dir.glob('patchvolume_*_0000.nii.gz'))
    print(f'{class_folder}: Found {len(image_files)} image files in {input_dir}', flush=True)

    # indicate if no image files found
    if len(image_files) == 0:
        print(f'No image files found for {class_folder}, skipping.', flush=True)

    # shuffle image files and determine how many patches to extract from each volume (redistribute the remainder)
    np.random.shuffle(image_files)
    num_volumes = len(image_files)
    num_patches_per_volume = [PATCHES_PER_CLASS // num_volumes] * num_volumes
    for idx in np.random.choice(num_volumes, PATCHES_PER_CLASS % num_volumes, replace=False):
        num_patches_per_volume[idx] += 1

    # initialize counter for patches per volume and make list of unprocessed volumes
    volume_patch_counts = [0] * num_volumes
    pending_volumes = list(range(num_volumes))

    # extract patches from each volume
    while sum(volume_patch_counts) < PATCHES_PER_CLASS and pending_volumes:
        
        # loop over pending volumes
        for i in pending_volumes.copy():

            # remove pending volumes if sufficient patches extracted
            if volume_patch_counts[i] >= num_patches_per_volume[i]:
                pending_volumes.remove(i)
                continue

            # load image and label volumes
            image_file = image_files[i]
            vol_id = int(image_file.stem.split('_')[1])
            label_file = label_dir / f'patchvolume_{vol_id:03d}.nii.gz'

            # skip if label file does not exist
            if not label_file.exists():
                print(f'   Skipping {vol_id:03d}: missing label file', flush=True)
                pending_volumes.remove(i)
                continue

            # get label and affine
            label_nii = load_nifti(label_file)
            label = label_nii.get_fdata()
            affine = label_nii.affine
            label = pad_to_min_size(label, PATCH_SIZE) # pad label to minimum size
            fg_mask = label != 0 # create foreground mask

            # skip if no foreground voxels
            if not fg_mask.any():
                print(f'   Skipping volume {vol_id:03d}: label has no foreground voxels', flush=True)
                pending_volumes.remove(i)
                continue

            # get volumes
            try:
                
                # pad if necessary
                image_channels = [pad_to_min_size(load_nifti(image_file).get_fdata(), PATCH_SIZE)]

                # process vessel images (2 channels)
                if 'vessels' in class_folder:
                    ch1_file = image_file.with_name(image_file.name.replace('0000', '0001'))
                    if ch1_file.exists():
                        image_channels.append(pad_to_min_size(load_nifti(ch1_file).get_fdata(), PATCH_SIZE))

                # stack image channels
                image_channels = np.stack(image_channels, axis=0)
                if image_channels.ndim != 4:
                    raise ValueError(f'Image {image_file} has more than 3 spatial dimensions after padding.')
                
                # get shape of image channels
                C, Z, Y, X = image_channels.shape

            except Exception as e:
                print(f'   Skipping volume {vol_id:03d}: error during image loading {e}', flush=True)
                pending_volumes.remove(i)
                continue

            # extract patches
            for z, y, x in get_centered_coords(fg_mask, (Z, Y, X), PATCH_SIZE):
                
                # extract label patch
                label_patch = extract_patch(label, (z, y, x), PATCH_SIZE)

                # extract image patch
                if np.any(label_patch):
                    for ch in range(C):
                        img_patch = extract_patch(image_channels[ch], (z, y, x), PATCH_SIZE)

                        # save patch with labels
                        save_patch_with_label(img_patch, label_patch, output_dir, patch_id_counter, vol_id, ch, affine)
                    patch_id_counter += 1 
                    volume_patch_counts[i] += 1
                    if volume_patch_counts[i] >= num_patches_per_volume[i]:
                        break

    # print summary of patches extracted
    print(f'\nSummary for {class_folder}:', flush=True)
    for idx, count in enumerate(volume_patch_counts):
        vol_id = int(image_files[idx].stem.split('_')[1])
        if count == 0:
            print(f'   Volume {vol_id:03d}: No patches extracted', flush=True)
        else:
            print(f'   Volume {vol_id:03d}: {count} patches extracted', flush=True)
    print(f'\nTotal patches saved for {class_folder}: {patch_id_counter}', flush=True)


# --- Main entry ---

if __name__ == '__main__':

    # get class folder and paths
    class_name = args.class_name
    class_paths = DATA_CLASSES[class_name]

    # process class
    process_class(class_name, class_paths)

    ## UP TO HERE - make config and slurm script





























