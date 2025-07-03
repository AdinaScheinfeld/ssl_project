# Script to extract labeled patches from selma3d for foundation model finetuning


# --- Setup ---

# imports
from collections import defaultdict
import glob as glob
import nibabel as nib
import numpy as np
import os
import random
import torch
from tqdm import tqdm


# --- Config ---

# config

PATCH_SIZE = (96, 96, 96)
PATCHES_PER_CLASS = 25

OUTPUT_DIR = '/midtier/paetzollab/scratch/ads4015/data_selma3d/lsm_fm_selma3d_finetune'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# image and label sources per class
classes = {
    'cfos': {
        'img_dir': '/midtier/paetzollab/scratch/ads4015/data_selma3d/SELMA3D/brain_c_fos_positive_patches/cFos-Active_Neurons/raw',
        'label_dir': '/midtier/paetzollab/scratch/ads4015/data_selma3d/SELMA3D/annotation_brain_c_fos_positive/cFos-Active_Neurons/gt'
    },
    'vessel': {
        'img_dir': '/midtier/paetzollab/scratch/ads4015/data_selma3d/SELMA3D/brain_vessels_patches/VessAP_vessel/raw',
        'label_dir': '/midtier/paetzollab/scratch/ads4015/data_selma3d/SELMA3D/annotation_brain_vessels/VessAP_vessel/gt'
    },
    'nucleus': {
        'img_dir': '/midtier/paetzollab/scratch/ads4015/data_selma3d/SELMA3D/brain_cell_nucleus_patches/shannel_cells/raw',
        'label_dir': '/midtier/paetzollab/scratch/ads4015/data_selma3d/SELMA3D/annotation_brain_cell_nucleus/shannel_cells/gt'
    },
    'plaque': {
        'img_dir': '/midtier/paetzollab/scratch/ads4015/data_selma3d/SELMA3D/brain_amyloid_plaque_patches/AD_plaques/raw',
        'label_dir': '/midtier/paetzollab/scratch/ads4015/data_selma3d/SELMA3D/annotation_brain_amyloid_plaque/AD_plaques/gt'
    },
}


# --- Functions ---

# function to pad image to minimum size required
def pad_to_min_size(volume, target_shape):

    # ensure correct dimensions
    if volume.ndim == 4 and volume.shape[-1] == 1:
        volume = np.squeeze(volume, axis=-1)

    # create list to hold padding width
    pad_width = []

    # loop through volumes
    for dim, target in zip(volume.shape, target_shape):

        # get total padding
        total_pad = max(target - dim, 0)

        # get padding for before and after
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before

        # add padding amounts to list
        pad_width.append((pad_before, pad_after))

    # return padded volume
    return np.pad(volume, pad_width, mode='constant')


# function to load image-label pairs
def load_image_label_pair(img_path, label_path, is_vessel=False):

    # load image
    img_nii = nib.load(img_path)
    img_data = img_nii.get_fdata()
    img_affine = img_nii.affine

    # get both channels from vessel data
    if is_vessel:
        ch0_path = img_path.replace('_0001.nii.gz', '_0000.nii.gz')
        ch0_nii = nib.load(ch0_path)
        ch0_data = ch0_nii.get_fdata()

        # correct dimensions
        ch0_data = np.squeeze(ch0_data)
        img_data = np.squeeze(img_data)

        img_data = np.stack([ch0_data, img_data], axis=0) # shape (2, D, H, W)
        img_affine = ch0_nii.affine

    # all other datatypes have just 1 channel
    else:
        img_data = np.expand_dims(img_data, axis=0) # shape (1, D, H, W)

    # get label (all images have 1 label, vessel channels are labeled together)
    label_nii = nib.load(label_path)
    label_data = label_nii.get_fdata()
    label_affine = label_nii.affine

    # pad image and label if too small
    C, D, H, W = img_data.shape
    d, h, w = PATCH_SIZE
    if D < d or H < h or W < w:
        img_data = pad_to_min_size(img_data, (img_data.shape[0], max(D, d), max(H, h), max(W, w)))
        label_data = pad_to_min_size(label_data, PATCH_SIZE)

    # return image-label pair
    return img_data.astype(np.float32), label_data.astype(np.uint16), img_affine, label_affine


# function to extract patches from image-label pair
def extract_valid_patch(img, label):

    # get image and patch dimensions
    C, D, H, W = img.shape
    d, h, w = PATCH_SIZE

    # create foreground mask using label (only foreground regions will have a label)
    label_mask = (label > 0).astype(np.uint16)
    
    # create list to hold valid coordinates
    valid_coords = []

    # add coords with foreground to list
    for z in range(0, D - d + 1):
        for y in range(0, H - h + 1):
            for x in range(0, W - w + 1):
                sub = label_mask[z:z+d, y:y+h, x:x+w]
                # if sub.sum() / sub.size > FOREGROUND_THRESHOLD:
                if sub.any():
                    valid_coords.append((z, y, x))
    
    # indicate if no valid patches were found
    if not valid_coords:
        print('[WARNING] No valid patches found', flush=True)
        return None, None, None
    
    # return random selection of valid coords
    # print(f'Valid coords: {valid_coords}', flush=True)
    z, y, x = random.choice(valid_coords)
    return img[:, z:z+d, y:y+h, x:x+w], label[z:z+d, y:y+h, x:x+w], (z, y, x)


# function to extract patches for a class
def extract_patches_for_class(class_name, config):

    # get image and label paths
    print(f'[INFO] Extracting patches for class: {class_name}', flush=True)
    img_paths = sorted(glob.glob(os.path.join(config['img_dir'], '*.nii.gz')))
    label_paths = sorted(glob.glob(os.path.join(config['label_dir'], '*.nii.gz')))

    # create output directory
    output_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_dir, exist_ok=True)

    # determine patches per image
    n_images = len(img_paths)
    base_num_patches = PATCHES_PER_CLASS // n_images
    remainder_num_patches = PATCHES_PER_CLASS % n_images
    patch_counts = [base_num_patches] * n_images

    # randomly assign remainder
    for idx in random.sample(range(n_images), remainder_num_patches):
        patch_counts[idx] += 1

    # define counter
    count = 0

    # load image-label pairs
    for i, (img_path, label_path) in enumerate(tqdm(zip(img_paths, label_paths), total=n_images)):

        # determine how many patches are needed
        num_patches_needed = patch_counts[i]
        num_extracted = 0

        # get image label and data
        img, label, affine_img, affine_lbl = load_image_label_pair(img_path, label_path, is_vessel=(class_name == 'vessel'))

        # try extracting valid patches from volume
        num_attempts = 0
        while num_extracted < num_patches_needed and num_attempts < 100:
            img_patch, label_patch, coords = extract_valid_patch(img, label)
            num_attempts += 1

            # save valid patches as .pt
            if img_patch is not None:
                torch.save({
                    'image': torch.tensor(img_patch),
                    'label': torch.tensor(label_patch),
                    'class': class_name,
                    'channels': img_patch.shape[0]
                }, os.path.join(OUTPUT_DIR, class_name, f'patch_{count:03d}.pt'))

                # save as nifti
                for ch in range(img_patch.shape[0]):
                    nib.save(nib.Nifti1Image(img_patch[ch], affine=affine_img), 
                             os.path.join(output_class_dir, f'patch{count:03d}_ch{ch}_img.nii.gz'))
                    
                # save label as nifti
                nib.save(nib.Nifti1Image(label_patch, affine=affine_lbl), 
                         os.path.join(output_class_dir, f'patch_{count:03d}_label.nii.gz'))

                # increment counter and stop when quota has been reached
                print(f'[INFO] Patch {count:03d} extracted from {os.path.basename(img_path)} at (z={coords[0]}, y={coords[1]}, x={coords[2]})', flush=True)
                count += 1
                num_extracted += 1
                if count >= PATCHES_PER_CLASS:
                    print(f'[INFO] Finished extracting {count} patches for class {class_name}', flush=True)
                    return
                
    print(f'[INFO] Extracted a total of {count} patches for class: {class_name}', flush=True)
           

# --- Main ---

# main function
def main():
    for cls, cfg in classes.items():
        extract_patches_for_class(cls, cfg)


if __name__ == '__main__':
    main()





