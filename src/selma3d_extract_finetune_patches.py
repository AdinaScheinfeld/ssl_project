# Script to extract labeled patches from selma3d for foundation model finetuning

# --- Setup ---

# imports
import glob
import nibabel as nib
import numpy as np
import os
import random
import torch
from tqdm import tqdm


# --- Config ---
PATCH_SIZE = (96, 96, 96)
PATCHES_PER_CLASS = 25
FOREGROUND_THRESHOLD = 0.01 # proportion of nonzero voxels in label

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

    # create list to hold paddding width
    pad_width = []

    for dim, target in zip(volume.shape, target_shape):

        # get total padding
        total_pad = max(target - dim, 0)
        print(f'Total padding: {total_pad}', flush=True)

        # get padding for before and after
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before

        # add padding amounts to list
        pad_width.append((pad_before, pad_after))

    return np.pad(volume, pad_width, mode='constant')


# function to load image-label pairs
def load_image_label_pair(img_path, label_path, is_vessel=False):

    # load image
    img_nii = nib.load(img_path)
    img = img_nii.get_fdata()
    affine = img_nii.affine

    # get both channels from vessel data
    if is_vessel:
        ch0_path = img_path.replace('_0001.nii.gz', '_0000.nii.gz')
        ch0_nii = nib.load(ch0_path)
        ch0 = ch0_nii.get_fdata()

        # correct dimensions
        # if ch0.ndim == 5 and ch0.shape[-1] == 1:
        # print(f'Ch0 dim: {ch0.shape}')
        ch0 = np.squeeze(ch0)
        # print(f'Ch0 squeezed: {ch0.shape}')
        # print(f'Img shape: {img.shape}')
        img = np.squeeze(img)
        # print(f'Img squeezed: {img.shape}')

        img = np.stack([ch0, img], axis=0) # shape (2, D, H, W)
        affine = ch0_nii.affine

        # print(f'[DEBUG] Combined vessel image shape: {img.shape}, ndim: {img.ndim}', flush=True)

    # all other datatypes have just 1 channel
    else:
        img = np.expand_dims(img, axis=0) # shape (1, D, H, W)

    # get label 
    # (all images, have 1 label, vessel channels are labeled together)
    label_nii = nib.load(label_path)
    label = label_nii.get_fdata()

    # correct dimensions
    if label.ndim == 4 and label.shape[-1] == 1:
        label = np.squeeze(label, axis=-1)

    label_affine = label_nii.affine

    # pad image and label if too small
    C, D, H, W = img.shape
    d, h, w = PATCH_SIZE
    if D < d or H < h or W < w:
        img = pad_to_min_size(img, (img.shape[0], max(D, d), max(H, h), max(W, w)))
        label = pad_to_min_size(label, (max(D, d), max(H, h), max(W, w)))

    # return image-label pair
    return img.astype(np.float32), label.astype(np.uint16), affine, label_affine


# function to extract patches from image-label pair
def extract_valid_patch(img, label):

    # get image and patch dimensions
    # print('Image shape:', img.shape, flush=True)
    C, D, H, W = img.shape
    d, h, w = PATCH_SIZE

    # generate random coordinates
    for i in range(100):
        z = random.randint(0, D - d)
        y = random.randint(0, H - h)
        x = random.randint(0, W - w)

        # print(f'D:{D} d:{d} H:{H} h:{h} W:{W} w:{w}')
        # print(f'D - d: {D-d}, H - h: {H-h}, W - w: {W-w}')
        # print(f'z:{z} y:{y} x:{x}')

        # get patch from label
        label_patch = label[z:z+d, y:y+h, x:x+w]

        # get patch from image, ensuring sufficient foreground in the patch
        if (label_patch > 0).sum() / label_patch.size > FOREGROUND_THRESHOLD:
            img_patch = img[:, z:z+d, y:y+h, x:x+w]
            print(f'[DEBUG] Patch extracted at (z={z}, y={y}, x={x})', flush=True)
            
            # return image-label pair
            # print(f'Saving patch.', flush=True)
            return img_patch, label_patch
        
    # if no patches found, return none
    print('[WARNING] Patch with sufficient foreground not found', flush=True)
    return None, None


# function to extract patches for a class
def extract_patches_for_class(class_name, config):

    # get image and label paths
    print(f'[INFO] Extracting patches for class: {class_name}', flush=True)
    img_paths = sorted(glob.glob(os.path.join(config['img_dir'], '*.nii.gz')))
    label_paths = sorted(glob.glob(os.path.join(config['label_dir'], '*.nii.gz')))

    # create output directory
    output_class_dir = os.path.join(OUTPUT_DIR, class_name)
    os.makedirs(output_class_dir, exist_ok=True)
    count = 0

    # load image-label pairs
    for img_path, label_path in tqdm(zip(img_paths, label_paths), total=len(label_paths)):
        img, label, affine_img, affine_lbl = load_image_label_pair(img_path, label_path, is_vessel=(class_name == 'vessel'))

        # try extracting up to 10 patches per volume
        for i in range(10):
            img_patch, label_patch = extract_valid_patch(img, label)

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
                             os.path.join(output_class_dir, f'patch{count:03d}_img.nii.gz'))
                    
                # save label as nifti
                nib.save(nib.Nifti1Image(label_patch, affine=affine_lbl), 
                         os.path.join(output_class_dir, f'patch_{count:03d}_label.nii.gz'))

                # increment counter and stop when quota has been reached
                count += 1
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





