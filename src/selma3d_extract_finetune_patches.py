import os
import nibabel as nib
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm

import argparse

# Configuration
INPUT_ROOT = Path("/midtier/paetzollab/scratch/ads4015/data_selma3d/SELMA3D")
OUTPUT_ROOT = Path("/midtier/paetzollab/scratch/ads4015/data_selma3d/lsm_fm_selma3d_finetune2")
PATCH_SIZE = (96, 96, 96)
PATCHES_PER_CLASS = 25

# Mapping from class folder to full label and image paths
DATA_CLASSES = {
    "brain_amyloid_plaque_patches": {
        "label": "annotation_brain_amyloid_plaque/AD_plaques/gt",
        "image": "brain_amyloid_plaque_patches/AD_plaques/raw"
    },
    "brain_c_fos_positive_patches": {
        "label": "annotation_brain_c_fos_positive/cFos-Active_Neurons/gt",
        "image": "brain_c_fos_positive_patches/cFos-Active_Neurons/raw"
    },
    "brain_cell_nucleus_patches": {
        "label": "annotation_brain_cell_nucleus/shannel_cells/gt",
        "image": "brain_cell_nucleus_patches/shannel_cells/raw"
    },
    "brain_vessels_patches": {
        "label": "annotation_brain_vessels/VessAP_vessel/gt",
        "image": "brain_vessels_patches/VessAP_vessel/raw"
    }
}

def load_nifti(path):
    return nib.load(str(path))

def extract_patch(volume, start, size):
    z, y, x = start
    dz, dy, dx = size
    return volume[z:z+dz, y:y+dy, x:x+dx]

def save_patch_with_label(patch, label_patch, save_dir, patch_id, vol_id, ch, affine):
    save_dir.mkdir(parents=True, exist_ok=True)
    filename_pt = f"patch_{patch_id:03d}_vol{vol_id:03d}_ch{ch}.pt"
    filename_nii = f"patch_{patch_id:03d}_vol{vol_id:03d}_ch{ch}.nii.gz"
    filename_lbl_nii = f"patch_{patch_id:03d}_vol{vol_id:03d}_label.nii.gz"

    torch.save({
        "image": torch.tensor(patch).float(),
        "label": torch.tensor(label_patch).long()
    }, save_dir / filename_pt)

    nib.save(nib.Nifti1Image(patch.astype(np.float32), affine), save_dir / filename_nii)
    nib.save(nib.Nifti1Image(label_patch.astype(np.uint8), affine), save_dir / filename_lbl_nii)

    return filename_pt, filename_nii, filename_lbl_nii

def pad_to_minimum_size(volume, target_size):
    pad_width = []
    for dim, target in zip(volume.shape[-3:], target_size):
        total_pad = max(0, target - dim)
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_width.append((pad_before, pad_after))
    pad_width = [(0, 0)] * (volume.ndim - 3) + pad_width
    return np.pad(volume, pad_width, mode='constant')

def get_centered_coordinates(fg_mask, shape, patch_size):
    dz, dy, dx = patch_size
    valid_coords = np.argwhere(fg_mask)
    np.random.shuffle(valid_coords)
    for zc, yc, xc in valid_coords:
        z = zc - dz // 2
        y = yc - dy // 2
        x = xc - dx // 2
        if z >= 0 and y >= 0 and x >= 0 and z + dz <= shape[0] and y + dy <= shape[1] and x + dx <= shape[2]:
            yield (z, y, x)

def process_class(class_folder, paths):
    patch_id_counter = 0

    input_dir = INPUT_ROOT / paths["image"]
    label_dir = INPUT_ROOT / paths["label"]
    output_dir = OUTPUT_ROOT / class_folder.split("_", 1)[-1]

    image_files = sorted(input_dir.glob("patchvolume_*_0000.nii.gz"))
    print(f"{class_folder}: Found {len(image_files)} image files in {input_dir}", flush=True)

    if len(image_files) == 0:
        print(f"No image files found for {class_folder}, skipping.\n", flush=True)
        return

    np.random.shuffle(image_files)
    num_volumes = len(image_files)
    if num_volumes == 0:
        print(f"No volumes found for {class_folder}, skipping.", flush=True)
        return

    patches_per_volume = [PATCHES_PER_CLASS // num_volumes] * num_volumes
    for idx in np.random.choice(num_volumes, PATCHES_PER_CLASS % num_volumes, replace=False):
        patches_per_volume[idx] += 1

    volume_patch_counts = [0] * num_volumes
    pending_volumes = list(range(num_volumes))

    while sum(volume_patch_counts) < PATCHES_PER_CLASS and pending_volumes:
        for i in pending_volumes.copy():
            if volume_patch_counts[i] >= patches_per_volume[i]:
                pending_volumes.remove(i)
                continue

            image_file = image_files[i]
            vol_id = int(image_file.stem.split("_")[1])
            label_file = label_dir / f"patchvolume_{vol_id:03d}.nii.gz"

            if not label_file.exists():
                print(f"  Skipping volume {vol_id:03d}: missing label file {label_file}", flush=True)
                pending_volumes.remove(i)
                continue

            label_nii = load_nifti(label_file)
            label = label_nii.get_fdata()
            affine = label_nii.affine
            label = pad_to_minimum_size(label, PATCH_SIZE)
            fg_mask = label != 0

            if not fg_mask.any():
                print(f"  Skipping volume {vol_id:03d}: label has no foreground", flush=True)
                pending_volumes.remove(i)
                continue

            try:
                image_channels = [pad_to_minimum_size(load_nifti(image_file).get_fdata(), PATCH_SIZE)]
                if "vessels" in class_folder:
                    ch1_file = image_file.with_name(image_file.name.replace("0000", "0001"))
                    if ch1_file.exists():
                        image_channels.append(pad_to_minimum_size(load_nifti(ch1_file).get_fdata(), PATCH_SIZE))
                image_channels = np.stack(image_channels, axis=0)
                if image_channels.ndim != 4:
                    raise ValueError(f"Expected 4D image, got shape {image_channels.shape}")
                C, Z, Y, X = image_channels.shape
            except Exception as e:
                print(f"  Skipping volume {vol_id:03d}: malformed image shape {image_channels.shape} ({e})", flush=True)
                pending_volumes.remove(i)
                continue

            if Z < PATCH_SIZE[0] or Y < PATCH_SIZE[1] or X < PATCH_SIZE[2]:
                print(f"  Skipping volume {vol_id:03d}: dimensions too small ({Z}, {Y}, {X})", flush=True)
                pending_volumes.remove(i)
                continue

            for z, y, x in get_centered_coordinates(fg_mask, (Z, Y, X), PATCH_SIZE):
                label_patch = extract_patch(label, (z, y, x), PATCH_SIZE)
                if np.any(label_patch):
                    for ch in range(C):
                        img_patch = extract_patch(image_channels[ch], (z, y, x), PATCH_SIZE)
                        save_patch_with_label(
                            img_patch, label_patch, output_dir, patch_id_counter, vol_id, ch, affine
                        )
                    patch_id_counter += 1
                    volume_patch_counts[i] += 1
                    if volume_patch_counts[i] >= patches_per_volume[i]:
                        break

    print(f"\nSummary for {class_folder}:", flush=True)
    for idx, count in enumerate(volume_patch_counts):
        vol_id = int(image_files[idx].stem.split("_")[1])
        if count == 0:
            print(f"  Volume {vol_id:03d}: 0 patches (skipped)", flush=True)
        else:
            print(f"  Volume {vol_id:03d}: {count} patches", flush=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--class_name", type=str, required=True, choices=list(DATA_CLASSES.keys()))
    args = parser.parse_args()

    class_name = args.class_name
    class_paths = DATA_CLASSES[class_name]
    process_class(class_name, class_paths)
