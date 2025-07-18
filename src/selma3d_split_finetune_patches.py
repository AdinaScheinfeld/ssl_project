# Script to split annotated SELMA3D patches into train/val sets

# --- Setup ---

# imports
import argparse
import os
from pathlib import Path
import random
import shutil

# --- Functions ---

# function to extract volume id from filename formatted as: patch_xxx_volxxx_chx.pt
def extract_volume_id(file_path):
    stem = file_path.stem
    parts = stem.split('_')
    for p in parts:
        if p.startswith('vol'):
            return p
    return None


# function to group files by volume id (to prevent images from the same volume in both train and val sets)
def group_by_volume(pt_files):

    # create a dictionary to hold volume id to file mapping
    volume_to_files = {}

    # iterate through files and group by volume id
    for file in pt_files:
        vol_id = extract_volume_id(file)
        if vol_id is None:
            continue
        volume_to_files.setdefault(vol_id, []).append(file)
    
    # return the grouped dictionary with mapping
    return volume_to_files


# function to split volumes into train and val sets
def split_volumes(volume_to_files, val_frac, seed):

    random.seed(seed)

    # get list of volume ids and shuffle
    volume_ids = list(volume_to_files.keys())
    random.shuffle(volume_ids)

    # get split index
    n_val = max(1, int(len(volume_ids) * val_frac))
    val_vols = set(volume_ids[:n_val])
    train_vols = set(volume_ids[n_val:])

    # create train and val file lists
    train_files = [f for vid in train_vols for f in volume_to_files[vid]]
    val_files = [f for vid in val_vols for f in volume_to_files[vid]]
    return train_files, val_files, train_vols, val_vols


# main
def splt_and_copy(input_root, output_root, val_frac=0.2, seed=100):

    # get input and output paths and classes
    input_root = Path(input_root)
    output_root = Path(output_root)
    classes = [d for d in input_root.iterdir() if d.is_dir()]

    # iterate through each class directory
    for class_dir in classes:

        # split
        print(f'\nProcessing {class_dir.name}...')
        pt_files = list(class_dir.glob('*.pt')) # get all .pt files in the class directory
        volume_to_files = group_by_volume(pt_files) # group files by volume id
        train_files, val_files, train_vols, val_vols = split_volumes(volume_to_files, val_frac, seed) # split volumes into train and val sets

        print('  Train volumes:', sorted(train_vols))
        print('  Val volumes:', sorted(val_vols))

        # copy files to train and val directories
        for split, file_list in zip(['train', 'val'], [train_files, val_files]):
            split_dir = output_root / split / class_dir.name
            split_dir.mkdir(parents=True, exist_ok=True) # create output directory if it doesn't exist
            print(f'  {split.upper()} PATCHES:')
            for file in sorted(file_list):
                print(f'    {file.name}')
                shutil.copy(file, split_dir / file.name)

        print(f'  -> {len(train_files)} train files, {len(val_files)} val files')


# --- Main entry point ---

if __name__ == '__main__':

    # parse command line args
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_root', type=str, required=True, help='Path to input directory with .pt files')
    parser.add_argument('--output_root', type=str, required=True, help='Path to output directory for train/val splits')
    parser.add_argument('--val_frac', type=float, default=0.2, help='Fraction of data to use for validation')
    parser.add_argument('--seed', type=int, default=100, help='Random seed for reproducibility')
    args = parser.parse_args()

    # run the split and copy function
    splt_and_copy(args.input_root, args.output_root, args.val_frac, args.seed)











