# allen_developing_mouse_get_patches.py - python script to extract patches from Allen Developing Mouse dataset

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

# function to parse command line args
def parse_args():
    p = argparse.ArgumentParser(description='Extract patches from Allen Developing Mouse dataset')
    p.add_argument('--input_dir', type=str, required=True, help='Input directory containing per slice tiffs, e.g. Z00001_ch00.tif, ...')
    p.add_argument('--output_dir', type=str, required=True, help='Output directory for extracted patches')
    p.add_argument('--patch_size', type=int, default=96, help='Size of the patches to extract, cubic (default: 96)')
    p.add_argument('--num_patches', type=int, default=1, help='Number of patches to extract from each image (default: 1)')
    p.add_argument('--min_fg', type=float, default=0.05, help='Minimum foreground fraction to consider a patch (default: 0.05)')
    p.add_argument('--pattern', type=str, default='Z*_ch*.tif', help='Pattern to match input tiff files (default: Z*_ch*.tif)')
    p.add_argument('--seed', type=int, default=100, help='Random seed for reproducibility, combined with per-folder hash for determinism (default: 100)')
    p.add_argument('--stride', type=int, default=None, help='Optional stride for sliding window (default: patch_size for non-overlap)')
    return p.parse_args()


# function to sort tifs by Z**** number
def slice_sort_key(path):

    # regex matches
    _num_re = re.compile(r'Z(\d+)', re.IGNORECASE)

    # search for Z**** pattern
    name = os.path.basename(path)
    m = _num_re.search(name)
    return int(m.group(1))


# function to read tif stack
def read_stack(dirpath, pattern):

    # get files
    files = sorted(glob.glob(os.path.join(dirpath, pattern)), key=slice_sort_key)
    files = [f for f in files if not os.path.basename(f).startswith('.')]  # ignore hidden files
    if not files:
        raise FileNotFoundError(f'No tiff files matching pattern "{pattern}" in "{dirpath}"')
    
    # create list of slices
    slices = []
    for f in files:
        arr = tiff.imread(f)
        if arr.ndim != 2:
            raise ValueError(f'Slice {f} is not 2D; got shape {arr.shape}')
        slices.append(arr)
    
    # create and return volume
    vol = np.stack(slices, axis=0)
    return vol, files


# function to set seed
def deterministic_seed(folder_path, user_seed=None):

    # hash seed
    base = int(hashlib.md5(folder_path.encode('utf-8')).hexdigest()[:8], 16)
    seed = base if user_seed is None else ((base ^ user_seed) & 0xFFFFFFFF)

    # set seed
    random.seed(seed)
    np.random.seed(seed)
    return seed


# compute global otsu threshold on downsampled image (to increase speed)
def compute_global_otsu_threshold(vol, downsample_factor=4):

    # downsample and threshold
    vol_ds = vol[::downsample_factor, ::downsample_factor, ::downsample_factor]
    vmin, vmax = float(vol_ds.min()), float(vol_ds.max())
    if vmin == vmax:
        return vmin
    return float(threshold_otsu(vol_ds))


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


# main
def main():

    # parse args
    args = parse_args()

    # set seed
    seed_used = deterministic_seed(os.path.abspath(args.input_dir), args.seed)

    # create output dirs
    os.makedirs(args.output_dir, exist_ok=True)

    # print info
    print(f'[INFO] Input dir: {args.input_dir}', flush=True)
    print(f'[INFO] Output dir: {args.output_dir}', flush=True)
    print(f'[INFO] Patch size: {args.patch_size}', flush=True)
    print(f'[INFO] Number of patches: {args.num_patches}', flush=True)
    print(f'[INFO] Minimum foreground fraction: {args.min_fg}', flush=True)
    print(f'[INFO] Seed used: {seed_used}', flush=True)

    # read volume (Z, Y, X)
    vol, files = read_stack(args.input_dir, args.pattern)
    Z, Y, X = vol.shape
    ps = int(args.patch_size)
    if Z < ps or Y < ps or X < ps:
        print(f'[WARN] Volume shape ({vol.shape}) smaller than patch size: {ps}, skipping', flush=True)
        return
    
    # threshold
    thresh = compute_global_otsu_threshold(vol)
    print(f'[INFO] Global Otsu threshold: {thresh}', flush=True)

    # reservoir sampling logic
    K = int(args.num_patches)
    reservoir: list[dict] = [] # each: {"patch": np.ndarray, "z":int,"y":int,"x":int,"fg":float}
    seen = 0 # number of qualifying candidates seen so far

    stride = args.stride if args.stride is not None else ps

    for (z, y, x) in get_grid_candidates(vol.shape, ps, stride):

        patch = vol[z:z+ps, y:y+ps, x:x+ps]
        pmin, pmax = float(patch.min()), float(patch.max())
        if pmin == pmax:
            fg = 0.0
        else:
            # guard against trivially dark patches
            fg = float((patch > max(thresh, pmin)).mean())
        
        # ensure sufficient foreground
        if fg < args.min_fg:
            continue

        # update reservoir with qualifying candidate
        seen += 1
        if len(reservoir) < K:
            reservoir.append({'patch':patch.copy(), 'z':z, 'y':y, 'x':x, 'fg':fg})
        else:
            r = random.randint(1, seen) # 1..seen (inclusive)
            if r <= K: # replace random slot 1..K
                idx = r - 1
                reservoir[idx] = {'patch':patch.copy(), 'z':z, 'y':y, 'x':x, 'fg':fg}
    
    # log info
    print(f'[INFO] Qualifying candidates seen: {seen}. Will save up to {len(reservoir)} patches.', flush=True)

    if not reservoir:
        print(f'[WARN] No qualifying patches found.', flush=True)
        return

    # save reservoir patches as nifti
    base_folder_tag = os.path.basename(os.path.normpath(args.input_dir))

    for i, item in enumerate(reservoir):
        patch = item['patch'].astype(np.uint16)
        z, y, x, fg = item['z'], item['y'], item['x'], item['fg']
        data_xyz = np.transpose(patch, (2, 1, 0)) # (Z,Y,X) -> (X,Y,Z)
        affine = np.eye(4, dtype=np.float32)
        img = nib.Nifti1Image(data_xyz, affine)
        outname = f'{base_folder_tag}_ps{ps}_p{i}_z{z}_y{y}_x{x}.nii.gz'
        outpath = os.path.join(args.output_dir, outname)
        nib.save(img, outpath)
        print(f'[INFO] Saved patch {i} to {outpath}', flush=True)

    # indicate completion
    print(f'[INFO] Saved {len(reservoir)} patches to {args.output_dir}', flush=True)


# main entry point
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'[ERROR] {e}', flush=True)
        sys.exit(1)
