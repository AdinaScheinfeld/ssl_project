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
    p.add_argument('--sort_regex', type=str, default=r'Z(\d+)', help='Regex to extract slice number for sorting (default: Z(\\d+))')
    p.add_argument('--seed', type=int, default=100, help='Random seed for reproducibility, combined with per-folder hash for determinism (default: 100)')
    p.add_argument('--stride', type=int, default=None, help='Optional stride for sliding window (default: patch_size for non-overlap)')
    p.add_argument('--channels', type=str, default='all', help="Selection of channels to extract: 'all' or comma-separated indices, e.g. '0,1,2'")
    return p.parse_args()


# # function to sort tifs by Z**** number
# def slice_sort_key(path):

#     # regex matches
#     _num_re = re.compile(r'Z(\d+)', re.IGNORECASE)

#     # search for Z**** pattern
#     name = os.path.basename(path)
#     m = _num_re.search(name)
#     return int(m.group(1))

# function to sort slices in tiff stack
def make_sort_key(regex_pattern):

    # regex matches
    regex = re.compile(regex_pattern)

    # sort key
    def sort_key(path):
        name = os.path.basename(path)
        m = regex.search(name)
        if not m:
            raise ValueError(f'Filename "{name}" does not match regex "{regex_pattern}"')
        return int(m.group(1))

    return sort_key


# function to read tif stack
def read_stack_channels(dirpath, pattern, sort_regex):

    # get files
    files = glob.glob(os.path.join(dirpath, pattern))
    files = [f for f in files if not os.path.basename(f).startswith('.')]  # ignore hidden files
    if not files:
        raise FileNotFoundError(f'No tiff files matching pattern "{pattern}" in "{dirpath}"')
    
    # build key function
    sort_key = make_sort_key(sort_regex)
    files = sorted(files, key=sort_key)

    # create list of slices
    slices = []
    for f in files:
        arr = tiff.imread(f)
        if arr.ndim == 2: # add channel dimension if missing: (Y,X) -> (Y,X,1)
            arr = arr[..., None]
        elif arr.ndim == 3:
            if arr.shape[-1] in (3,4): # has channel dimension: (Y,X,C)
                pass
            elif arr.shape[0] in (3,4): # likely channel-first: (C,Y,X) -> (Y,X,C)
                arr = np.moveaxis(arr, 0, -1)
            else:
                raise ValueError(f'Slice {f} has unexpected shape {arr.shape}')
        else:
            raise ValueError(f'Slice {f} has unsupported ndim={arr.ndim}')

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
# vol: (Z, Y, X, C)
def compute_global_otsu_threshold_channels(vol, downsample_factor=4):

    # downsample for speed
    vol_ds = vol[::downsample_factor, ::downsample_factor, ::downsample_factor, :]

    # threshold each channel individually
    C = vol_ds.shape[-1]
    th = []
    for c in range(C):
        ch = vol_ds[..., c]
        vmin, vmax = float(ch.min()), float(ch.max())
        th.append(vmin if vmin == vmax else float(threshold_otsu(ch)))

    # return array with shape: (C,)
    return np.array(th, dtype=float)


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

    # read volume (Z, Y, X, C)
    vol, files = read_stack_channels(args.input_dir, args.pattern, args.sort_regex)
    Z, Y, X, C = vol.shape
    ps = int(args.patch_size)
    if Z < ps or Y < ps or X < ps:
        print(f'[WARN] Volume shape ({vol.shape}) smaller than patch size: {ps}, skipping', flush=True)
        return
    
    # select channel based on command line input
    if args.channels.strip().lower() == 'all':
        ch_indices = list(range(C))
    else:
        ch_indices = [int(s) for s in args.channels.split(',') if s.strip() != '']
        for c in ch_indices:
            if c < 0 or c >= C:
                raise ValueError(f'Requested channel {c} but volume has {C} channels {ch_indices}')
    print(f'[INFO] Volume: {vol.shape} (Z,Y,X,C). Using channels: {ch_indices}', flush=True)

    # threshold
    thresh = compute_global_otsu_threshold_channels(vol) # length C
    print(f'[INFO] Global Otsu thresholds for each channel: {c: float(thresh[c]) for c in ch_indices}', flush=True)

    # reservoir sampling logic per channel
    K = int(args.num_patches)
    reservoirs = {c: [] for c in ch_indices} # each: {'patch': (ps,ps,ps), 'z':..., 'y':..., 'x':..., 'fg':...}
    seen = {c: 0 for c in ch_indices} # number of qualifying candidates seen so far

    stride = args.stride if args.stride is not None else ps

    for (z, y, x) in get_grid_candidates((Z, Y, X), ps, stride):

        patch = vol[z:z+ps, y:y+ps, x:x+ps, :] # (ps,ps,ps,C)

        for c in ch_indices:
            pch = patch[..., c]
            pmin, pmax = float(pch.min()), float(pch.max())
            if pmin == pmax:
                fg = 0.0
            else:
                # guard against trivially dark patches
                fg = float((pch > max(thresh[c], pmin)).mean())
            
            # ensure sufficient foreground
            if fg < args.min_fg:
                continue

            # update reservoir with qualifying candidate
            seen[c] += 1
            if len(reservoirs[c]) < K:
                reservoirs[c].append({'patch':pch.copy(), 'z':z, 'y':y, 'x':x, 'fg':fg})
            else:
                r = random.randint(1, seen[c]) # 1..seen (inclusive)
                if r <= K: # replace random slot 1..K
                    reservoirs[c][r-1] = {'patch':pch.copy(), 'z':z, 'y':y, 'x':x, 'fg':fg}
    
    # log info
    print(f'[INFO] Qualifying candidates seen (per channel): {c: seen[c] for c in ch_indices}.', flush=True)
    total_saved = sum(len(reservoirs[c]) for c in ch_indices)
    if total_saved == 0:
        print(f'[WARN] No qualifying patches found.', flush=True)
        return

    # save reservoir patches as nifti
    base_folder_tag = os.path.basename(os.path.normpath(args.input_dir))

    for c in ch_indices:
        for i, item in enumerate(reservoirs[c]):
            patch = item['patch'].astype(np.uint16) # (ps,ps,ps)
            z, y, x, fg = item['z'], item['y'], item['x'], item['fg']
            data_xyz = np.transpose(patch, (2, 1, 0)) # (Z,Y,X) -> (X,Y,Z) for nifti
            affine = np.eye(4, dtype=np.float32)
            img = nib.Nifti1Image(data_xyz, affine)
            outname = f'{base_folder_tag}_ps{ps}_p{i}_z{z}_y{y}_x{x}_c{c}.nii.gz'
            outpath = os.path.join(args.output_dir, outname)
            nib.save(img, outpath)
            print(f'[INFO] Saved channel {c} patch {i} to {outpath}', flush=True)

    # indicate completion
    print(f'[INFO] Saved {total_saved} patches across channels to {args.output_dir}', flush=True)


# main entry point
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'[ERROR] {e}', flush=True)
        sys.exit(1)
