# tif_stack_get_patches_multichannel.py - python script to extract patches from stack of tif images (with multiple channel)

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

sys.path.append('/home/ads4015/ssl_project/preprocess_patches/src/')
from tif_patches_functions import deterministic_seed, make_sort_key


# --- Functions ---


# *** arg parsing ***

# function to parse command line args
def parse_args():

    p = argparse.ArgumentParser(description='Extract patches from Allen Developing Mouse dataset')
    
    # i/o and selection
    p.add_argument('--input_dir', type=str, required=True, help='Input directory containing per slice tiffs, e.g. Z00001_ch00.tif, ...')
    p.add_argument('--output_dir', type=str, required=True, help='Output directory for extracted patches')
    p.add_argument('--pattern', type=str, default='Z*_ch*.tif', help='Pattern to match input tiff files (default: Z*_ch*.tif)')
    p.add_argument('--sort_regex', type=str, default=r'Z(\d+)', help='Regex to extract slice number for sorting (default: Z(\\d+))')

    # patches
    p.add_argument('--patch_size', type=int, default=96, help='Size of the patches to extract, cubic (default: 96)')
    p.add_argument('--num_patches', type=int, default=1, help='Number of patches to extract from each image (default: 1)')
    p.add_argument('--min_fg', type=float, default=0.05, help='Minimum foreground fraction to consider a patch (default: 0.05)')
    p.add_argument('--stride', type=int, default=None, help='Optional stride for sliding window (default: patch_size for non-overlap)')
    p.add_argument('--max_candidates', type=int, default=2000, help='Maximum number of candidates to consider per channel (default: 2000)')

    # channels and determinism
    p.add_argument('--seed', type=int, default=100, help='Random seed for reproducibility, combined with per-folder hash for determinism (default: 100)')
    p.add_argument('--channels', type=str, default='all', help="Selection of channels to extract: 'all' or comma-separated indices, e.g. '0,1,2'")
    
    return p.parse_args()


# *** file reading and cropping ***


# function to read tif slice and return (Y, X, C)
def read_slice_yxc(path):

    arr = tiff.imread(path)
    if arr.ndim == 2: # add channel dimension if missing: (Y,X) -> (Y,X,1)
        arr = arr[..., None]
    elif arr.ndim == 3:
        if arr.shape[-1] in (3,4): # has channel dimension: (Y,X,C)
            pass
        elif arr.shape[0] in (3,4): # likely channel-first: (C,Y,X) -> (Y,X,C)
            arr = np.moveaxis(arr, 0, -1)
        else:
            raise ValueError(f'Slice {path} has unexpected shape {arr.shape}')
    else:
        raise ValueError(f'Slice {path} has unsupported ndim={arr.ndim}')
    
    return arr


# function to read cropped region (Y,X,*) from tiff page
# tries aszarr-windowing, falls back to full read
def read_crop_xyc(path, y0, y1, x0, x1):

    # try aszarr windowing
    try:
        with tiff.TiffFile(path) as tf:
            z = tf.aszarr(series=0) # zarr style array interface
            region = np.asarray(z[y0:y1, x0:x1, ...]) # (y1-y0, x1-x0, C)
            if region.ndim == 2: # add channel dimension if missing: (Y,X) -> (Y,X,1)
                region = region[..., None]
            elif region.ndim == 3 and region.shape[0] in (3,4) and region.shape[-1] not in (3,4):
                region = np.moveaxis(region, 0, -1) # likely channel-first: (C,Y,X) -> (Y,X,C)
            return region
        
    # fallback to load full slice then crop
    except Exception:
        full = read_slice_yxc(path) # (Y,X,C)
        return full[y0:y1, x0:x1, :] # (y1-y0, x1-x0, C)
    

# *** file listing and probing ***


# function to list sorted files
def list_sorted_files(input_dir, pattern, sort_regex): 

    files = glob.glob(os.path.join(input_dir, pattern))
    files = [f for f in files if not os.path.basename(f).startswith('.')]  # ignore hidden files
    if not files:
        raise FileNotFoundError(f'No tiff files matching pattern "{pattern}" in "{input_dir}"')
    
    files = sorted(files, key=make_sort_key(sort_regex))
    print(f'[INFO] First 5 files after sort:', flush=True)
    for f in files[:5]:
        print(f'   {os.path.basename(f)}', flush=True)

    return files


# function to open first slice and get shape
def probe_shape(files):

    first = read_slice_yxc(files[0]) # (Y,X,C)
    Y, X, C = first.shape
    Z = len(files)
    return Z, Y, X, C, first.dtype


# *** thresholding ***

# estimate per channel otsu thesholds from sampled pixels across a subset of slices
def compute_global_otsu_threshold_channels_sampled(files, zstep=32, xy_down=16, cap=8_000_000):

    assert zstep >= 1 and xy_down >= 1
    sample_c = None
    total = 0
    step = max(1, zstep)

    for zi in range(0, len(files), step):

        # prefer windowed read
        try:
            with tiff.TiffFile(files[zi]) as tf:
                z = tf.aszarr(series=0)
                sl = np.asarray(z[::xy_down, ::xy_down, ...]) # (Y//xy_down, X//xy_down, C)
                if sl.ndim == 2: # add channel dimension if missing: (Y,X) -> (Y,X,1)
                    sl = sl[..., None]
                elif sl.ndim == 3 and sl.shape[0] in (3,4) and sl.shape[-1] not in (3,4):
                    sl = np.moveaxis(sl, 0, -1) # likely channel-first: (C,Y,X) -> (Y,X,C)
        except Exception:
            sl = read_slice_yxc(files[zi])[::xy_down, ::xy_down, :]

        flat = sl.reshape(-1, sl.shape[-1]) # (N, C)
        if sample_c is None:
            sample_c = [flat[:, c].copy() for c in range(sl.shape[-1])]
        else:
            for c in range(sl.shape[-1]):
                sample_c[c] = np.concatenate((sample_c[c], flat[:, c]), axis=0)

        total += flat.shape[0]
        if sample_c[0].size >= cap:
            break

    th = np.zeros((len(sample_c),), dtype=np.float32)
    for c in range(len(sample_c)):
        v = sample_c[c]
        vmin, vmax = float(v.min()), float(v.max())
        th[c] = vmin if vmin == vmax else float(threshold_otsu(v))

    return th


# *** patch extraction (with streaming to reduce memory) ***

# patch extraction function, builds a (ps,ps,ps) patch per requested channel by reading only needed slices and cropping
def extract_patch(files, z0, y0, x0, ps, ch_indices, out_dtype):

    patches = {c: np.empty((ps, ps, ps), dtype=out_dtype) for c in ch_indices}

    for dz in range(ps):
        path = files[z0 + dz]
        tile = read_crop_xyc(path, y0, y0 + ps, x0, x0 + ps) # (ps, ps, C)
        for c in ch_indices:
            patches[c][:, :, dz] = tile[:, :, c]

    return patches


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
    print(f'[INFO] Number of patches per channel (K): {args.num_patches}', flush=True)
    print(f'[INFO] Minimum foreground fraction: {args.min_fg}', flush=True)
    print(f'[INFO] Seed used: {seed_used}', flush=True)

    # list and sort files
    files = list_sorted_files(args.input_dir, args.pattern, args.sort_regex)

    # probe dimension and type without loading full volume
    Z, Y, X, C, dtype0 = probe_shape(files)
    
    # select channel based on command line input
    if args.channels.strip().lower() == 'all':
        ch_indices = list(range(C))
    else:
        ch_indices = [int(s) for s in args.channels.split(',') if s.strip() != '']
        for c in ch_indices:
            if c < 0 or c >= C:
                raise ValueError(f'Requested channel {c} but volume has {C} channels {ch_indices}')
    print(f'[INFO] Volume dims: Z={Z}, Y={Y}, X={X}, C={C}. Using channels: {ch_indices}', flush=True)

    # check patch size
    ps = int(args.patch_size)
    if Z < ps or Y < ps or X < ps:
        print(f'[WARN] Volume ({Z},{Y},{X}) smaller than patch size ({ps}), no patches can be extracted. Exiting', flush=True)
        return

    # estimate per channel otsu thresholds from sampled pixels
    thresh = compute_global_otsu_threshold_channels_sampled(files)
    th_view = {c: float(thresh[c]) for c in ch_indices}
    print(f'[INFO] Global Otsu thresholds for each channel: {th_view}', flush=True)

    # reservoir sampling logic per channel
    K = int(args.num_patches)
    reservoirs = {c: [] for c in ch_indices} # each: {'patch': (ps,ps,ps), 'z':..., 'y':..., 'x':..., 'fg':...}
    seen = {c: 0 for c in ch_indices} # number of qualifying candidates seen so far

    # candidate loop (with random x/y/z positions)
    zmax, ymax, xmax = (Z - ps + 1, Y - ps + 1, X - ps + 1)
    attempts = 0
    while attempts < args.max_candidates and any(len(reservoirs[c]) < K for c in ch_indices):
        attempts += 1
        z0 = random.randrange(0, zmax)
        y0 = random.randrange(0, ymax)
        x0 = random.randrange(0, xmax)

        # read only needed crop from each ps slice
        patches = extract_patch(files, z0, y0, x0, ps, ch_indices, dtype0) # {c: (ps,ps,ps)}

        # per channel foreground gating (compare to max(threshold, patch min))
        for c in ch_indices:
            pch = patches[c]
            pmin, pmax = float(pch.min()), float(pch.max())
            if pmin == pmax:
                fg = 0.0
            else:
                fg = float((pch > max(thresh[c], pmin)).mean())

            if fg < args.min_fg:
                continue

            # reservoir sampling
            seen[c] += 1
            if len(reservoirs[c]) < K:
                reservoirs[c].append({'patch':pch, 'z':z0, 'y':y0, 'x':x0, 'fg':fg})
            else:
                r = random.randint(1, seen[c]) # 1..seen (inclusive)
                if r <= K: # replace random slot 1..K
                    reservoirs[c][r-1] = {'patch':pch, 'z':z0, 'y':y0, 'x':x0, 'fg':fg}

        # periodic log
        if attempts % 50 == 0:
            fill = {c: f'{len(reservoirs[c])}/{K}' for c in ch_indices}
            print(f'[INFO] Attempts: {attempts}, Reservoir fill: {fill}', flush=True)

    print(f'[INFO] Attempts done: {attempts}', flush=True)
    seen_view = {c: seen[c] for c in ch_indices}
    print(f'[INFO] Qualifying candidates seen (per channel): {seen_view}.', flush=True)

    total_saved = sum(len(reservoirs[c]) for c in ch_indices)
    if total_saved == 0:
        print(f'[WARN] No qualifying patches found.', flush=True)
        return
    
    # save patches as nifti
    # (X,Y,Z) ordering for nifti, so transpose from (Z,Y,X)
    base_folder_tag = os.path.basename(os.path.normpath(args.input_dir))
    saved = 0
    for c in ch_indices:
        for i, item in enumerate(reservoirs[c]):
        
            # preserve integer dtype if present, otherwise keep original
            out_dtype = np.uint16 if np.issubdtype(dtype0, np.integer) else dtype0
            pch = item['patch'].astype(out_dtype, copy=False) # (Z,Y,X)
            data_xyz = np.transpose(pch, (1, 0, 2)) # (Z,Y,X) -> (X,Y,Z) for nifti
            img = nib.Nifti1Image(data_xyz, np.eye(4, dtype=np.float32))
            outname = f'{base_folder_tag}_ps{ps}_p{i}_z{item["z"]}_y{item["y"]}_x{item["x"]}_c{c}.nii.gz'
            outpath = os.path.join(args.output_dir, outname)
            nib.save(img, outpath)
            print(f'[INFO] Saved channel {c} patch {i} to {outpath}', flush=True)
            saved += 1

    print(f'[INFO] Saved {saved} patches across channels to {args.output_dir}', flush=True)
   

# --- Main ---

# main entry point
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f'[ERROR] {e}', flush=True)
        sys.exit(1)




