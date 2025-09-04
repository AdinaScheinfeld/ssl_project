# tiff_volume_get_patches.py - python script to extract patches from tif/nifti volume

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
from tif_patches_functions import compute_global_otsu_threshold, deterministic_seed, get_grid_candidates


# --- Functions ---

# *** I/O functions ***

# helper function to determine if file is tif
def _is_tif(path):
    p = path.lower()
    return p.endswith('.tif') or p.endswith('.tiff')

# helper function to determine if file is nifti
def _is_nifti(path):
    p = path.lower()
    return p.endswith('.nii') or p.endswith('.nii.gz')


# function to read single tif volume
# supports 3d array (Z,Y,X), 4d array (Z,Y,X,C), 2d array treated as (1,Y,X)
def read_single_tiff(filepath, channel):

    # read filepath
    arr = tiff.imread(filepath)

    # 2d array
    if arr.ndim == 2:
        vol = arr[np.newaxis, ...] # (1,Y,X)
        return vol, [filepath]
    
    # 3d array (assume (Z,Y,X))
    if arr.ndim == 3:
        return arr, [filepath]
    
    # 4d array (assume (Z,Y,X,C))
    if arr.ndim == 4:
        if channel is None:
            raise ValueError(f'Tiff {filepath} has shape {arr.shape} (likely (Z,Y,X,C)). Please specify a channel using --channel (0..{arr.shape[3]-1}).')
        
        if channel < 0 or channel >= arr.shape[3]:
            raise ValueError(f'--channel {channel} out of range for last dim size {arr.shape[3]} in "{filepath}"')
        
        vol = arr[..., channel] 
        return vol, [filepath]
    
    raise ValueError(f'Unsupported number of dimensions ({arr.ndim}) in tiff file: {filepath}')


# function to read single nifti volume
def read_single_nifti(filepath, nifti_index=None):

    # load nifti
    img = nib.load(filepath)

    # use array proxy to avoid immediate full load
    data = np.asanyarray(img.dataobj)

    # (X,Y,Z) -> (Z,Y,X)
    if data.ndim == 3:
        vol = np.transpose(data, (2, 1, 0))
        return vol, [filepath]

    # (X,Y,Z,C) -> (Z,Y,X,C)
    if data.ndim == 4:

        # ensure index specified
        if nifti_index is None:
            raise ValueError(f'Nifti {filepath} has shape {data.shape} (likely (X,Y,Z,C)). Please specify an index using --nifti_index (0..{data.shape[3]-1}).')
        
        # ensure valid index
        if nifti_index < 0 or nifti_index >= data.shape[3]:
            raise ValueError(f'--nifti_index {nifti_index} out of range for last dim size {data.shape[3]} in "{filepath}"')
        
        # transpose
        vol_xyz = data[..., nifti_index]
        vol = np.transpose(vol_xyz, (2, 1, 0))
        return vol, [filepath]
    
    raise ValueError(f'Unsupported number of dimensions ({data.ndim}) in nifti file: {filepath}')


# function to read tif/nifti when appropriate
def read_single_image(filepath, channel=None, nifti_index=None):

    if _is_tif(filepath):
        return read_single_tiff(filepath, channel)
    
    if _is_nifti(filepath):
        return read_single_nifti(filepath, nifti_index)
    
    raise ValueError(f'Unsupported file format (not tif or nifti): {filepath}')


# function to load list of volumes
def load_path_list(list_path):

    # list of paths
    paths = []

    # get paths
    with open(list_path, 'r') as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith('#'):
                continue
            paths.append(s)
    if not paths:
        raise ValueError(f'No usable paths found in list: {list_path}')
    return paths


# function to clean base tag for saving
def clean_base_tag(filepath):

    # get basename of filepath
    name = os.path.basename(filepath)

    # remove extensions
    name = re.sub(r'\.nii(\.gz)?$', '', name, flags=re.IGNORECASE)
    name = re.sub(r'(\.ome)\.tiff?$', '', name, flags=re.IGNORECASE)
    return name


# *** CLI and core functions ***

# function to parse command line args
def parse_args():
    p = argparse.ArgumentParser(description='Extract patches from Allen Developing Mouse dataset')

    # input, choose exactly one of --input_tif or --tif_list
    p.add_argument('--input_tif', type=str, default=None, help='Path to single .tif/.tiff/.nii/.nii.gz volume file')
    p.add_argument('--tif_list', type=str, default=None, help='Path to text file with one .tif/.tiff/.nii/.nii.gz path per line')

    # output
    p.add_argument('--output_dir', type=str, required=True, help='Output directory for extracted patches')

    # volume/patch params
    p.add_argument('--patch_size', type=int, default=96, help='Size of the patches to extract, cubic (default: 96)')
    p.add_argument('--num_patches', type=int, default=1, help='Number of patches to extract from each image (default: 1)')
    p.add_argument('--min_fg', type=float, default=0.05, help='Minimum foreground fraction to consider a patch (default: 0.05)')
    p.add_argument('--stride', type=int, default=None, help='Optional stride for sliding window (default: patch_size for non-overlap)')

    # channel selection if applicable
    p.add_argument('--channel', type=int, default=None, help="If tiff is 4d (Z,Y,X,C), specify channel index (0..C-1) to extract. Required if tiff is 4d (default: None)")

    # seed
    p.add_argument('--seed', type=int, default=100, help='Random seed for reproducibility, combined with per-folder hash for determinism (default: 100)')

    # nifti 4d selection if applicable
    p.add_argument('--nifti_index', type=int, default=None, help="If nifti is 4d (X,Y,Z,C), specify index (0..C-1) to extract. Required if nifti is 4d (default: None)")

    return p.parse_args()


# function to extract and save patches from a single tif volume and save as nifti
def extract_and_save_patches_for_file(filepath, output_dir, patch_size, num_patches, min_fg, stride, channel, user_seed, nifti_index):

    # set deterministic seed for filepath
    seed_used = deterministic_seed(os.path.abspath(filepath), user_seed)
    print(f'[INFO] Seed used for {filepath}: {seed_used}', flush=True)

    # read volume
    vol, files = read_single_image(filepath, channel, nifti_index) # (Z,Y,X)
    Z, Y, X = vol.shape
    ps = int(patch_size)
    if Z < ps or Y < ps or X < ps:
        print(f'[WARN] Volume shape ({vol.shape}) smaller than patch size: {ps}, skipping', flush=True)
        return 0
    
    # global otsu threshold
    thresh = compute_global_otsu_threshold(vol)
    print(f'[INFO] Global Otsu threshold for {filepath}: {thresh}', flush=True)

    # reservoir sampling logic
    K = int(num_patches)
    reservoir = [] # each: {'patch': (ps,ps,ps), 'z':..., 'y':..., 'x':..., 'fg':...}
    seen = 0 # number of qualifying candidates seen so far
    st = stride if stride is not None else ps

    for (z, y, x) in get_grid_candidates((Z, Y, X), ps, st):

        patch = vol[z:z+ps, y:y+ps, x:x+ps] # (ps,ps,ps)
        pmin, pmax = float(patch.min()), float(patch.max())
        if pmin == pmax:
            fg = 0.0
        else:
            # guard against trivially dark patches
            fg = float((patch > max(thresh, pmin)).mean())

        # ensure sufficient foreground
        if fg < min_fg:
            continue

        # update reservoir with qualifying candidate
        seen += 1
        if len(reservoir) < K:
            reservoir.append({'patch':patch.copy(), 'z':z, 'y':y, 'x':x, 'fg':fg})
        else:
            r = random.randint(1, seen) # 1..seen (inclusive)
            if r <= K: # replace random slot 1..K
                reservoir[r-1] = {'patch':patch.copy(), 'z':z, 'y':y, 'x':x, 'fg':fg}

    # log info
    print(f'[INFO] Qualifying candidates seen: {seen}. Will save up to {len(reservoir)}.', flush=True)

    if not reservoir:
        print(f'[WARN] No qualifying patches found in {filepath}.', flush=True)
        return 0
    
    # save reservoir patches as nifti
    os.makedirs(output_dir, exist_ok=True)
    base_tag = clean_base_tag(filepath)

    saved = 0
    for i, item in enumerate(reservoir):
        patch = item['patch'].astype(np.uint16) # (ps,ps,ps)
        z, y, x, fg = item['z'], item['y'], item['x'], item['fg']
        data_xyz = np.transpose(patch, (2, 1, 0)) # (Z,Y,X) -> (X,Y,Z) for nifti
        affine = np.eye(4, dtype=np.float32)
        img = nib.Nifti1Image(data_xyz, affine)
        outname = f'{base_tag}_ps{ps}_p{i}_z{z}_y{y}_x{x}.nii.gz'
        outpath = os.path.join(output_dir, outname)
        nib.save(img, outpath)
        print(f'[INFO] Saved patch {i} to {outpath}', flush=True)
        saved += 1

    return saved # number of patches saved


# main
def main():

    # parse args
    args = parse_args()

    # input mode
    if bool(args.input_tif) == bool(args.tif_list):
        print(f'[ERROR] Please specify exactly one of --input_tif or --tif_list', flush=True)
        sys.exit(1)

    if args.input_tif:
        if not os.path.isfile(args.input_tif):
            print(f'[ERROR] --input_tif "{args.input_tif}" is not a valid file', flush=True)
            sys.exit(1)
        if not (_is_tif(args.input_tif) or _is_nifti(args.input_tif)):
            print(f'[ERROR] --input_tif "{args.input_tif}" is not a tif or nifti file', flush=True)
            sys.exit(1)
        files = [args.input_tif]
    else:
        if not os.path.isfile(args.tif_list):
            print(f'[ERROR] --tif_list "{args.tif_list}" is not a valid file', flush=True)
            sys.exit(1)
        files = load_path_list(args.tif_list)

    # print info
    print(f'[INFO] Output dir: {args.output_dir}', flush=True)
    print(f'[INFO] Patch size: {args.patch_size}', flush=True)
    print(f'[INFO] Number of patches: {args.num_patches}', flush=True)
    print(f'[INFO] Minimum foreground fraction: {args.min_fg}', flush=True)
    if args.channel is not None:
        print(f'[INFO] Extracting channel: {args.channel}', flush=True)
    if args.nifti_index is not None:
        print(f'[INFO] Extracting nifti index: {args.nifti_index}', flush=True)

    # process each file
    total_saved = 0
    total_files = 0

    for f in files:
        try:
            saved = extract_and_save_patches_for_file(
                filepath=f,
                output_dir=args.output_dir,
                patch_size=args.patch_size,
                num_patches=args.num_patches,
                min_fg=args.min_fg,
                stride=args.stride,
                channel=args.channel,
                nifti_index=args.nifti_index,
                user_seed=args.seed 
            )
            total_saved += saved
            total_files += 1
        except Exception as e:
            print(f'[ERROR] Processing "{f}": {e}', flush=True)

    # print summary
    print(f'[INFO] Processed {total_files} files, saved {total_saved} patches.', flush=True)



# --- Entry point ---
if __name__ == '__main__':
    main()





