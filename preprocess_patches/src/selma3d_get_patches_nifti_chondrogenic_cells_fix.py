#!/usr/bin/env python3
# selma3d_get_patches_nifti.py
# Robust patch extractor for 2D .tif stacks → random 96^3 patches (.nii.gz or .tiff)
# - Single-threaded TIFF decode (avoids flaky threaded decode issues)
# - Logs metadata for unreadable slices
# - Interpolates (or zero-fills) unreadable slices so one bad file won't crash
# - Supports limiting to a single structure via --only_structure
# - Preserves vessel C00/C01 sync logic

import argparse
import hashlib
import nibabel as nib
import numpy as np
import os
import random
import re
from skimage.filters import threshold_otsu
import tifffile as tiff

# --- Defaults (can be overridden by CLI) ---
PATCH_SIZE = 96
MIN_FOREGROUND_FRACTION = 0.05
NUM_RANDOM_PATCHES = 10
SAVE_AS_NIFTI = True
SKIP_BORDER_XY_TILES = 1     # skip outermost tiles in X/Y
SKIP_Z_BORDER_BLOCKS = 1     # skip first/last z blocks (each block is 96 slices)
SYNC_VESSEL_CHANNELS = True
ROOT_DIR = '/midtier/paetzollab/scratch/ads4015/data_selma3d'
GLOBAL_SEED = None

# Map folder → (human name, prefix)
structures = {
    'unannotated_ab_plaque': ('Ab_plaques', 'ab_plaque'),
    'unannotated_cfos': ('c-Fos_brain_cells', 'cfos'),
    'unannotated_chondrocytes': ('chondrocytes', 'chondrocytes'),
    'unannotated_chondrogenic_cells': ('chondrogenic_cells', 'chondrogenic_cells'),
    'unannotated_nucleus': ('cell_nucleus', 'nucleus'),
    'unannotated_vessel': ('vessel', 'vessel'),
}

# Vessel channels
channels = {
    'vessel': {'C00': 'vessel_wga', 'C01': 'vessel_eb'}
}

# -----------------------
# Safe TIFF IO utilities
# -----------------------
def safe_imread(path, *, maxworkers=1):
    """Read a TIFF slice safely with single-threaded decode."""
    try:
        return tiff.imread(path, maxworkers=maxworkers)
    except Exception:
        with tiff.TiffFile(path) as tf:
            return tf.pages[0].asarray(maxworkers=1)

def try_imread_or_none(path, *, maxworkers=1):
    try:
        return tiff.imread(path, maxworkers=maxworkers)
    except Exception:
        try:
            with tiff.TiffFile(path) as tf:
                return tf.pages[0].asarray(maxworkers=1)
        except Exception:
            return None

def inspect_tiff(path):
    try:
        with tiff.TiffFile(path) as tf:
            p = tf.pages[0]
            return dict(
                shape=getattr(p, "shape", None),
                dtype=str(getattr(p, "dtype", None)),
                compression=str(getattr(p, "compression", None)),
                rows_per_strip=getattr(p, "rowsperstrip", None),
                samples_per_pixel=getattr(p, "samplesperpixel", None),
                tiled=getattr(p, "is_tiled", None),
            )
    except Exception as e:
        return {"error": repr(e)}

# -----------------------
# Helpers
# -----------------------
def pad_slice(slice_array, target_shape, verbose=False):
    if verbose:
        print('Padding slice...', flush=True)
    pad_y = max(0, target_shape[0] - slice_array.shape[0])
    pad_x = max(0, target_shape[1] - slice_array.shape[1])
    if pad_y or pad_x:
        return np.pad(slice_array, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
    return slice_array

def _num_key(s):
    m = re.search(r'_(\d+)\.(?:tif|tiff)$', s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r'\d+', s)
    return int(nums[-1]) if nums else -1

def block_otsu(arr: np.ndarray):
    if arr is None:
        return 0
    ds = arr[::4, ::4, ::4]
    return threshold_otsu(ds) if ds.min() != ds.max() else ds.min()

def _repair_block(slice_list, shape):
    """Replace None entries by avg of neighbors, else zeros."""
    for k in range(len(slice_list)):
        if slice_list[k] is None:
            prev_arr = next_arr = None
            i = k - 1
            while i >= 0 and slice_list[i] is None:
                i -= 1
            if i >= 0:
                prev_arr = slice_list[i]
            j = k + 1
            while j < len(slice_list) and slice_list[j] is None:
                j += 1
            if j < len(slice_list):
                next_arr = slice_list[j]
            if prev_arr is not None and next_arr is not None and prev_arr.shape == next_arr.shape:
                repaired = ((prev_arr.astype(np.float32) + next_arr.astype(np.float32)) / 2).astype(prev_arr.dtype)
            elif prev_arr is not None:
                repaired = prev_arr
            elif next_arr is not None:
                repaired = next_arr
            else:
                repaired = np.zeros(shape, dtype=np.uint16)
            slice_list[k] = repaired
    return slice_list

# -----------------------
# Core extraction
# -----------------------
def extract_patches_from_stack(tiff_dir, output_dir, prefix):
    print(f'Processing: {prefix} ({tiff_dir})', flush=True)

    slice_files = sorted(
        [f for f in os.listdir(tiff_dir) if f.lower().endswith('.tif') and not f.startswith('.')],
        key=_num_key
    )
    if len(slice_files) == 0:
        print('No .tif slices found. Skipping.', flush=True)
        return

    os.makedirs(output_dir, exist_ok=True)

    # vessel pairing
    sibling_dir = None
    use_pair = False
    base_dir = os.path.dirname(tiff_dir)
    current_channel = os.path.basename(tiff_dir)

    if SYNC_VESSEL_CHANNELS and prefix.startswith('vessel_'):
        sibling_channel = 'C01' if current_channel == 'C00' else 'C00'
        candidate = os.path.join(base_dir, sibling_channel)
        if os.path.isdir(candidate):
            sibling_dir = candidate
            use_pair = True

    # Determine padded shapes and z padding
    if use_pair:
        slice_files_b = sorted(
            [f for f in os.listdir(sibling_dir) if f.lower().endswith('.tif') and not f.startswith('.')],
            key=_num_key
        )
        sample_slice_a = safe_imread(os.path.join(tiff_dir, slice_files[0]))
        sample_slice_b = safe_imread(os.path.join(sibling_dir, slice_files_b[0]))
        ha, wa = sample_slice_a.shape
        hb, wb = sample_slice_b.shape
        height, width = max(ha, hb), max(wa, wb)
        pad_y = (PATCH_SIZE - height % PATCH_SIZE) % PATCH_SIZE
        pad_x = (PATCH_SIZE - width % PATCH_SIZE) % PATCH_SIZE
        padded_shape = (height + pad_y, width + pad_x)
        z_len = max(len(slice_files), len(slice_files_b))
        pad_z = (PATCH_SIZE - z_len % PATCH_SIZE) % PATCH_SIZE
        total_slices = z_len + pad_z
    else:
        sample_slice = safe_imread(os.path.join(tiff_dir, slice_files[0]))
        height, width = sample_slice.shape
        pad_y = (PATCH_SIZE - height % PATCH_SIZE) % PATCH_SIZE
        pad_x = (PATCH_SIZE - width % PATCH_SIZE) % PATCH_SIZE
        padded_shape = (height + pad_y, width + pad_x)
        pad_z = (PATCH_SIZE - len(slice_files) % PATCH_SIZE) % PATCH_SIZE
        total_slices = len(slice_files) + pad_z

    print(f'Height, width: ({height}, {width}); padded: {padded_shape}', flush=True)
    print(f'z padding: {pad_z}, total slices: {total_slices}', flush=True)

    # datatype from prefix
    if prefix.startswith('ab_plaque'):
        datatype = 'ab_plaque'
    elif prefix.startswith('cfos'):
        datatype = 'cfos'
    elif prefix.startswith('chondrocytes'):
        datatype = 'chondrocytes'
    elif prefix.startswith('chondrogenic_cells'):
        datatype = 'chondrogenic_cells'
    elif prefix.startswith('nucleus'):
        datatype = 'nucleus'
    elif prefix.startswith('vessel_eb'):
        datatype = 'vessel_eb'
    elif prefix.startswith('vessel_wga'):
        datatype = 'vessel_wga'
    else:
        datatype = prefix.split('_')[0]

    sample_name = prefix.replace(datatype + '_', '')
    subfolder = os.path.join(output_dir, datatype)
    os.makedirs(subfolder, exist_ok=True)
    channel = os.path.basename(tiff_dir)

    # deterministic seed for vessel pairs
    if use_pair:
        sample_id = os.path.basename(base_dir)
        seed_base = int(hashlib.md5(sample_id.encode('utf-8')).hexdigest()[:8], 16)
        seed_final = seed_base if GLOBAL_SEED is None else ((seed_base ^ GLOBAL_SEED) & 0xFFFFFFFF)
        random.seed(seed_final)
        np.random.seed(seed_final)

    # reservoir selection
    selected = []
    candidates_seen = 0

    num_blocks = total_slices // PATCH_SIZE
    if 2 * SKIP_Z_BORDER_BLOCKS >= num_blocks:
        print(f'[WARNING] skip_z_border_blocks={SKIP_Z_BORDER_BLOCKS} removes all z blocks (only {num_blocks} available). Lower the value.', flush=True)

    start_z = SKIP_Z_BORDER_BLOCKS * PATCH_SIZE
    end_z = total_slices - SKIP_Z_BORDER_BLOCKS * PATCH_SIZE

    # Iterate over z blocks
    for z0 in range(start_z, end_z, PATCH_SIZE):

        # --- Channel A block ---
        slices_a = []
        for zi in range(z0, z0 + PATCH_SIZE):
            if zi < len(slice_files):
                fname = os.path.join(tiff_dir, slice_files[zi])
                sdata = try_imread_or_none(fname)
                if sdata is None:
                    meta = inspect_tiff(fname)
                    print(f"[WARN] Failed to read {fname}; meta={meta}. Will repair later.", flush=True)
                else:
                    sdata = pad_slice(sdata, padded_shape)
            else:
                sdata = np.zeros(padded_shape, dtype=np.uint16)
            slices_a.append(sdata)

        # --- Channel B block (vessels) ---
        if use_pair:
            slices_b = []
            for zi in range(z0, z0 + PATCH_SIZE):
                if zi < len(slice_files_b):
                    fname_b = os.path.join(sibling_dir, slice_files_b[zi])
                    sdata = try_imread_or_none(fname_b)
                    if sdata is None:
                        meta = inspect_tiff(fname_b)
                        print(f"[WARN] Failed to read {fname_b}; meta={meta}. Will repair later.", flush=True)
                    else:
                        sdata = pad_slice(sdata, padded_shape)
                else:
                    sdata = np.zeros(padded_shape, dtype=np.uint16)
                slices_b.append(sdata)
        else:
            slices_b = None

        # repair & stack
        slices_a = _repair_block(slices_a, padded_shape)
        volume = np.stack(slices_a)

        if use_pair:
            slices_b = _repair_block(slices_b, padded_shape)
            volume_b = np.stack(slices_b)
        else:
            volume_b = None

        # Thresholds (block-level)
        print('Thresholding...', flush=True)
        threshold_a = block_otsu(volume)
        threshold_b = block_otsu(volume_b) if volume_b is not None else None

        # XY borders to skip
        y_start = SKIP_BORDER_XY_TILES * PATCH_SIZE
        y_stop = volume.shape[1] - SKIP_BORDER_XY_TILES * PATCH_SIZE
        x_start = SKIP_BORDER_XY_TILES * PATCH_SIZE
        x_stop = volume.shape[2] - SKIP_BORDER_XY_TILES * PATCH_SIZE

        # scan non-overlapping tiles
        for y in range(y_start, y_stop, PATCH_SIZE):
            for x in range(x_start, x_stop, PATCH_SIZE):
                patch = volume[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                if patch.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
                    continue

                if volume_b is None:
                    ds_patch = patch[::4, ::4, ::4]
                    thr_local = threshold_otsu(ds_patch) if ds_patch.min() != ds_patch.max() else ds_patch.min()
                    thr_eff = max(threshold_a, thr_local)
                    fg_fraction = (patch > thr_eff).mean()
                    keep = fg_fraction >= MIN_FOREGROUND_FRACTION
                else:
                    patch_b = volume_b[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    if patch_b.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
                        continue
                    ds_a = patch[::4, ::4, ::4]
                    ds_b = patch_b[::4, ::4, ::4]
                    thr_loc_a = threshold_otsu(ds_a) if ds_a.min() != ds_a.max() else ds_a.min()
                    thr_loc_b = threshold_otsu(ds_b) if ds_b.min() != ds_b.max() else ds_b.min()
                    thr_eff_a = max(threshold_a, thr_loc_a)
                    thr_eff_b = max(threshold_b, thr_loc_b)
                    fg_a = (patch > thr_eff_a).mean()
                    fg_b = (patch_b > thr_eff_b).mean()
                    keep = (fg_a >= MIN_FOREGROUND_FRACTION) or (fg_b >= MIN_FOREGROUND_FRACTION)

                if keep:
                    candidates_seen += 1
                    if len(selected) < NUM_RANDOM_PATCHES:
                        selected.append({'patch': patch.copy(), 'idx': candidates_seen-1, 'pos': (z0, y, x)})
                    else:
                        r = random.randint(0, candidates_seen - 1)
                        if r < NUM_RANDOM_PATCHES:
                            selected[r] = {'patch': patch.copy(), 'idx': candidates_seen-1, 'pos': (z0, y, x)}

    # Save selected patches
    to_save = min(NUM_RANDOM_PATCHES, len(selected))
    print(f'Foreground-qualified candidates found: {candidates_seen}. Will save up to {to_save}.', flush=True)

    for saved_count, item in enumerate(selected[:to_save]):
        patch = item['patch']
        cand_idx = item.get('idx')
        z0, y0, x0 = item.get('pos', (0, 0, 0))

        if SAVE_AS_NIFTI:
            # transpose (z, y, x) -> (x, y, z) for NIfTI
            patch_nifti = np.transpose(patch.astype(np.uint16), (2, 1, 0))
            patch_path = os.path.join(subfolder, f'{datatype}_{sample_name}_{channel}_p{saved_count}_cand{cand_idx}_z{z0}_y{y0}_x{x0}.nii.gz')
            nib.save(nib.Nifti1Image(patch_nifti, affine=np.eye(4)), patch_path)
        else:
            patch_path = os.path.join(subfolder, f'{datatype}_{sample_name}_{channel}_p{saved_count}_cand{cand_idx}_z{z0}_y{y0}_x{x0}.tiff')
            tiff.imwrite(patch_path, patch.astype(np.uint16), imagej=True)

        print(f'Saved random patch {saved_count} (cand {cand_idx} @ z{z0},y{y0},x{x0}) -> {patch_path}', flush=True)

    print(f'Done. Saved {to_save} patches.', flush=True)

# -----------------------
# Discovery
# -----------------------
def get_all_sample_dirs(only_structure=None):
    """Return list of (tiff_dir, prefix) pairs. Can filter to a single structure."""
    samples = []

    def has_tifs(d):
        try:
            return any(f.lower().endswith('.tif') and not f.startswith('.') for f in os.listdir(d))
        except Exception:
            return False

    for structure_key, (_human, prefix_name) in structures.items():
        # filter to one structure if requested
        if only_structure is not None and structure_key != only_structure:
            continue

        base_path = os.path.join(ROOT_DIR, structure_key)
        print('base_path:', base_path, flush=True)
        if not os.path.exists(base_path):
            continue

        for sample in os.listdir(base_path):
            sample_path = os.path.join(base_path, sample)
            if not os.path.isdir(sample_path):
                continue

            if structure_key == 'unannotated_vessel':
                for channel_name, channel_prefix in channels['vessel'].items():
                    channel_path = os.path.join(sample_path, channel_name)
                    if os.path.isdir(channel_path) and has_tifs(channel_path):
                        prefix = f'{channel_prefix}_{sample}'
                        samples.append((channel_path, prefix))
            else:
                for subdir in os.listdir(sample_path):
                    tiff_dir = os.path.join(sample_path, subdir)
                    if os.path.isdir(tiff_dir) and has_tifs(tiff_dir):
                        prefix = f'{prefix_name}_{sample}'
                        samples.append((tiff_dir, prefix))

    print(f'Discovered {len(samples)} stack(s).', flush=True)
    return samples

def process_single(tiff_dir, prefix, output_dir):
    extract_patches_from_stack(tiff_dir, output_dir, prefix)

# -----------------------
# Main
# -----------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_dir', type=str, help='Path to directory with tif slices')
    parser.add_argument('--prefix', type=str, help='Prefix for filenames of saved patches')
    parser.add_argument('--output_dir', type=str, required=False, help='Directory to save extracted patches')
    parser.add_argument('--sample_index', type=int, help='Index for SLURM array job')
    parser.add_argument('--num_patches', type=int, default=10, help='Max random patches per image (default 10)')
    parser.add_argument('--min_fg', type=float, default=0.05, help='Minimum required foreground fraction (default 0.05)')
    parser.add_argument('--seed', type=int, default=100, help='Random seed for reproducible sampling.')
    parser.add_argument(
        '--only_structure',
        type=str,
        choices=[
            'unannotated_ab_plaque',
            'unannotated_cfos',
            'unannotated_chondrocytes',
            'unannotated_chondrogenic_cells',
            'unannotated_nucleus',
            'unannotated_vessel',
        ],
        help='Limit processing to a single top-level structure folder.'
    )
    args = parser.parse_args()

    # overrides
    if args.num_patches is not None:
        NUM_RANDOM_PATCHES = args.num_patches
    if args.min_fg is not None:
        MIN_FOREGROUND_FRACTION = args.min_fg
    if args.output_dir is None:
        raise SystemExit("Please provide --output_dir")

    # seed
    GLOBAL_SEED = args.seed
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    print(f'Using num_patches={NUM_RANDOM_PATCHES}, min_fg={MIN_FOREGROUND_FRACTION}, seed={GLOBAL_SEED}, output_dir={args.output_dir}', flush=True)

    if args.sample_index is not None:
        all_samples = get_all_sample_dirs(only_structure=args.only_structure)
        if not all_samples:
            print(f'No tiff stacks found under {ROOT_DIR} (filter={args.only_structure}).', flush=True)
            raise SystemExit(1)
        if 0 <= args.sample_index < len(all_samples):
            input_dir, prefix = all_samples[args.sample_index]
            os.makedirs(args.output_dir, exist_ok=True)
            process_single(input_dir, prefix, args.output_dir)
        else:
            print(f'Invalid sample index {args.sample_index}. Range: 0 to {len(all_samples)-1}')
            raise SystemExit(1)
    elif args.single_dir and args.prefix:
        os.makedirs(args.output_dir, exist_ok=True)
        process_single(args.single_dir, args.prefix, args.output_dir)
    else:
        raise SystemExit('Specify --sample_index or (--single_dir and --prefix).')
