# selma3d_get_patches.py - Script to Extract Smaller Patches from 2D .tif Stacks and Create .tiff volumes

# --- Setup ---

# imports
import argparse
import hashlib
import nibabel as nib
import numpy as np
import os
import random
import re
from skimage.filters import threshold_otsu
import tifffile as tiff


# define constants
PATCH_SIZE = 96
# STRIDE = 96
MIN_FOREGROUND_FRACTION = 0.05
NUM_RANDOM_PATCHES = 10
SAVE_AS_NIFTI = True
SKIP_BORDER_XY_TILES = 1 # set to 1 to skip outermost tiles in x and y, set to 0 otherwise
SKIP_Z_BORDER_BLOCKS = 1 # set to 1 to skip first and last z blocks, set to 0 otherwise
SYNC_VESSEL_CHANNELS = True
ROOT_DIR = '/midtier/paetzollab/scratch/ads4015/data_selma3d'  # base path to selma3d data
GLOBAL_SEED = None

# create dictionary mapping subfolder names to structure names and prefixes ## UP TO HERE - uncomment structures
structures = {
    # 'unannotated_ab_plaque': ('ab_plaque'),
    # 'unannotated_cfos': ('cfos'),
    # 'unannotated_chondrocytes': ('chondrocytes'),
    # 'unannotated_chondrogenic_cells': ('chondrogenic_cells'),
    # 'unannotated_nucleus': ('nucleus'),
    # 'unannotated_vessel': ('vessel'),
    'unannotated_astrocytes': ('astrocytes'),
    'unannotated_beta3tubulin_cranial_nerve': ('beta3tubulin_nerve'),
    'unannotated_ctip2_neurons': ('ctip2_neurons'),
    'unannotated_lyve1_lymphatic_vessel': ('lyve1_vessel'),
    'unannotated_nf_peripheral_nerve': ('nf_nerve'),
    'unannotated_p75_nerves': ('p75_nerves'),
    'unannotated_pgp_peripheral_nerve': ('pgp_nerve'),
    'unannotated_sma_artery': ('sma_artery'),
    'unannotated_th_neurons': ('th_neurons'),
    'unannotated_th_sympathetic_nerve': ('th_nerve')
}

# define specific channels for vessel images
channels = {'vessel': {'C00': 'vessel_wga',
                        'C01': 'vessel_eb'}}


# --- Functions ---

# robust tiff helper functions (to find and fix bad slices)

# function to safely read tiff stack single-threaded
def safe_imread(path, *, maxworkers=1):

    try:
        return tiff.imread(path, maxworkers=maxworkers)
    except Exception:
        with tiff.TiffFile(path) as tf:
            return tf.pages[0].asarray(maxworkers=maxworkers)
        

# function to try reading tiff stack and return None on failure
def try_imread_or_none(path, *, maxworkers=1):

    try:
        return tiff.imread(path, maxworkers=maxworkers)
    except Exception:
        try:
            with tiff.TiffFile(path) as tf:
                return tf.pages[0].asarray(maxworkers=maxworkers)
        except Exception:
            return None
        

# function to inspect tiff metadata
def inspect_tiff(path):

    try:
        with tiff.TiffFile(path) as tf:
            p = tf.pages[0]
            return dict(
                shape=getattr(p, 'shape', None),
                dtype=str(getattr(p, 'dtype', None)),
                compression=str(getattr(p, 'compression', None)),
                rows_per_strip=getattr(p, 'rowsperstrip', None),
                samples_per_pixel=getattr(p, 'samplesperpixel', None),
                tiled=getattr(p, 'is_tiled', None)
            )
    except Exception as e:
        return {'error': repr(e)}
    

# function to replace none entries with average of nearest neighbors, else zeros
def _repair_block(slice_list, shape):

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

            # replace corrupted slice with average of nearest neighbors
            if prev_arr is not None and next_arr is not None and prev_arr.shape == next_arr.shape:
                repaired = ((prev_arr.astype(np.float32) + next_arr.astype(np.float32)) / 2).astype(prev_arr.dtype)
                print(f'Repaired missing slice {k} using average of neighbors {i} and {j}.', flush=True)

            # if only have prev (not next), replace corrupted slice with prev
            elif prev_arr is not None:
                repaired = prev_arr
                print(f'Repaired missing slice {k} using previous neighbor {i}.', flush=True)

            # if only have next (not prev), replace corrupted slice with next
            elif next_arr is not None:
                repaired = next_arr
                print(f'Repaired missing slice {k} using next neighbor {j}.', flush=True)

            # if don't have prev or next, replace missing slice with zeros
            else:
                repaired = np.zeros(shape, dtype=np.uint16)

            # add slice back into array
            slice_list[k] = repaired

    return slice_list

# ---

# function to pad slice to correct x,y dimensions
def pad_slice(slice_array, target_shape, verbose=False):
    if verbose:
        print('Padding slice...', flush=True)
    pad_y = max(0, target_shape[0] - slice_array.shape[0])
    pad_x = max(0, target_shape[1] - slice_array.shape[1])
    if pad_y or pad_x:
        return np.pad(slice_array, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0)
    return slice_array


# function to sort filenames numerically instead of alphabetically
def _num_key(s):
    m = re.search(r'_(\d+)\.(?:tif|tiff)$', s, flags=re.IGNORECASE)
    if m:
        return int(m.group(1))
    nums = re.findall(r'\d+', s)
    return int(nums[-1]) if nums else -1


# function to get random seed
def seed_rng_for_sample(sample_id):

    # deterministically seed rngs for given sample id
    seed_base = int(hashlib.md5(sample_id.encode('utf-8')).hexdigest()[:8], 16)
    seed_final = seed_base if GLOBAL_SEED is None else ((seed_base ^ GLOBAL_SEED) & 0xFFFFFFFF)
    random.seed(seed_final)
    np.random.seed(seed_final)
    print(f'[DEBUG] Seed RNG for sample {sample_id}: {seed_final}', flush=True)


# function to get sorted slices and ensure correct ordering
def get_sorted_slices(tiff_dir, label=''):

    # get slices
    slice_files = sorted([f for f in os.listdir(tiff_dir) if f.lower().endswith(('.tif', '.tiff')) and not f.startswith('.')], 
                         key=_num_key)
    
    # ensure correct sorting
    print(f'[DEBUG] Sorted {len(slice_files)} slices in {tiff_dir} {label}.', flush=True)

    return slice_files


# function to confirm that tiff stack is sorted correctly
def preview_stack_list(tiff_dir, n=10, slice_files=None):

    if slice_files is None:
        slice_files = get_sorted_slices(tiff_dir)
    if not slice_files:
        print(f'[WARNING] No valid slice files found in {tiff_dir}.', flush=True)
        return

    # get sorted slice files
    slice_files = get_sorted_slices(tiff_dir)

    # preview first n slices
    for f in slice_files[:n]:
        print(f'  {f}', flush=True)
    for f in slice_files[-n:]:
        print(f'  {f}', flush=True)


# block-level otsu thresholding
def block_otsu(arr: np.ndarray):
    if arr is None:
        return 0
    ds = arr[::4, ::4, ::4]
    return threshold_otsu(ds) if ds.min() != ds.max() else ds.min()


# function to extract patches from directory of 2d tiff slices
def extract_patches_from_stack(tiff_dir, output_dir, prefix):

    print(f'Processing: {prefix} ({tiff_dir})', flush=True)

    # define channel
    base_dir = os.path.dirname(tiff_dir)
    current_channel = os.path.basename(tiff_dir)

    # reseed deterministically 
    sample_id = os.path.basename(base_dir)
    seed_rng_for_sample(sample_id)

    # get slices from primary channel (all data and vessels)
    slice_files = get_sorted_slices(tiff_dir, label=f'(primary {current_channel})')
    preview_stack_list(tiff_dir, n=10, slice_files=slice_files)

    # ensure tif slices exist
    if len(slice_files) == 0:
        print('No .tif slices found. Skipping.', flush=True)
        return
    
    # create output directory
    os.makedirs(output_dir, exist_ok=True)


    # compute padding and seeding based on both channels when syncing vessels
    # image will be kept if either channel passes threshold requirements
    sibling_dir = None
    use_pair = False # will be set to True for vessel images only

    if SYNC_VESSEL_CHANNELS and prefix.startswith('vessel_'):
        sibling_channel = 'C01' if current_channel == 'C00' else 'C00'
        candidate = os.path.join(base_dir, sibling_channel)
        if os.path.isdir(candidate):
            sibling_dir = candidate
            use_pair = True
            slice_files_b = get_sorted_slices(sibling_dir, label=f'(sibling {sibling_channel})')

    if use_pair:
        sample_slice_a = safe_imread(os.path.join(tiff_dir, slice_files[0]))
        sample_slice_b = safe_imread(os.path.join(sibling_dir, slice_files_b[0]))
        ha, wa = sample_slice_a.shape
        hb, wb = sample_slice_b.shape
        height, width = max(ha, hb), max(wa, wb)
        pad_y = (PATCH_SIZE - height % PATCH_SIZE) % PATCH_SIZE
        pad_x = (PATCH_SIZE - width % PATCH_SIZE) % PATCH_SIZE
        padded_shape = (height + pad_y, width + pad_x)

        # determine z padding based on longer channel
        z_len = max(len(slice_files), len(slice_files_b))
        pad_z = (PATCH_SIZE - z_len % PATCH_SIZE) % PATCH_SIZE
        total_slices = z_len + pad_z

    # single-channel padding logic
    else:

        # determine max dimensions of slices to define padding target for slice
        sample_slice = safe_imread(os.path.join(tiff_dir, slice_files[0]))
        height, width = sample_slice.shape
        pad_y = (PATCH_SIZE - height % PATCH_SIZE) % PATCH_SIZE
        pad_x = (PATCH_SIZE - width % PATCH_SIZE) % PATCH_SIZE
        padded_shape = (height + pad_y, width + pad_x)

        # pad z if necessary
        pad_z = (PATCH_SIZE - len(slice_files) % PATCH_SIZE) % PATCH_SIZE
        total_slices = len(slice_files) + pad_z

    print(f'Height, width: ({height}, {width}); padded: {padded_shape}', flush=True)
    print(f'z padding: {pad_z}, total slices: {total_slices}', flush=True)

    # extract datatype and sample name from prefix
    datatype, sample_name = prefix.rsplit('_', 1)

    subfolder = os.path.join(output_dir, datatype)
    os.makedirs(subfolder, exist_ok=True)
    channel = os.path.basename(tiff_dir)

    # reservoir sampling to keep at most NUM_RANDOM_PATCHES patches in memory
    # each item is dict with keys {'patch': np.ndarray, 'idx': int}
    selected = []
    candidates_seen = 0

    # ensure that not all z blocks are being skipped
    num_blocks = total_slices // PATCH_SIZE
    if 2 * SKIP_Z_BORDER_BLOCKS >= num_blocks:
        print(f'[WARNING] skip_z_border_blocks={SKIP_Z_BORDER_BLOCKS} removes all z blocks (only {num_blocks} available). Lower the value.', flush=True)

    # set start and end in z direction
    start_z = SKIP_Z_BORDER_BLOCKS * PATCH_SIZE
    end_z = total_slices - SKIP_Z_BORDER_BLOCKS * PATCH_SIZE

    # iterate over z blocks of 96 slices
    for z0 in range(start_z, end_z, PATCH_SIZE):

        # build a 96-slice padded volume block
        slices_a = []
        for zi in range(z0, z0 + PATCH_SIZE):
            if zi < len(slice_files):
                fname = os.path.join(tiff_dir, slice_files[zi])
                sdata = try_imread_or_none(fname)
                if sdata is None:
                    meta = inspect_tiff(fname)
                    print(f'[WARN] Failed to read {fname}; meta={meta}. Will repair later.', flush=True)
                else:
                    sdata = pad_slice(sdata, padded_shape)
            else:
                sdata = np.zeros(padded_shape, dtype=np.uint16)
            slices_a.append(sdata)

        

        # build sibling block volume for vessels
        # volume: np.ndarray (z, y, x) for current 96 slice block
        # volume_b: optional paired vessel channel (z, y, x) or None
        if use_pair:
            slices_b = []
            for zi in range(z0, z0+PATCH_SIZE):
                if zi < len(slice_files_b):
                    fname_b = os.path.join(sibling_dir, slice_files_b[zi])
                    sdata = try_imread_or_none(fname_b)
                    if sdata is None:
                        meta = inspect_tiff(fname_b)
                        print(f'[WARN] Failed to read {fname_b}; meta={meta}. Will repair later.', flush=True)
                    else:
                        sdata = pad_slice(sdata, padded_shape)
                else:
                    sdata = np.zeros(padded_shape, dtype=np.uint16)
                slices_b.append(sdata)
        else:
            volume_b = None


        # repair and stack a and b
        slices_a = _repair_block(slices_a, padded_shape)
        volume = np.stack(slices_a)
        del slices_a # delete to save memory

        if use_pair:
            slices_b = _repair_block(slices_b, padded_shape)
            volume_b = np.stack(slices_b)
            del slices_b # delete to save memroy
        else:
            volume_b = None


        # thresholding
        print('Thresholding...', flush=True)
        threshold_a = block_otsu(volume)
        threshold_b = block_otsu(volume_b) if volume_b is not None else None

        
        # get start and stop positions for x and y (want to skip outermost patches because images not good there)
        y_start = SKIP_BORDER_XY_TILES * PATCH_SIZE
        y_stop = volume.shape[1] - SKIP_BORDER_XY_TILES * PATCH_SIZE
        x_start = SKIP_BORDER_XY_TILES * PATCH_SIZE
        x_stop = volume.shape[2] - SKIP_BORDER_XY_TILES * PATCH_SIZE

        # scan non-overlapping (y, x) windows
        for y in range(y_start, y_stop, PATCH_SIZE):
            for x in range(x_start, x_stop, PATCH_SIZE):
                patch = volume[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                # ensure correct size 
                if patch.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
                    continue

                # single channel with local/global otsu thresholding (effective threshold is max of block and local)
                if volume_b is None:
                    ds_patch = patch[::4, ::4, ::4]
                    threshold_local = threshold_otsu(ds_patch) if ds_patch.min() != ds_patch.max() else ds_patch.min()
                    threshold_effective = max(threshold_a, threshold_local)

                    # ensure sufficient foreground
                    fg_fraction = (patch > threshold_effective).mean()
                    keep = fg_fraction >= MIN_FOREGROUND_FRACTION

                # for vessels (2 channels), keep patch if either channel passes threshold
                else:
                    patch_b = volume_b[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                    if patch_b.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
                        continue

                    ds_a = patch[::4, ::4, ::4]
                    ds_b = patch_b[::4, ::4, ::4]
                    threshold_local_a = threshold_otsu(ds_a) if ds_a.min() != ds_a.max() else ds_a.min()
                    threshold_local_b = threshold_otsu(ds_b) if ds_b.min() != ds_b.max() else ds_b.min()
                    threshold_effective_a = max(threshold_a, threshold_local_a)
                    threshold_effective_b = max(threshold_b, threshold_local_b)

                    fg_a = (patch > threshold_effective_a).mean()
                    fg_b = (patch_b > threshold_effective_b).mean()
                    keep = (fg_a >= MIN_FOREGROUND_FRACTION) or (fg_b >= MIN_FOREGROUND_FRACTION)


                # if keeping this patch
                if keep:

                    candidates_seen += 1

                    # reservoir sampling to keep a uniform random sample of size K
                    if len(selected) < NUM_RANDOM_PATCHES:
                        selected.append({'patch': patch.copy(), 'idx': candidates_seen-1, 'pos': (z0, y, x)})
                    else:
                        r = random.randint(0, candidates_seen - 1)
                        if r < NUM_RANDOM_PATCHES:
                            selected[r] = {'patch': patch.copy(), 'idx': candidates_seen-1, 'pos': (z0, y, x)}


    # write selected patches
    to_save = min(NUM_RANDOM_PATCHES, len(selected))
    print(f'Foreground-qualified candidates found: {candidates_seen}. Will save up to {to_save}.', flush=True)

    for saved_count, item in enumerate(selected[:to_save]):

        patch = item['patch']
        cand_idx = item.get('idx')
        z0, y0, x0 = item.get('pos', (0, 0, 0))

        # save as nifti
        if SAVE_AS_NIFTI:

            # transpose (z, y, x) -> (x, y, z) for nifti
            patch_nifti = np.transpose(patch.astype(np.uint16), (2, 1, 0))
            patch_path = os.path.join(subfolder, f'{datatype}_{sample_name}_{channel}_ps{PATCH_SIZE}_p{saved_count}_cand{cand_idx}_z{z0}_y{y0}_x{x0}.nii.gz')

            # save patch using nibabel
            nib.save(nib.Nifti1Image(patch_nifti, affine=np.eye(4)), patch_path)
        
        # save as tiff
        else:
            patch_path = os.path.join(subfolder, f'{datatype}_{sample_name}_{channel}_ps{PATCH_SIZE}_p{saved_count}_cand{cand_idx}_z{z0}_y{y0}_x{x0}.tiff')
            tiff.imwrite(patch_path, patch.astype(np.uint16), imagej=True)

        print(f'Saved random patch {saved_count} (cand {cand_idx} @ z{z0},y{y0},x{x0}) -> {patch_path}', flush=True)

    print(f'Done. Saved {to_save} patches.', flush=True)


# function to get all subdirectories
# based on flat layout. Ex: /midtier/.../data_selma3d/unannotated_*/<sample>/<[C01|C00]>
def get_all_sample_dirs(only_structure=None):  

    # define list for samples
    samples = []

    # function to check if there are tif files in a folder
    def has_tifs(d):
        try:
            return any(f.lower().endswith(('.tif', '.tiff')) and not f.startswith('.') for f in os.listdir(d))
        except Exception:
            return False

    # get base path
    for structure_key, prefix_name in structures.items():
        if only_structure is not None and structure_key != only_structure:
            continue
        base_path = os.path.join(ROOT_DIR, structure_key)
        print('base_path:', base_path, flush=True)
        if not os.path.exists(base_path):
            continue

        # get sample
        for sample in os.listdir(base_path):
            sample_path = os.path.join(base_path, sample)
            if not os.path.isdir(sample_path):
                continue

            # process vessel channels
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

# function to process all samples
def process_all(output_root):

    # define output root
    os.makedirs(output_root, exist_ok=True)
    for tiff_dir, prefix in get_all_sample_dirs():
        extract_patches_from_stack(tiff_dir, output_root, prefix)

# function to process just 1 sample
def process_single(tiff_dir, prefix, output_dir):
    extract_patches_from_stack(tiff_dir, output_dir, prefix)



# --- Main ---

if __name__ == '__main__':

    # get args from command line
    parser = argparse.ArgumentParser()
    parser.add_argument('--single_dir', type=str, help='Path to directory with tif slices')
    parser.add_argument('--prefix', type=str, help='Prefix for filenames of saved patches')
    parser.add_argument('--output_dir', type=str, help='Directory to save extracted patches')
    parser.add_argument('--sample_index', type=int, help='Index for SLURM array job')
    parser.add_argument('--num_patches', type=int, default=10, help='Max random patches per image (default 10)')
    parser.add_argument('--min_fg', type=float, default=0.05, help='Minimum required foreground fraction to keep a patch (default 0.05)')
    parser.add_argument('--seed', type=int, default=100, help='Random seed for reproducible sampling.')
    parser.add_argument('--patch_size', type=int, default=96, help='Cubic patch size in voxels (default 96)')
    parser.add_argument('--only_structure', type=str, choices=list(structures.keys()), help='Only process this structure (e.g. vessel)')
    args = parser.parse_args()

    # set seed for reproducibility
    GLOBAL_SEED = args.seed
    random.seed(GLOBAL_SEED)
    np.random.seed(GLOBAL_SEED)

    # override defaults from cli if provided
    if args.num_patches is not None:
        NUM_RANDOM_PATCHES = args.num_patches
    if args.min_fg is not None:
        MIN_FOREGROUND_FRACTION = args.min_fg

    # get patch size
    PATCH_SIZE = int(args.patch_size)
    if PATCH_SIZE <= 0:
        raise ValueError(f'Invalid patch size: {PATCH_SIZE}. Must be positive integer.')

    # resolve output directory
    output_dir = args.output_dir

    print(f'Using patch_size={PATCH_SIZE}, num_patches={NUM_RANDOM_PATCHES}, min_fg={MIN_FOREGROUND_FRACTION}, seed={GLOBAL_SEED}, output_dir={output_dir}', flush=True)

    # function to process single tif stack
    if args.sample_index is not None:
        all_samples = get_all_sample_dirs(only_structure=args.only_structure)

        if not all_samples:
            print(f'No tiff stacks found under {ROOT_DIR} (filter={args.only_structure}). Check directory layout and permissions.', flush=True)
            raise SystemExit(1)
        if 0 <= args.sample_index < len(all_samples):
            input_dir, prefix = all_samples[args.sample_index]
            os.makedirs(output_dir, exist_ok=True)
            process_single(input_dir, prefix, output_dir)

        else:
            print(f'Invalid sample index {args.sample_index}. Range: 0 to {len(all_samples)-1}')
            raise SystemExit(1)
    
    elif args.single_dir and args.prefix and args.output_dir:
        os.makedirs(output_dir, exist_ok=True)
        process_single(args.single_dir, args.prefix, output_dir)

    else:
        raise ValueError('Specify --sample_index or (--single_dir and --prefix).')



















