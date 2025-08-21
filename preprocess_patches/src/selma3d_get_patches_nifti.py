# selma3d_get_patches.py - Script to Extract Smaller Patches from 2D .tif Stacks and Create .tiff volumes

# --- Setup ---

# imports
import argparse
import nibabel as nib
import numpy as np
import os
import random
from skimage.filters import threshold_otsu
import tifffile as tiff

# define constants
PATCH_SIZE = 96
STRIDE = 96
MIN_FOREGROUND_FRACTION = 0.05
NUM_RANDOM_PATCHES = 10
SAVE_AS_NIFTI = True

# create dictionary mapping subfolder names to structure names and prefixes
structures = {
    'unannotated_ab_plaque': ('Ab_plaques', 'ab_plaque'),
    'unannotated_cfos': ('c-Fos_brain_cells', 'cfos'),
    'unannotated_chondrocytes': ('chondrocytes', 'chondrocytes'),
    'unannotated_chondrogenic_cells': ('chondrogenic_cells', 'chondrogenic_cells'),
    'unannotated_nucleus': ('cell_nucleus', 'nucleus'),
    'unannotated_vessel': ('vessel', 'vessel')
}

# define specific channels for vessel images
channels = {'vessel': {'C00': 'vessel_wga',
                        'C01': 'vessel_eb'}}


# --- Functions ---

# function to pad slice to correct x,y dimensions
def pad_slice(slice_array, target_shape, verbose=False):
    if verbose:
        print('Padding slice...', flush=True)
    pad_y = max(0, target_shape[0] - slice_array.shape[0])
    pad_x = max(0, target_shape[1] - slice_array.shape[1])
    return np.pad(slice_array, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0)

# function to extract patches from directory of 2d tiff slices
def extract_patches_from_stack(tiff_dir, output_dir, prefix):

    print(f'Processing: {prefix} ({tiff_dir})', flush=True)

    # get slices
    slice_files = sorted([f for f in os.listdir(tiff_dir) if f.endswith('.tif') and not f.startswith('.')])
    if len(slice_files) == 0:
        return
    
    # create output directory
    os.makedirs(output_dir, exist_ok=True)

    # determine max dimensions of slices to define padding target for slice
    sample_slice = tiff.imread(os.path.join(tiff_dir, slice_files[0]))
    height, width = sample_slice.shape
    pad_y = (PATCH_SIZE - height % PATCH_SIZE) % PATCH_SIZE
    pad_x = (PATCH_SIZE - width % PATCH_SIZE) % PATCH_SIZE
    padded_shape = (height + pad_y, width + pad_x)

    # pad z if necessary
    pad_z = (PATCH_SIZE - len(slice_files) % PATCH_SIZE) % PATCH_SIZE
    total_slices = len(slice_files) + pad_z

    print(f'Height, width: ({height}, {width}); padded: {padded_shape}', flush=True)
    print(f'z padding: {pad_z}, total slices: {total_slices}', flush=True)

                    
    # get full datatype from prefix
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
        datatype = prefix.split('_')[0] # fallback

    sample_name = prefix.replace(datatype + '_', '')

    subfolder = os.path.join(output_dir, datatype)
    os.makedirs(subfolder, exist_ok=True)
    channel = os.path.basename(tiff_dir)

    # reservoir sampling to keep at most NUM_RANDOM_PATCHES patches in memory
    # each item is dict with keys {'patch': np.ndarray, 'idx': int}
    selected = []
    candidates_seen = 0
    saved_count = 0

    # create list of slices

    patch_block_index = 0

    # iterate over z blocks of 96 slices
    for z0 in range(0, total_slices, PATCH_SIZE):

        # build a 96-slice padded volume block
        slices = []
        for zi in range(z0, z0 + PATCH_SIZE):
            if zi < len(slice_files):
                slice_data = tiff.imread(os.path.join(tiff_dir, slice_files[zi]))
                slice_data = pad_slice(slice_data, padded_shape)
            else:
                slice_data = np.zeros(padded_shape, dtype=np.uint16)
            slices.append(slice_data)

        # stack slices into volume
        volume = np.stack(slices)

        print('Thresholding...', flush=True)

        # use downsampled version of image for otsu's thresholding
        ds = volume[::4, ::4, ::4]
        threshold = threshold_otsu(ds) if ds.min() != ds.max() else ds.min()

        # scan non-overlapping (y, x) windows
        for y in range(0, volume.shape[1], PATCH_SIZE):
            for x in range(0, volume.shape[2], PATCH_SIZE):
                patch = volume[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]

                # ensure correct size 
                if patch.shape != (PATCH_SIZE, PATCH_SIZE, PATCH_SIZE):
                    continue

                # ensure sufficient foreground
                fg_fraction = (patch > threshold).sum() / patch.size
                if fg_fraction >= MIN_FOREGROUND_FRACTION:

                    candidates_seen += 1

                    # reservoir sampling to keep a uniform random sample of size K
                    if len(selected) < NUM_RANDOM_PATCHES:
                        selected.append({'patch': patch.copy(), 'idx': candidates_seen-1})
                    else:
                        r = random.randint(0, candidates_seen - 1)
                        if r < NUM_RANDOM_PATCHES:
                            selected[r] = {'patch': patch.copy(), 'idx': candidates_seen-1}

        patch_block_index += 1


    # write selected patches
    print(f'Foreground-qualified candidates found: {candidates_seen}. Will save up to {min(NUM_RANDOM_PATCHES, candidates_seen)}.', flush=True)

    for i, item in enumerate(selected):

        patch = item['patch']

        # save as nifti
        if SAVE_AS_NIFTI:

            # transpose (z, y, x) -> (x, y, z) for nifti
            patch_nifti = np.transpose(patch.astype(np.uint16), (2, 1, 0))
            patch_path = os.path.join(subfolder, f'{datatype}_{sample_name}_{channel}_p{saved_count}.nii.gz')

            # save patch using nibabel
            nib.save(nib.Nifti1Image(patch_nifti, affine=np.eye(4)), patch_path)
        
        # save as tiff
        else:
            patch_path = os.path.join(subfolder, f'{datatype}_{sample_name}_{channel}_p{saved_count}.tiff')
            tiff.imwrite(patch_path, patch.astype(np.uint16), imagej=True)

        print(f'Saving random patch {saved_count} -> {patch_path}', flush=True)
        saved_count += 1

    print(f'Done. Saved {saved_count} patches.', flush=True)


# function to get all subdirectories
# based on flat layout. Ex: /midtier/.../data_selma3d/unannotated_*/<sample>/<[C01|C00]>
def get_all_sample_dirs():

    # define root directory
    root_dir = '/midtier/paetzollab/scratch/ads4015/data_selma3d'

    # define list for samples
    samples = []

    # function to check if there are tif files in a folder
    def has_tifs(d):
        try:
            return any(f.endswith('.tif') and not f.startswith('.') for f in os.listdir(d))
        except Exception:
            return False

    # get base path
    for structure_key, (structure_folder, prefix_name) in structures.items():
        base_path = os.path.join(root_dir, structure_key)
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
                    if os.path.isdir(channel_path):
                        prefix = f'{channel_prefix}_{sample}'
                        samples.append((channel_path, prefix))

            else:
                for subdir in os.listdir(sample_path):
                    tiff_dir = os.path.join(sample_path, subdir)
                    if os.path.isdir(tiff_dir) and has_tifs(tiff_dir):
                        prefix = f'{prefix_name}_{sample}'
                        samples.append((tiff_dir, prefix))

    print(f'Discovered {len(samples)} stacks(s).', flush=True)
    return samples

# function to process all samples
def process_all():

    # define output root
    output_root = '/midtier/paetzollab/scratch/ads4015/data_selma3d/small_patches'
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
    args = parser.parse_args()

    # override defaults from cli if provided
    if args.num_patches is not None:
        NUM_RANDOM_PATCHES = args.num_patches
    if args.min_fg is not None:
        MIN_FOREGROUND_FRACTION = args.min_fg

    # set seed for reproducibility
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)

    print(f'Using num_patches={NUM_RANDOM_PATCHES}, min_fg={MIN_FOREGROUND_FRACTION}, seed={args.seed}, output_dir={args.output_dir}', flush=True)

    # function to process single tif stack
    if args.sample_index is not None:
        print('Correct location', flush=True)
        all_samples = get_all_sample_dirs()
        if 0 <= args.sample_index < len(all_samples):
            input_dir, prefix = all_samples[args.sample_index]
            output_dir = args.output_dir
            os.makedirs(output_dir, exist_ok=True)
            process_single(input_dir, prefix, output_dir)

        else:
            print(f'Invalid sample index {args.sample_index}. Range: 0 to {len(all_samples)-1}')
    
    elif args.single_dir and args.prefix and args.output_dir:
        raise ValueError('Processing individual?')
        # process_single(args.single_dir, args.prefix, args.output_dir)

    else:
        raise ValueError('Processing multiple?')
        # process_all()



















