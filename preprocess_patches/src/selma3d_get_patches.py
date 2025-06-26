# Script to Extract Smaller Patches from 2D .tif Stacks and Create .tiff volumes

# --- Setup ---

# imports
import argparse
import numpy as np
import os
from skimage.filters import threshold_otsu
import tifffile as tiff

# define constants
PATCH_SIZE = 96
STRIDE = 96
MIN_FOREGROUND_FRACTION = 0.05

# create dictionary mapping subfolder names to structure names and prefixes
structures = {
    'unannotated_ab_plaque': ('Ab_plaques', 'ab_plaque'),
    'unannotated_cfos': ('c-Fos_brain_cells', 'cfos'),
    'unannotated_nucleus': ('cell_nucleus', 'nucleus'),
    'unannotated_vessel': ('vessel', 'vessel')
}

# define specific channels for vessel images
channels = {'vessel': {'C00': 'vessel_wga',
                        'C01': 'vessel_eb'}}


# --- Functions ---

# function to pad slice to correct x,y dimensions
def pad_slice(slice_array, target_shape):
    print('Padding slice...', flush=True)
    pad_y = max(0, target_shape[0] - slice_array.shape[0])
    pad_x = max(0, target_shape[1] - slice_array.shape[1])
    return np.pad(slice_array, ((0, pad_y), (0, pad_x)), mode='constant', constant_values=0)

# function to extract patches from directory of 2d tiff slices
def extract_patches_from_stack(tiff_dir, output_dir, prefix):

    print(f'Processing: {prefix} ({tiff_dir})', flush=True)

    print('Padding...', flush=True)

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

    print(f'Height, width: ({height}, {width}); padded: {padded_shape}')

    # pad z if necessary
    pad_z = (PATCH_SIZE - len(slice_files) % PATCH_SIZE) % PATCH_SIZE
    total_slices = len(slice_files) + pad_z

    print(f'z padding: {pad_z}, total slices: {total_slices}')

    # create list of slices
    patch_id = 0
    for z in range(0, total_slices, PATCH_SIZE):
        slices = []
        for zi in range(z, z + PATCH_SIZE):
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
        threshold = threshold_otsu(volume[::4, ::4, ::4])

        # save extracted patch
        for y in range(0, volume.shape[1], PATCH_SIZE):
            for x in range(0, volume.shape[2], PATCH_SIZE):
                patch = volume[:, y:y+PATCH_SIZE, x:x+PATCH_SIZE]
                fg_fraction = (patch > threshold).sum() / patch.size
                if fg_fraction >= MIN_FOREGROUND_FRACTION:
                    
                    # get full datatype from prefix
                    if prefix.startswith('vessel_eb'):
                        datatype = 'vessel_eb'
                    elif prefix.startswith('vessel_wga'):
                        datatype = 'vessel_wga'
                    elif prefix.startswith('ab_plaque'):
                        datatype = 'ab_plaque'
                    elif prefix.startswith('cfos'):
                        datatype = 'cfos'
                    elif prefix.startswith('nucleus'):
                        datatype = 'nucleus'
                    else:
                        datatype = prefix.split('_')[0] # fallback

                    sample_name = prefix.replace(datatype + '_', '')

                    subfolder = os.path.join(output_dir, datatype)
                    os.makedirs(subfolder, exist_ok=True)
                    channel = os.path.basename(tiff_dir)
                    patch_path = os.path.join(subfolder, f'{datatype}_{sample_name}_{channel}_p{patch_id}.tiff')
                    tiff.imwrite(patch_path, patch.astype(np.uint16), imagej=True)

                    print(f'Saving extract patch, id {patch_id}...', flush=True)

                    patch_id += 1

# function to get all subdirectories
def get_all_sample_dirs():

    # define root directory
    root_dir = '/midtier/paetzollab/scratch/ads4015/data_selma3d'

    # define list for samples
    samples = []

    # get base path
    for structure_key, (structure_folder, prefix_name) in structures.items():
        base_path = os.path.join(root_dir, structure_key, 'brain_microscopy_image', 'brain_microscopy_image', structure_folder)
        print('base_path:', base_path, flush=True)
        if not os.path.exists(base_path):
            continue

        # get sample
        for sample in os.listdir(base_path):
            sample_path = os.path.join(base_path, sample)
            # print('sample_path', sample_path, flush=True)
            if not os.path.isdir(sample_path):
                continue

            # process vessel channels
            if structure_key == 'unannotated_vessel': 
                for channel, channel_prefix in channels['vessel'].items():
                    channel_path = os.path.join(sample_path, channel)
                    if os.path.isdir(channel_path):
                        prefix = f'{channel_prefix}_{sample}'
                        samples.append((channel_path, prefix))

            else:
                for subdir in os.listdir(sample_path):
                    tiff_dir = os.path.join(sample_path, subdir)
                    # print('tiff_dir', tiff_dir)
                    if os.path.isdir(tiff_dir):
                        prefix = f'{prefix_name}_{sample}'
                        samples.append((tiff_dir, prefix))

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
    args = parser.parse_args()

    # function to process single tif stack
    if args.sample_index is not None:
        print('Correct location', flush=True)
        all_samples = get_all_sample_dirs()
        if 0 <= args.sample_index < len(all_samples):
            input_dir, prefix = all_samples[args.sample_index]
            output_dir = '/midtier/paetzollab/scratch/ads4015/data_selma3d/small_patches'
            process_single(input_dir, prefix, output_dir)

        else:
            print(f'Invalid sample index {args.sample_index}. Range: 0 to {len(all_samples)-1}')
    
    elif args.single_dir and args.prefix and args.output_dir:
        raise ValueError('Processing individual?')
        # process_single(args.single_dir, args.prefix, args.output_dir)

    else:
        raise ValueError('Processing multiple?')
        # process_all()



















