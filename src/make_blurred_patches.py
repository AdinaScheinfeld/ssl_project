# /home/ads4015/ssl_project/src/make_blurred_patches.py - Script to create blurred image patches for deblurring/supperresolution downstream tasks

# --- Setup ---

# imports
import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
from scipy.ndimage import gaussian_filter


# --- Functions ---

# function to parse args
def parse_args():

    parser = argparse.ArgumentParser(description="Create blurred image patches for deblurring/supperresolution downstream tasks")
    parser.add_argument('--input_root', type=str, required=True, help='Path to the root directory containing input images')
    parser.add_argument('--output_root', type=str, required=True, help='Path to the root directory to save blurred image patches')
    parser.add_argument('--sigma', type=float, default=1.0, help='Standard deviation for Gaussian kernel used for blurring')
    parser.add_argument('--noise_std', type=float, default=0.0, help='Standard deviation of Gaussian noise to add')
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing files (otherwise skip files that already exist)')

    args = parser.parse_args()
    return args

# function to check if file is image or binary label (only process images)
def is_image_file(file_path):

    # ensure path is file
    if not file_path.is_file():
        return False
    
    # check if 'label' in filename
    name = file_path.name
    if 'label' in name.lower():
        return False
    
    # check file extension
    if name.endswith('.nii') or name.endswith('.nii.gz'):
        return True
    
    return False

# function to create blurred patches (load nifti file, blur, optionally add noise, save new nifti file)
def create_blurred_patch(input_path, output_path, sigma, noise_std, overwrite=False):

    # create output directory if it doesn't exist
    if output_path.exists() and not overwrite:
        print(f'[INFO] Skipping existing file: {output_path}', flush=True)
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # load image
    img = nib.load(str(input_path))
    img_data = img.get_fdata(dtype=np.float32)
    print(f'[INFO] Loaded image: {input_path} with shape {img_data.shape}', flush=True)

    # apply 3d gaussian blur
    if img_data.ndim == 3:
        blurred_data = gaussian_filter(img_data, sigma=sigma)

    # handle 4d images (e.g., multi-channel)
    elif img_data.ndim == 4:

        # assume channels are first or last (detect based on shape; smaller dimension is likely channels)
        if img_data.shape[0] <= 4: # assume (C, Z, Y, X)
            blurred_data = np.empty_like(img_data)
            for c in range(img_data.shape[0]):
                blurred_data[c] = gaussian_filter(img_data[c], sigma=sigma)
        elif img_data.shape[-1] <= 4: # assume (Z, Y, X, C)
            blurred_data = np.empty_like(img_data)
            for c in range(img_data.shape[-1]):
                blurred_data[..., c] = gaussian_filter(img_data[..., c], sigma=sigma)
        
        # fallback to blur all dimensions
        else:
            blurred_data = gaussian_filter(img_data, sigma=sigma)

    # fallback to blur any shape
    else:
        blurred_data = gaussian_filter(img_data, sigma=sigma)

    # optionally add gaussian noise
    if noise_std > 0.0:
        noise = np.random.normal(loc=0.0, scale=noise_std, size=blurred_data.shape).astype(np.float32)
        blurred_data += noise

    # cast back to float32
    blurred_data = blurred_data.astype(np.float32)

    # save new nifti file with same affine and header
    output_img = nib.Nifti1Image(blurred_data, affine=img.affine, header=img.header)
    nib.save(output_img, str(output_path))
    print(f'[INFO] Saved blurred image: {output_path}', flush=True)


# --- Main ---

# main function
def main():

    # parse args
    args = parse_args()

    # get input and output root paths
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()
    if not input_root.exists():
        raise FileNotFoundError(f'Input root directory does not exist: {input_root}')
    
    # print settings
    print(f'[INFO] Input root directory: {input_root}', flush=True)
    print(f'[INFO] Output root directory: {output_root}', flush=True)
    print(f'[INFO] Gaussian sigma: {args.sigma}', flush=True)
    print(f'[INFO] Noise std: {args.noise_std}', flush=True)

    # iterate over all files in input root
    all_files = sorted(input_root.rglob('*.nii*'))
    print(f'[INFO] Found {len(all_files)} files in input root directory', flush=True)

    # process each file
    for input_path in all_files:

        # ensure it's an image file
        if not is_image_file(input_path):
            continue

        # determine relative path and output path
        relative_path = input_path.relative_to(input_root)
        output_path = output_root / relative_path

        # create blurred patch
        create_blurred_patch(input_path, output_path, sigma=args.sigma, noise_std=args.noise_std, overwrite=args.overwrite)


# entry point
if __name__ == '__main__':
    main()



























