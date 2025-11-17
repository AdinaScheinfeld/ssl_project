# tif2nii.py - Script to convert TIFF image volumes to NIfTI format

# --- Setup ---

# imports
import argparse
import nibabel as nib
import numpy as np
from pathlib import Path
import tifffile as tiff


# --- Functions ---

# function to parse arguments
def parse_args():
    parser = argparse.ArgumentParser(description='Convert .tif volumes to .nii.gz')
    parser.add_argument('--input_tifs', nargs='+', required=True, help='List of input .tif volumes')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for .nii.gz files')
    args = parser.parse_args()
    return args

# function to convert tif to nii
def convert_tif_to_nifti(input_path, output_dir):

    # load tif as numpy array
    print(f'[INFO] Loading tiff: {input_path}', flush=True)
    img_arr = tiff.imread(str(input_path))

    # transpose if needed (tifffile loads as ZYX, we want XYZ)
    if img_arr.ndim == 3:
        img_arr = np.transpose(img_arr, (2, 1, 0))  # ZYX to XYZ

    # ensure 3d
    if img_arr.ndim < 3:
        print(f'[WARN] Array from {input_path} is {img_arr.shape}, expanding to 3D.', flush=True)
        while img_arr.ndim < 3:
            img_arr = np.expand_dims(img_arr, axis=-1)

    # affine (use identity since no spatial info in tiff)
    affine = np.eye(4, dtype=np.float32)

    # create nibabel image
    img_nii = nib.Nifti1Image(img_arr.astype(np.float32), affine)

    # prepare output path
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / (input_path.stem + '.nii.gz')

    # save nifti
    print(f'[INFO] Saving NIfTI: {output_path}', flush=True)
    nib.save(img_nii, str(output_path))


# --- Main ---

# main function
def main():

    # parse args
    args = parse_args()

    # output dir
    output_dir = Path(args.output_dir)

    # convert each input tif to nii
    for tif_path in args.input_tifs:
        tif_path = Path(tif_path)
        if not tif_path.is_file():
            print(f'[ERROR] Input file does not exist: {tif_path}', flush=True)
            continue
        convert_tif_to_nifti(tif_path, output_dir)

# run main
if __name__ == '__main__':
    main()




















