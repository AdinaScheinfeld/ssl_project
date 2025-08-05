# Script to download a patch from a neuroglancer volume and save it as a NIfTI file

# script supports input as either:
# 1. 2 corners in format "x1,y1,z1;x2,y2,z2"
# 2. ranges in format "x1:x2,y1:y2,z1:z2"
# 3. center coordinates and size in format "x,y,z" and "size_x,size_y,size_z"

# imports
import argparse
from cloudvolume import CloudVolume
import nibabel as nib
import numpy as np
import os
import re
import requests


# function to get affine transformation matrix from neuroglancer info file
def get_affine_from_neuroglancer(vol_path):

    # create path to info.json file of volumn
    info_url = vol_path.rstrip('/') + '/info'

    # get info from info.json
    response = requests.get(info_url)
    response.raise_for_status()
    info = response.json()

    # get data from highest resolution
    scale = info['scales'][0]
    resolution_nm = scale['resolution'] # get resolution in nanometers
    resolution_mm = [r * 1e-6 for r in resolution_nm] # convert resolution from nanometers to millimeters
    voxel_offset = scale['voxel_offset']
    offset_mm = [o*s for o, s in zip(voxel_offset, resolution_mm)] # convert voxel coords into world coords (mm)

    # construct affine transform matrix
    affine = np.array([
        [resolution_mm[0], 0, 0, offset_mm[0]],
        [0, resolution_mm[1], 0, offset_mm[1]],
        [0, 0, resolution_mm[2], offset_mm[2]],
        [0, 0, 0, 1]
    ])

    # return affine transform matrix
    return affine


# function to parse patch input
def parse_patch_input(coord_input, size_input):

    # case 1: two corners provided in format "x1,y1,z1;x2,y2,z2"
    if ';' in coord_input:

        # get two corners and extract coordinates
        corner1, corner2 = coord_input.split(';')
        x1, y1, z1 = map(int, corner1.strip().split(','))
        x2, y2, z2 = map(int, corner2.strip().split(','))

        # calculate center coordinates
        center = [(x1 + x2) // 2, (y1 + y2) // 2, (z1 + z2) // 2]

        # get size from input
        size = [int(s.strip()) for s in size_input.split(',')]

        # return center coordinates and size
        return center, size



    # case 2: range provided in format "x1:x2,y1:y2,z1:z2"
    if ':' in coord_input:

        # extract x, y, z ranges
        ranges = [c.strip() for c in coord_input.split(',')]

        # ensure that there are 3 ranges (x, y, z)
        if len(ranges) != 3:
            raise ValueError('coord_input must be in the form "x1:x2,y1:y2,z1:z2"')
        
        # create list for coordinates and sizes
        coords = []
        sizes = []

        # loop through ranges and get coordinates starts and end and sizes
        for range in ranges:
            start, end = map(int, range.split(':'))
            coords.append((start + end) // 2) # get center of range
            sizes.append(end - start) # get size of range
        
        # return center coordinates and size
        return coords, sizes
    
    # case 3: center and size are provided
    else:

        # get center and size from input
        center = [int(c.strip()) for c in coord_input.split(',')]
        size = [int(s.strip()) for s in size_input.split(',')]

        # return center coordinate and size
        return center, size


# function to download patch from neuroglancer
def download_patch(vol_path, center, filename, size=(256, 256, 128)):

    # use cloudvolume to get volume
    vol = CloudVolume(vol_path, use_https=True)

    # download the patch centered at center with size=size
    patch = vol.download_point(center, size)
    print(f'Volume shape: {tuple(int(dim) for dim in vol.shape)}, patch shape: {patch.shape}')
    print(f'Volume origin: {vol.voxel_offset}')

    # get affine transform matrix
    affine = get_affine_from_neuroglancer(vol_path)

    # save to nifti format
    nifti_img = nib.Nifti1Image(patch, affine=affine) 
    nib.save(nifti_img, filename)
    print(f'Saved patch to {filename}')

# main
if __name__ == '__main__':

    # get args
    parser = argparse.ArgumentParser(description='Download patch from CloudVolume and save as nifti file.')
    parser.add_argument('--vol_path', type=str, required=True, help='CloudVolume path, ex: "https://redcloud.cac.cornell.edu:8443/swift/v1/demo_datasets/TH_C-R45/Ex_647"')
    parser.add_argument('--coord_input', type=str, required=True, help='Either center coordinates as "x,y,z" or ranges as "x1:x2,y1:y2,z1:z2"')
    parser.add_argument('--folder', type=str, required=True, help='Output folder, ex: /path/to/folder')
    parser.add_argument('--size', type=str, default='256,256,128', help='Patch size (used if center is given OR for 2 corner input), e.g. 256, 256, 128')
    parser.add_argument('--suffix', type=str, default='', help='Optional string to append to filename, ex: "_patch"')
    args = parser.parse_args()

    # get center and size from input
    center, size = parse_patch_input(args.coord_input, args.size)

    # generate filename (filename is the folder name in lowercase followed by the center coordinates separated by underscores)
    full_folder = os.path.basename(args.folder.rstrip('/')) # filename begins with folder name
    folder_name_stripped = re.sub(f'^\d+_', '', full_folder) # use regex to remove leading number and underscore from folder name
    folder_name = folder_name_stripped.lower() # convert to lowercase
    center_str = '_'.join(map(str, center)) # get center coordinates as underscore-separated string
    suffix_str = args.suffix.strip() if args.suffix else '' # get suffix if provided
    filename = os.path.join(args.folder, f'input/{folder_name}{suffix_str}_{center_str}.nii.gz')


    # download patch
    download_patch(args.vol_path, center, filename, size)



# to use the script:
# python ng2nii.py "<vol_path>" "coord_input (as comma sep list of center coords or x,y,z ranges)" "filename (with path)"

