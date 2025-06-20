# Python script to organize SELMA 3D data into 1 folder with datatype appended to sample folder names

# --- Setup ---

# imports
import argparse
import os
from pathlib import Path
import shutil

# parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--index', type=int, required=True, help='Index of sample to process.')
args = parser.parse_args()


# --- Copy files ---

# define base and output directories
base_dir = Path('/midtier/paetzollab/scratch/ads4015/data_selma3d')
output_dir = base_dir / 'unannotated_combined'
output_dir.mkdir(parents=True, exist_ok=True)

# define class info
class_info = {
    'ab_plaque': base_dir / 'unannotated_ab_plaque/brain_microscopy_image/brain_microscopy_image/Ab_plaques',
    'cfos': base_dir / 'unannotated_cfos/brain_microscopy_image/brain_microscopy_image/c-Fos_brain_cells',
    'nucleus': base_dir / 'unannotated_nucleus/brain_microscopy_image/brain_microscopy_image/cell_nucleus',
    'vessel': base_dir / 'unannotated_vessel/brain_microscopy_image/brain_microscopy_image/vessel'
}

# create mapping for vessel channels
vessel_channel_map = {
    'C00': 'wga', # microvessels
    'C01': 'eb' # major vessels
}

# dynamically build a list of (class_label, sample_path) pairs
sample_list = []
for class_label, class_path in class_info.items():
    if class_path.exists():
        for sample_folder in sorted(class_path.iterdir()):
            if sample_folder.is_dir():
                sample_list.append((class_label, sample_folder))

# get the specific sample by index passed in on command line
try: 
    class_label, sample_folder = sample_list[args.index] # select the class/sample combination to process
except IndexError:
    raise ValueError(f'Index {args.index} out of range. Only {len(sample_list)} samples available.')

# get the folder name (ex: sample1, sample2, ...)
sample_id = sample_folder.name

# recursively walk through all subdirectories and files for the sample
for subdir, _, files in os.walk(sample_folder):
    rel_subdir = Path(subdir).relative_to(sample_folder) # get relative path from sample folder
    subdir_name = rel_subdir.parts[0] if len(rel_subdir.parts) > 0 else ''

    # determine output folder name based on class and vessel subtype
    if class_label == 'vessel' and subdir_name in vessel_channel_map:
        subtype = vessel_channel_map[subdir_name] # wga or eb
        out_subdir = output_dir / f'{class_label}_{subtype}_{sample_id}'
    else:
        out_subdir = output_dir / f'{class_label}_{sample_id}'

    # create the otuput subdirectory if it doesn't already exist
    out_subdir.mkdir(parents=True, exist_ok=True)

    # rename files with prefix to avoid collisions and copy them to destination
    prefix = str(rel_subdir).replace(os.sep, '_')
    for fname in files:
        if fname.lower().endswith('.tif'):
            src = Path(subdir) / fname

            # prepend subfolder name to filename to avoid collisions
            new_fname = f'{prefix}_{fname}' if prefix != '.' else fname
            dst = out_subdir / new_fname
            shutil.copy2(src, dst) # copy file to new location

# indicate completion
print(f'Done. All images organized in: {output_dir}', flush=True)








