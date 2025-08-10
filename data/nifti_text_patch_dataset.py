# dataset class

# --- Setup ---

# imports
import json
from monai.transforms import Compose as MonaiCompose, LoadImaged, EnsureChannelFirstd, ToTensord
import nibabel as nib
import random
import torch
from torch.utils.data import Dataset

# --- Dataset Class ---

# nifti patch dataset class with text
class NiftiTextPatchDataset(Dataset):

    # init
    def __init__(self, file_paths, transforms=None, text_prompts=None, use_sub_patches=False, base_patch_size=96, sub_patch_size=64):
        self.file_paths = file_paths
        self.transforms = transforms
        self.use_sub_patches = use_sub_patches
        self.base_patch_size = base_patch_size
        self.sub_patch_size = sub_patch_size

        # load stain-text mapping from json file
        if text_prompts is None:
            raise ValueError('text_prompts must be provided to NiftiTextPatchDataset')
        with open(text_prompts, 'r') as f:
            self.stain_map = json.load(f)

        # split transforms into load and other transforms
        self.full_transforms = self.transforms
        load_transforms, train_val_transforms = self.full_transforms.transforms
        if load_transforms is not None:
            self.transforms_no_load = MonaiCompose([MonaiCompose([t for t in load_transforms.transforms if not isinstance(t, LoadImaged)]), train_val_transforms]) # remove LoadImaged from transforms
        else:
            self.transforms_no_load = None

        # if using sub_patches, split each file path into sub_patches
        if self.use_sub_patches:
            self.sub_patches = []
            for path in self.file_paths:
                vol = self.load_volume(path) # shape: (1, base_patch_size, base_patch_size, base_patch_size)
                sub_patch_list = self.split_into_sub_patches(vol)
                text = self.extract_text(path)
                for sub_patch in sub_patch_list:
                    self.sub_patches.append((sub_patch, text))

    # function to load volume from file path
    def load_volume(self, path):
        vol = nib.load(path).get_fdata() # shape: (D, H, W)
        # vol = torch.tensor(vol).float().unsqueeze(0) # shape: (1, D, H, W)
        return vol
    
    # function to split volume into sub_patches
    def split_into_sub_patches(self, vol):

        print(f'[DEBUG] Splitting volume of shape {vol.shape} into sub_patches of size {self.sub_patch_size}', flush=True)

        # create list to hold sub_patches
        sub_patches = []

        # get list of valid start corners for extracting sub_patches
        valid_starts = [(x, y, z) for x in (0, self.sub_patch_size/2) for y in (0, self.sub_patch_size/2) for z in (0, self.sub_patch_size/2)]

        # randomly select 2 unique, non-overlapping start corners
        selected_starts = random.sample(valid_starts, 2)
        # print(f'[DEBUG] Selected starts: {selected_starts}', flush=True)

        # extract sub_patches from selected start corners
        for x, y, z in selected_starts:
            x, y, z = int(x), int(y), int(z) # convert float to int for indexing
            sub_patch = vol[z:z+self.sub_patch_size,  y:y+self.sub_patch_size, x:x+self.sub_patch_size ]

            # sub_patch = sub_patch.squeeze() # remove extra channel dimensions
            # sub_patch = sub_patch.unsqueeze(0) # add channel dimension back

            sub_patches.append(sub_patch)


        
        # return list of sub_patches
        return sub_patches


    # function to extract text prompt from file
    def extract_text(self, path):
        for k, v in self.stain_map.items():
            if k in path.lower():
                return v
        return 'Unknown microscopy patch with unannotated staining and location.' # if unknown folder

    # length
    def __len__(self):

        # if using sub_patches
        if self.use_sub_patches:
            return len(self.sub_patches)
        
        # if not using sub_patches
        return len(self.file_paths)
    
    # getter
    def __getitem__(self, idx):

        # if using sub_patches
        if self.use_sub_patches:

            # image and text
            image, text = self.sub_patches[idx]
            data = {'image': image, 'text': text}
            if self.transforms_no_load:
                data = self.transforms_no_load(data)
            return data
        
        # if not using sub_patches
        else:
            # image and text
            path = self.file_paths[idx]
            data = {'image': path, 'text': self.extract_text(path)}
            if self.full_transforms:
                data = self.full_transforms(data)

            return data



