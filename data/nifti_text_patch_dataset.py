# dataset class

# --- Setup ---

# imports
import json
from torch.utils.data import Dataset

# --- Dataset Class ---

# nifti patch dataset class with text
class NiftiTextPatchDataset(Dataset):

    # init
    def __init__(self, file_paths, transforms=None, text_prompts=None):
        self.file_paths = file_paths
        self.transforms = transforms

        # load stain-text mapping from json file
        if text_prompts is None:
            raise ValueError('text_prompts must be provided to NiftiTextPatchDataset')
        with open(text_prompts, 'r') as f:
            self.stain_map = json.load(f)


    # function to extract text prompt from file
    def extract_text(self, path):
        for k, v in self.stain_map.items():
            if k in path.lower():
                return v
        return 'Unknown microscopy patch with unannotated staining and location.' # if unknown folder

    # length
    def __len__(self):
        return len(self.file_paths)
    
    # getter
    def __getitem__(self, idx):

        # image
        data = {'image': self.file_paths[idx]}
        if self.transforms:
            data = self.transforms(data)

        # text
        data['text'] = self.extract_text(self.file_paths[idx])

        return data



