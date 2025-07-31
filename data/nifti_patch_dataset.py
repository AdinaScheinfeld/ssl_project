# Nifti Patch Dataset

# --- Setup ---

# imports
from torch.utils.data import Dataset

# --- Dataset Class ---

# nifti patch dataset class
class NiftiPatchDataset(Dataset):

    # init
    def __init__(self, file_paths, transforms=None):
        self.file_paths = file_paths
        self.transforms = transforms

    # length
    def __len__(self):
        return len(self.file_paths)
    
    # getter
    def __getitem__(self, idx):
        data = {'image': self.file_paths[idx]}
        if self.transforms:
            data = self.transforms(data)
        return data
    