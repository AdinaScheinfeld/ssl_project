# dataset class

# --- Setup ---
from torch.utils.data import Dataset

# --- Dataset Class ---

# nifti patch dataset class with text
class NiftiTextPatchDataset(Dataset):

    # init
    def __init__(self, file_paths, transforms=None):
        self.file_paths = file_paths
        self.transforms = transforms

    # function to extract text prompt from filename
    def extract_text(self, path):
        path_lower = path.lower()
        if '01_aa1-po-c-r45' in path_lower:
            return 'TH protein, cytoplasm'
        elif '02_az10-sr3b-6_a' in path_lower:
            return 'CTIP2 protein, nucleus'
        elif '03_aj12-lg1e-n_a' in path_lower:
            return 'GFAP protein, cytoplasm'
        elif '04_ae2-wf2a_a' in path_lower:
            return 'p75NTH protein, cell membrane and cytoplasm'
        elif '05_ch1-pcw1a_a' in path_lower:
            return 'DBH protein, cytoplasm and vesicular compartments'
        else:
            return 'unknown stain, unknown location'

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



