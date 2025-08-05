# dataset class

# --- Setup ---
from torch.utils.data import Dataset

# --- Dataset Class ---

# create dict of image-text mapping
STAIN_MAP = {
    '01_aa1-po-c-r45': 'TH protein, cytoplasm',
    '02_az10-sr3b-6_a': 'CTIP2 protein, nucleus',
    '03_aj12-lg1e-n_a': 'GFAP protein, cytoplasm',
    '04_ae2-wf2a_a': 'p75NTH protein, cell membrane and cytoplasm',
    '05_ch1-pcw1a_a': 'DBH protein, cytoplasm and vesicular compartments',
    '06_az10-sr3b-6_a': 'SYTO 24, nucleus',
    '07_az10-sr3b-6_a': 'pAAV-GFP, cytoplasm'
}

# nifti patch dataset class with text
class NiftiTextPatchDataset(Dataset):

    # init
    def __init__(self, file_paths, transforms=None):
        self.file_paths = file_paths
        self.transforms = transforms


    # function to extract text prompt from filename
    def extract_text(self, path):
        for k, v in STAIN_MAP.items():
            if k in path.lower():
                return v
        return 'unknown stain, unknown location' # if unknown folder

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



