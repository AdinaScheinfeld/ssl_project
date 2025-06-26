# Functions used in visualization notebooks and scripts

# --- Setup ---

# imports
import numpy as np
import tifffile as tiff

from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    MapTransform,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ToTensord
)


# --- Functions ---

# custom loader for tiff files
class LoadTiffd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        img = tiff.imread(d['image'])
        if img.ndim == 3:
            img = np.expand_dims(img, axis=0) # add channel first
        d['image'] = img
        return d
    
# function to get training transforms
def get_train_transforms():
    return Compose([
        LoadTiffd(keys=['image']),
        RandFlipd(keys=['image'], spatial_axis=[0, 1, 2], prob=0.5),
        RandAffined(keys=['image'], rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1), prob=0.5),
        RandGaussianNoised(keys=['image'], prob=0.3, mean=0.0, std=0.01),
        RandScaleIntensityd(keys=['image'], factors=0.2, prob=0.5),
        RandShiftIntensityd(keys=['image'], offsets=0.1, prob=0.5),
        ToTensord(keys=['image'])
    ])

# function to get validation transforms
def get_val_transforms():
    return Compose([
        LoadTiffd(keys=['image']),
        ToTensord(keys=['image'])
    ])
    


    

