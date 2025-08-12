# wu_transforms.py - Functions used when finetuning model trained on Wu data

# --- Setup ---

# imports

import numpy as np

from monai.transforms import (
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    MapTransform,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd,
    SqueezeDimd,
    ToTensord
)

    
# --- Pretraining Transforms ---

# transform to clamp image intensity between 0-1
class ClampIntensityd(MapTransform):
    def __init__(self, keys, minv=0.0, maxv=1.0):
        super().__init__(keys)
        self.minv = minv
        self.maxv = maxv

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            d[key] = np.clip(d[key], self.minv, self.maxv)
        return d


# function to get training transforms
def get_train_transforms():
    return Compose([

        # spatial augmentations
        RandFlipd(keys=['image'], spatial_axis=[0, 1, 2], prob=0.2),
        RandRotate90d(keys=['image'], prob=0.2, max_k=3),
        RandAffined(keys=['image'], rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1), prob=0.2),

        # intensity augmentations
        RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.02),
        RandGaussianSmoothd(keys=['image'], prob=0.2),
        RandScaleIntensityd(keys=['image'], factors=0.2, prob=0.2),
        RandShiftIntensityd(keys=['image'], offsets=0.2, prob=0.2),
        ClampIntensityd(keys=['image'], minv=0.0, maxv=1.0),

        # convert to tensor (IMPORTANT)
        ToTensord(keys=['image'])
    ])


# function to get validation transforms
def get_val_transforms():
    return Compose([]) # empty transform for consistency in coding


# get loading transforms
def get_load_transforms():
    return Compose([
        LoadImaged(keys=['image']),
        SqueezeDimd(keys=['image'], dim=-1), # remove trailing channel dimension
        EnsureChannelFirstd(keys=['image'], channel_dim='no_channel'), # move channel dimension to front or add it if missing
        ScaleIntensityRangePercentilesd(keys=['image'], lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=['image']) # (IMPORTANT)
    ])


# --- Finetuning Transforms ---

# function to get train transforms for finetuning
def get_finetune_train_transforms():
    return Compose([

        # EnsureChannelFirstd(keys=['image', 'label']), # don't need this transform when using .pt files

        # scale intensity to normalize
        ScaleIntensityRangePercentilesd(keys=['image'], lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True),

        # spatial augmentations
        RandFlipd(keys=['image', 'label'], spatial_axis=[0, 1, 2], prob=0.2),
        RandRotate90d(keys=['image', 'label'], prob=0.2, max_k=3),
        RandAffined(keys=['image', 'label'], rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1), prob=0.2),

        # intensity augmentations
        RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.02),
        RandGaussianSmoothd(keys=['image'], prob=0.2),
        RandScaleIntensityd(keys=['image'], factors=0.2, prob=0.2),
        RandShiftIntensityd(keys=['image'], offsets=0.2, prob=0.2),
        ClampIntensityd(keys=['image'], minv=0.0, maxv=1.0),

        # convert to tensor
        ToTensord(keys=['image', 'label'])
    ])


# function to get validation transforms for finetuning
def get_finetune_val_transforms():
    return Compose([
        # EnsureChannelFirstd(keys=['image', 'label']), # don't need this transform when using .pt files
        ScaleIntensityRangePercentilesd(keys=['image'], lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=['image', 'label'])
    ])

























