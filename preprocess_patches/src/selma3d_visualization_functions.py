# Functions used in visualization notebooks and scripts

# --- Setup ---

# imports
import numpy as np
import tifffile as tiff

from monai.transforms import (
    CastToTyped,
    Compose,
    EnsureChannelFirstd,
    MapTransform,
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    RandGaussianSmoothd,
    RandRotate90d,
    RandScaleIntensityd,
    RandShiftIntensityd,
    ScaleIntensityRangePercentilesd,
    ThresholdIntensityd,
    ToTensord
)


# --- Functions ---

# custom loader for tiff files
class LoadTiffd(MapTransform):
    def __call__(self, data):
        d = dict(data)
        img = tiff.imread(d['image'])
        if img.ndim == 2:
            raise ValueError(f'Image is 2d, expected 3d. File: {d["image"]}')
        elif img.ndim == 3:
            img = np.expand_dims(img, axis=0) # add channel first
        elif img.ndim ==4:
            print(f'Warning: 4d image encountered, shape: {img.shape}. File: {d["image"]}')

        # ensure dtype is float32
        if not np.issubdtype(img.dtype, np.floating):
            img = img.astype(np.float32)

        d['image'] = img
        return d
    


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

        # load
        LoadTiffd(keys=['image']),
        CastToTyped(keys=['image'], dtype=np.float32),

        # normalize based on input percentiles
        ScaleIntensityRangePercentilesd(keys=['image'], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),

        # spatial augmentations
        RandFlipd(keys=['image'], spatial_axis=[0, 1, 2], prob=0.2),
        RandRotate90d(keys=['image'], prob=0.2, max_k=3),
        RandAffined(keys=['image'], rotate_range=(0.1, 0.1, 0.1), scale_range=(0.1, 0.1, 0.1), prob=0.2),

        # intensity augmentations
        RandGaussianNoised(keys=['image'], prob=0.2, mean=0.0, std=0.02),
        RandGaussianSmoothd(keys=['image'], prob=0.2),
        RandScaleIntensityd(keys=['image'], factors=0.2, prob=0.2),
        RandShiftIntensityd(keys=['image'], offsets=0.2, prob=0.2),
        ClampIntensityd(keys=["image"], minv=0.0, maxv=1.0),

        # convert to tensor
        ToTensord(keys=['image'])
    ])

# function to get validation transforms
def get_val_transforms():
    return Compose([
        LoadTiffd(keys=['image']),
        ScaleIntensityRangePercentilesd(keys=['image'], lower=1, upper=99, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=['image'])
    ])


# get loading transforms
def get_load_transforms():
    return Compose([
        LoadTiffd(keys=['image']),
        ToTensord(keys=['image'])
    ])
    


    