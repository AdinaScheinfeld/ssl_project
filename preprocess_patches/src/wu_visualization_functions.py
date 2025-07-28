# Functions used in visualization notebooks and scripts

# --- Setup ---

# imports
import matplotlib.pyplot as plt
import numpy as np
import wandb

from monai.transforms import (
    CastToTyped,
    Compose,
    EnsureChannelFirstd,
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
    ThresholdIntensityd,
    ToTensord
)

    
# --- Transforms Functions ---

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
        ClampIntensityd(keys=["image"], minv=0.0, maxv=1.0),

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
        EnsureChannelFirstd(keys=['image']),
        ScaleIntensityRangePercentilesd(keys=['image'], lower=1.0, upper=99.0, b_min=0.0, b_max=1.0, clip=True),
        ToTensord(keys=['image']) # (IMPORTANT)
    ])


# --- Visualization Functions ---


# visualization function
def visualize_patches(original, augmented, title):

    # get center slice
    center_z = original.shape[1] // 2

    # create figure
    fig, axs = plt.subplots(1, 2, figsize=(6, 3))
    fig.suptitle(title, fontsize=12)

    # plot original
    axs[0].imshow(original[0, center_z], cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')

    # plot augmented
    axs[1].imshow(augmented[0, center_z], cmap='gray')
    axs[1].set_title('Augmented')
    axs[1].axis('off')
    
    # format and show plot
    plt.tight_layout()
    plt.show()


# function to log image intensity histogram
def log_intensity_histogram(img_tensor, prefix, logger, step_or_epoch):

    # flatten image
    img_flat = img_tensor.detach().cpu().numpy().flatten()
    logger.experiment.log({
        f'{prefix} Intensity histogram (epoch {step_or_epoch})': wandb.Histogram(img_flat)
    })


# function to log images to wandb table
def log_images_to_wandb_table(logger, originals, maskeds, student_preds, prefix, step_or_epoch, max_rows=4, log_histograms=True):

    # create table
    table = wandb.Table(columns=['Original', 'Masked', 'Student'])

    # loop through number of entries to plot and get images for plotting
    for i in range(min(max_rows, originals.shape[0])):
        center_z = originals.shape[2] // 2
        original_img = originals[i, 0].cpu().numpy()[center_z]
        masked_img = maskeds[i, 0].cpu().numpy()[center_z]
        pred_img = student_preds[i, 0].detach().cpu().numpy()[center_z]

        # log intensity histogram for first sample in each batch
        if i == 0 and log_histograms:
            log_intensity_histogram(originals[i, 0], f'{prefix} Original', logger, step_or_epoch)
            log_intensity_histogram(maskeds[i, 0], f'{prefix} Masked', logger, step_or_epoch)
            log_intensity_histogram(student_preds[i, 0], f'{prefix} Student', logger, step_or_epoch)

        # create table
        table.add_data(
            wandb.Image(original_img, caption=f'Original {i}'),
            wandb.Image(masked_img, caption=f'Masked {i}'),
            wandb.Image(pred_img, caption=f'Predicted {i}')
        )
    
    # log table
    logger.experiment.log({f'{prefix} Samples (epoch {step_or_epoch})': table})

