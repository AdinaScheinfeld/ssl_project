# Functions used in visualization notebooks and scripts

# --- Setup ---

# imports
import matplotlib.pyplot as plt
import numpy as np
import wandb


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

