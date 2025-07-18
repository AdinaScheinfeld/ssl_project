# Inference script 

# --- Setup ---

# imports
import argparse
from monai.networks.nets import SwinUNETR
import nibabel as nib
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F


# --- Functions ---

# function to pad 3d tensor to be divisible by a factor
def pad_to_divisible(tensor, factor=32):

    # get tensor shape
    C, D, H, W = tensor.shape

    # get padding sizes
    pad_d = (factor - D % factor) % factor
    pad_h = (factor - H % factor) % factor
    pad_w = (factor - W % factor) % factor
    padding = (0, pad_w, 0, pad_h, 0, pad_d)  # (W_left, W_right, H_left, H_right, D_left, D_right)

    # pad tensor
    padded_tensor = F.pad(tensor, padding, mode="constant", value=0)

    # return padded tensor and original shape
    return padded_tensor, (D, H, W)


# function to crop tensor to original shape
def crop_to_shape(tensor, shape):

    # get tensor shape
    B, C, D, H, W = tensor.shape

    # get original shape
    d, h, w = shape

    # return cropped tensor
    return tensor[:, :, :d, :h, :w]


# function to load image from nifti file
def load_image_nii(path, pad_to_channels=2):

    # load image
    img = nib.load(str(path)).get_fdata()

    # ensure correct dimensions
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0) # (1, D, H, W)

    # pad channels if needed (1 channel to 2 channels)
    if img.shape[0] == 1 and pad_to_channels == 2:
        img = np.concatenate([img, np.zeros_like(img)], axis=0)

    # convert to tensor
    tensor = torch.tensor(img, dtype=torch.float32)

    # return padded tensor 
    return pad_to_divisible(tensor)


# function to save mask as nifti file
def save_mask_as_nii(mask_tensor, output_path):

    # convert mask to numpy
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)

    # save as nifti file
    nib.save(nib.Nifti1Image(mask_np, affine=np.eye(4)), str(output_path))


# function to run inference
def run_inference(model, image_tensor, original_shape):

    # set model to eval mode
    model.eval()

    # get device
    device = next(model.parameters()).device

    # run inference
    with torch.no_grad():
        input_tensor = image_tensor.unsqueeze(0).to(device) # add batch dimension -> (1, 2, D, H, W)
        logits = model(input_tensor) # get raw logits (pre-activation output) from model
        prob = torch.sigmoid(logits) # apply sigmoid activation to covner logits to probabilities (0-1 range for binary segmentation))
        mask = (prob > 0.5).long() # convert probabilities to binary mask (0 or 1)
        mask = crop_to_shape(mask, original_shape) # crop padded mask back to orignal shape

    # return mask (1, D, H, W)
    return mask


# function to load model from checkpoint
def load_model(checkpoint_path, device):

    # load model
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=2,
        out_channels=1,
        feature_size=48,
        use_checkpoint=False
    ).to(device)

    # load state dict
    state_dict = torch.load(checkpoint_path, map_location=device)['state_dict']
    model.load_state_dict({k.replace('model.', ''): v for k, v in state_dict.items()}, strict=False)

    # return model
    return model


# function to run inference, process image, and save mask
def process_and_save(model, input_file, output_dir):

    image_tensor, original_shape = load_image_nii(input_file) # load image and save original shape (for future cropping)
    mask = run_inference(model, image_tensor, original_shape) # run inference and crop to original shape
    output_path = output_dir / (input_file.name.replace('.nii.gz', '.nii').replace('.nii', '_pred.nii.gz'))
    save_mask_as_nii(mask, output_path) # save mask as nifti file
    print(f'Saved prediction to {output_path}')
    

# main function 
# supports inference on single file or all files in a directory
def main():

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path to a .nii.gz file or directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to finetuned model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save predictions')
    args = parser.parse_args()

    # get input and output paths
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load model
    model = load_model(args.checkpoint, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # inference on single file
    if input_path.is_file():
        process_and_save(model, input_path, output_dir)

    # inference on all files in directory
    elif input_path.is_dir():
        nii_files = sorted([f for f in input_path.glob('*') if f.suffix in ['.nii', '.gz'] or f.name.endswith('.nii.gz')]) # get all nifti files in directory (.nii or .nii.gz)
        for file in nii_files:
            print(f'Running inference on {file.name}')
            process_and_save(model, file, output_dir)
        print(f'Saved {len(nii_files)} predictions to {output_dir}') # indicate how many files were processed from the directory
    else:
        raise ValueError('input_path must be a file or directory')
    

# --- Main Entry Point ---
if __name__ == '__main__':
    main()  # run the main function to start the inference process


















