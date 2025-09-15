# all_datasets_inference.py - Script to run inference on a single 3D medical image or all images in a directory 

# --- Setup ---

# imports
import argparse
from monai.inferers import sliding_window_inference
from monai.networks.nets import SwinUNETR
import nibabel as nib
import numpy as np
from pathlib import Path
import tifffile as tiff
import torch
import torch.nn.functional as F


# --- Functions ---

# function to pad 3d tensor to be divisible by a factor
def pad_to_divisible(tensor, factor=32):

    # get tensor shape
    B, C, D, H, W = tensor.shape

    # get padding sizes
    pad_d = (factor - D % factor) % factor
    pad_h = (factor - H % factor) % factor
    pad_w = (factor - W % factor) % factor
    padding = (0, pad_w, 0, pad_h, 0, pad_d)  # (W_left, W_right, H_left, H_right, D_left, D_right)

    # pad tensor
    padded_tensor = F.pad(tensor, padding, mode="constant", value=0.0)

    # return padded tensor and original shape
    return padded_tensor, (D, H, W)


# function to crop tensor to original shape
def crop_to_shape(tensor, shape):

    # get tensor shape
    B, C, D, H, W = tensor.shape

    # get original shape
    d, h, w = shape

    # return cropped tensor
    return tensor[..., :d, :h, :w]


# function to load image from nifti file
def load_nii(path):

    # load nifti file
    nii = nib.load(str(path))

    # get image data as numpy array
    arr = nii.get_fdata(dtype=np.float32)

    # return array, affine, and header
    return arr, nii.affine, nii.header


# function to load tif file
def load_tiff_file(path):

    arr = tiff.imread(str(path)).astype(np.float32) # support 3d or 4d tiff
    
    # ensure single 3d volume (Z, Y, X) or (Z, Y, X, C)
    if arr.ndim not in (3, 4):
        raise ValueError(f'Unsupported ndim={arr.ndim} for {path}; expected 3d or 4d array')

    # use identity affine 
    affine = np.eye(4, dtype=np.float32)
    header = nib.Nifti1Header() # empty header
    
    return arr, affine, header


# function to load any image file (nifti or tiff)
def load_image_any(path):

    suf = path.suffix.lower()

    if suf == '.gz' and path.name.endswith('.nii.gz'):
        return load_nii(path)
    
    if suf == '.nii':
        return load_nii(path)
    
    if suf in ('.tif', '.tiff'):
        return load_tiff_file(path)
    
    raise ValueError(f'Unsupported file format: {suf}; supported: .nii, .nii.gz, .tif, .tiff')


# function to ensure channel first format and pad channels if needed
def ensure_channel_first(arr, desired_in_channels=2, channel_strategy='duplicate'):

    # get image as array
    arr = np.asarray(arr)

    # if 3D (D, H, W) add channel dimension in front
    if arr.ndim == 3:
        vol = arr[None, ...] # (1, D, H, W)

    # if 4D (D,H,W,C) move channel dimension to front
    elif arr.ndim == 4:
        vol = np.moveaxis(arr, -1, 0) # (C, D, H, W)

    else:
        raise ValueError(f'Unsupported ndim={arr.ndim}; expected 3d or 4d array')
    
    # normalize per volume
    vmin, vmax = np.percentile(vol, (1, 99))
    if vmax > vmin:
        vol = np.clip((vol - vmin) / (vmax - vmin), 0.0, 1.0)
    else:
        vol = np.zeros_like(vol)

    # adjust channels if needed
    if vol.shape[0] == desired_in_channels:
        return torch.from_numpy(vol) # already correct number of channels
    if vol.shape[0] == 1 and desired_in_channels == 2:
        if channel_strategy == 'duplicate':
            vol = np.concatenate([vol, vol], axis=0) # duplicate input channel to create 2 channels (2, D, H, W)
        else:
            vol = np.concatenate([vol, np.zeros_like(vol)], axis=0) # add zero channel to create 2 channels (2, D, H, W)

    elif vol.shape[0] > desired_in_channels:
        vol = vol[:desired_in_channels] # take first desired_in_channels channels
    elif vol.shape[0] < desired_in_channels:
        pad = [np.zeros_like(vol)] * (desired_in_channels - vol.shape[0])
        vol = np.concatenate([vol] + pad, axis=0) # pad with zero channels to desired_in_channels

    return torch.from_numpy(vol) # (C, D, H, W)
    

# function to save mask as nifti file
def save_mask_as_nii(mask_tensor, output_path, affine, header):

    # convert mask to numpy
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)

    # save as nifti file
    nii = nib.Nifti1Image(mask_np, affine=affine, header=header)
    nib.save(nii, str(output_path))


# function to save mask as tiff file
def save_mask_as_tiff(mask_tensor, output_path):

    # convert mask to numpy
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8) # (D, H, W) or (H, W)

    axes = 'ZYX' if mask_np.ndim == 3 else 'YX'

    # bigtiff if the array is >4GB
    bigtiff = mask_np.nbytes > (2**32 - 1)
    tiff.imwrite(
        str(output_path), 
        mask_np, 
        photometric='minisblack', 
        metadata={'axes': axes},
        ome=True,
        compression='zlib',
        bigtiff=bigtiff
    )


# function to run inference
def run_inference(model, image_tensor, pad_factor=32):

    # set model to eval mode
    model.eval()

    # get device
    device = next(model.parameters()).device

    # run inference
    with torch.no_grad():

        # add batch and pad to multiples of pad_factor
        x = image_tensor.unsqueeze(0) # (1, C, D, H, W)
        x, original_shape = pad_to_divisible(x, factor=pad_factor) # pad to be divisible by pad_factor
        x = x.to(device)

        logits = sliding_window_inference(inputs=x, 
                                         roi_size=(96, 96, 96), 
                                         sw_batch_size=1, 
                                         predictor=model, 
                                         overlap=0.5
                                        ) # (1, 1, D, H, W)
        prob = torch.sigmoid(logits) # apply sigmoid activation to covner logits to probabilities (0-1 range for binary segmentation))
        mask = (prob > 0.5).to(torch.uint8) # convert probabilities to binary mask (0 or 1)
        mask = crop_to_shape(mask, original_shape) # crop padded mask back to orignal shape

    # return mask (1, 1, D, H, W)
    return mask


# function to infer hparams from state dict
def _infer_hparams_from_state_dict(sd):

    # try lighting hparams
    hp = {}
    if 'hyper_parameters' in sd:
        hp = sd['hyper_parameters']
    elif 'hparams' in sd:
        hp = sd['hparams']

    # get in channels and feature size
    in_ch = hp.get('in_channels')
    feat_sz = hp.get('feature_size')

    # fallback to inferring from tensor shapes
    if in_ch is None:

        # patch_embed.proj.weight: [embed_dim, in_channels, 2, 2, 2]
        for k, v in sd.get('state_dict', sd).items():
            if k.endswith('swinViT.patch_embed.proj.weight'):
                in_ch = v.shape[1]
                break
    
    if feat_sz is None:

        # earliest encoder conv: [out_channels, in_channels, 3, 3, 3]
        for k, v in sd.get('state_dict', sd).items():
            if k.endswith('encoder1.layer.conv1.conv.weight'):
                feat_sz = v.shape[0]
                break
    
    return in_ch or 1, feat_sz or 24



# function to load model from checkpoint
def load_model(checkpoint_path, device):

    # get checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    in_channels, feature_size = _infer_hparams_from_state_dict(ckpt)

    # load model
    model = SwinUNETR(
        img_size=(96, 96, 96),
        in_channels=in_channels,
        out_channels=1,
        feature_size=feature_size,
        use_checkpoint=False
    ).to(device)

    state_dict = ckpt.get('state_dict', ckpt)  # support both with and without 'state_dict' key
    cleaned = {k.replace('model.', '').replace('module.', ''): v for k, v in state_dict.items()}  # remove 'model.' and 'module.' prefixes if present

    # load strictly
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing: print(f'[WARN] Missing keys in state_dict: {missing}', flush=True)
    if unexpected: print(f'[WARN] Unexpected keys in state_dict: {unexpected}', flush=True)

    # expose in_channels for preprocessing
    model._in_channels = in_channels

    # return model
    return model


# function to run inference, process image, and save mask
def process_and_save(model, input_file, output_dir, args):

    # load image and pad to 2 channels if needed
    arr, affine, header = load_image_any(input_file)

    # get in_channels from model or default to 1 if not set
    desired = getattr(model, '_in_channels', 1)

    # convert to channel first tensor and normalize
    vol = ensure_channel_first(arr, desired_in_channels=desired, channel_strategy='duplicate') # (C, D, H, W)

    # inference (handles padding too)
    mask = run_inference(model, vol, pad_factor=32) # (1, 1, D, H, W)

    # save
    name = input_file.name
    if name.endswith('.nii.gz'):
        base, in_ext = name[:-7], '.nii'
    elif name.endswith('.nii'):
        base, in_ext = name[:-4], '.nii'
    else:
        base, in_ext = input_file.stem, input_file.suffix.lower().lstrip('.')

    # select output format
    if args.save_as == 'auto':
        out_fmt = 'tiff' if in_ext in ('tif', 'tiff') else 'nii'
    else:
        out_fmt = args.save_as

    # append name suffix if provided
    suffix = args.name_suffix or ''
    if suffix and not suffix.startswith('_'):
        suffix = '_' + suffix

    if out_fmt == 'tiff':
        out_path = output_dir / f'{base}{suffix}_pred.tiff' # output path
        save_mask_as_tiff(mask, out_path) # save mask as tiff file

    else:
        out_path = output_dir / f'{base}{suffix}_pred.nii.gz' # output path
        save_mask_as_nii(mask, out_path, affine, header) # save mask as nifti file

    print(f'Saved prediction to {out_path}', flush=True)
    

# main function 
# supports inference on single file or all files in a directory
def main():

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, required=True, help='Path to a .nii/.nii.gz/.tif/.tiff file or directory')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to finetuned model checkpoint')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save predictions')
    parser.add_argument('--save_as', choices=['auto', 'nii', 'tiff'], default='auto', help='Format to save predictions (default: auto-detect from input file)')
    parser.add_argument('--name_suffix', type=str, default='', help="Optional text appended to the output base name (before _pred). Ex: '_new', '_1', '_v2'")
    args = parser.parse_args()

    # get input and output paths
    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # load model
    model = load_model(args.checkpoint, torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # inference on single file
    if input_path.is_file():
        process_and_save(model, input_path, output_dir, args)

    # inference on all files in directory
    elif input_path.is_dir():
        files = sorted([
            f for f in input_path.glob('*')
            if f.name.endswith('.nii') or f.name.endswith('.nii.gz')
            or f.suffix.lower() in ('.tif', '.tiff')
        ])
        for file in files:
            print(f'Running inference on {file.name}', flush=True)
            process_and_save(model, file, output_dir, args)
        print(f'Saved {len(files)} predictions to {output_dir}', flush=True)
    

# --- Main Entry Point ---
if __name__ == '__main__':
    main()  # run the main function to start the inference process


















