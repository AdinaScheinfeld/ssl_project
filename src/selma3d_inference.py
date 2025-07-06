# inference_lsm_finetuned.py

import argparse
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from monai.networks.nets import SwinUNETR


def pad_to_divisible(tensor, factor=32):
    """Pad a 3D (C, D, H, W) tensor so that D, H, W are divisible by `factor`. Returns padded tensor and original shape."""
    _, D, H, W = tensor.shape
    pad_d = (factor - D % factor) % factor
    pad_h = (factor - H % factor) % factor
    pad_w = (factor - W % factor) % factor
    padding = (0, pad_w, 0, pad_h, 0, pad_d)  # (W_left, W_right, H_left, H_right, D_left, D_right)
    padded_tensor = F.pad(tensor, padding, mode="constant", value=0)
    return padded_tensor, (D, H, W)


def crop_to_shape(tensor, shape):
    """Crop tensor to original spatial shape (D, H, W)."""
    _, _, D, H, W = tensor.shape
    d, h, w = shape
    return tensor[:, :, :d, :h, :w]


def load_image_nii(path, pad_to_channels=2):
    """Load a 3D NIfTI image and pad channels and shape if needed."""
    img = nib.load(str(path)).get_fdata()
    if img.ndim == 3:
        img = np.expand_dims(img, axis=0)  # (1, D, H, W)
    if img.shape[0] == 1 and pad_to_channels == 2:
        img = np.concatenate([img, np.zeros_like(img)], axis=0)
    tensor = torch.tensor(img, dtype=torch.float32)
    return pad_to_divisible(tensor)


def save_mask_as_nii(mask_tensor, output_path):
    mask_np = mask_tensor.squeeze().cpu().numpy().astype(np.uint8)
    nib.save(nib.Nifti1Image(mask_np, affine=np.eye(4)), str(output_path))


def run_inference(model, image_tensor, original_shape):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        input_tensor = image_tensor.unsqueeze(0).to(device)  # (1, 2, D, H, W)
        logits = model(input_tensor)
        prob = torch.sigmoid(logits)
        mask = (prob > 0.5).long()
        mask = crop_to_shape(mask, original_shape)
    return mask


def load_model(checkpoint_path, device):
    model = SwinUNETR(
        img_size=(96, 96, 96),  # deprecated but still supported
        in_channels=2,
        out_channels=1,
        feature_size=48,
        use_checkpoint=False
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)["state_dict"]
    model.load_state_dict({k.replace("model.", ""): v for k, v in state_dict.items()}, strict=False)
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True, help="Path to a .nii.gz file or directory")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to fine-tuned model checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save predictions")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = load_model(args.checkpoint, args.device)

    if input_path.is_file():
        image_tensor, shape = load_image_nii(input_path)
        mask = run_inference(model, image_tensor, shape)
        output_path = output_dir / (input_path.stem + "_pred.nii.gz")
        save_mask_as_nii(mask, output_path)
        print(f"Saved prediction to {output_path}")

    elif input_path.is_dir():
        nii_files = sorted(input_path.glob("*.nii.gz"))
        for file in nii_files:
            print(f"Running inference on {file.name}")
            image_tensor, shape = load_image_nii(file)
            mask = run_inference(model, image_tensor, shape)
            output_path = output_dir / (file.stem + "_pred.nii.gz")
            save_mask_as_nii(mask, output_path)
        print(f"Saved {len(nii_files)} predictions to {output_dir}")
    else:
        raise ValueError("input_path must be a file or directory")


if __name__ == "__main__":
    main()
