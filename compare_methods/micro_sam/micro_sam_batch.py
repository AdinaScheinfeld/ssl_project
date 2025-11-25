# /home/ads4015/ssl_project/compare_methods/micro_sam/micro_sam_batch.py

#!/usr/bin/env python

import os
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from micro_sam.automatic_segmentation import (
    get_predictor_and_segmenter,
    automatic_instance_segmentation,
)


def log(msg: str):
    """Simple logger with flush so Slurm output appears immediately."""
    print(msg, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run microSAM automatic 3D instance segmentation on a batch of NIfTI patches."
    )
    parser.add_argument(
        "--file_list",
        type=str,
        required=True,
        help="Path to a text file with one input NIfTI path per line.",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        required=True,
        help="Root directory to write predictions, e.g. /.../compare_methods/micro_sam.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="vit_b_lm",
        help="microSAM model type (default: vit_b_lm).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run on (default: auto).",
    )
    parser.add_argument(
        "--per_task",
        type=int,
        default=5,
        help="Number of images to process per Slurm array task (default: 5).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="If set, overwrite existing prediction files.",
    )
    return parser.parse_args()


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    vol = vol.astype(np.float32)
    vmin, vmax = vol.min(), vol.max()
    if vmax > vmin:
        vol = (vol - vmin) / (vmax - vmin)
    return vol


def main():
    args = parse_args()

    log("====================================================")
    log("=== run_microsam_batch.py: microSAM segmentation ===")
    log("====================================================")
    log(f"Args: {args}")

    file_list_path = Path(args.file_list)
    if not file_list_path.is_file():
        raise FileNotFoundError(f"file_list not found: {file_list_path}")

    log(f"Reading file list from: {file_list_path}")
    with file_list_path.open("r") as f:
        all_paths = [Path(line.strip()) for line in f if line.strip()]

    total_files = len(all_paths)
    log(f"Total files in file_list: {total_files}")

    if not all_paths:
        log("No input files in file_list. Exiting.")
        return

    # Figure out which subset this array task should process
    task_id_str = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    try:
        task_id = int(task_id_str)
    except ValueError:
        log(f"WARNING: SLURM_ARRAY_TASK_ID='{task_id_str}' is not an int. Using 0.")
        task_id = 0

    per_task = max(args.per_task, 1)
    start = task_id * per_task
    end = min(start + per_task, total_files)

    log(f"[Task {task_id}] per_task={per_task}, start={start}, end={end}")

    if start >= total_files:
        log(f"[Task {task_id}] No files to process (start={start} >= total={total_files}). Exiting.")
        return

    subset = all_paths[start:end]
    log(f"[Task {task_id}] Number of files assigned to this task: {len(subset)}")

    # Device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    log(f"[Task {task_id}] Using device: {device}")

    # Load microSAM model once per task
    log(f"[Task {task_id}] Loading microSAM model: {args.model_type}")
    predictor, segmenter = get_predictor_and_segmenter(
        model_type=args.model_type,
        device=device,
        amg=None,
    )
    log(f"[Task {task_id}] microSAM model loaded successfully.")
    log(f"[Task {task_id}] Starting microSAM segmentation for {len(subset)} volumes...")

    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    log(f"[Task {task_id}] Output root: {output_root}")

    processed = 0
    skipped_existing = 0
    skipped_missing = 0
    failed = 0

    for index_within_task, ip in enumerate(subset):
        if not ip.is_file():
            log(f"[Task {task_id}] WARNING: input not found, skipping: {ip}")
            skipped_missing += 1
            continue

        class_name = ip.parent.name
        out_dir = output_root / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / ip.name.replace(".nii.gz", "_micro_sam_instances.nii.gz")

        if out_path.exists() and not args.overwrite:
            log(f"[Task {task_id}] [{index_within_task+1}/{len(subset)}] Output exists, skipping: {out_path}")
            skipped_existing += 1
            continue

        log(f"[Task {task_id}] [{index_within_task+1}/{len(subset)}] Starting microSAM segmentation for: {ip}")

        try:
            # Load image
            img_nii = nib.load(str(ip))
            img = img_nii.get_fdata()

            log(
                f"[Task {task_id}]    Loaded volume: shape={img.shape}, "
                f"dtype={img.dtype}, min={float(img.min())}, max={float(img.max())}"
            )

            if img.ndim != 3:
                raise ValueError(f"Expected 3D patch, got shape {img.shape} for {ip}")

            img_norm = normalize_volume(img)
            log(
                f"[Task {task_id}]    Normalized volume to [0,1]: "
                f"min={float(img_norm.min())}, max={float(img_norm.max())}"
            )

            # Run microSAM
            log(f"[Task {task_id}]    Calling automatic_instance_segmentation(...)")
            try:
                instances = automatic_instance_segmentation(
                    predictor=predictor,
                    segmenter=segmenter,
                    input_path=img_norm,
                    ndim=3,
                    tile_shape=None,
                    halo=None,
                    verbose=True,
                    return_embeddings=False,
                )
                log(f"[Task {task_id}]    microSAM inference done, got instances with shape {instances.shape}")
            except TypeError as e:
                # Handle the specific nifty insertEdges issue (empty float64 edge array)
                if "insertEdges()" in repr(e):
                    log(
                        f"[Task {task_id}]    WARNING: microSAM / nifty insertEdges() TypeError "
                        f"(empty edge list). Treating as 'no instances' and writing zeros."
                    )
                    instances = np.zeros_like(img_norm, dtype=np.uint32)
                else:
                    raise  # re-raise other TypeErrors

            # Ensure uint32 and save
            instances = instances.astype(np.uint32, copy=False)
            inst_nii = nib.Nifti1Image(instances, affine=img_nii.affine, header=img_nii.header)
            nib.save(inst_nii, str(out_path))

            log(f"[Task {task_id}]    Saved prediction to: {out_path}")
            processed += 1

        except Exception as e:
            log(f"[Task {task_id}]    ERROR while processing {ip}: {repr(e)}")
            failed += 1


    log(f"[Task {task_id}] === Task summary ===")
    log(f"[Task {task_id}]   processed:        {processed}")
    log(f"[Task {task_id}]   skipped_existing: {skipped_existing}")
    log(f"[Task {task_id}]   skipped_missing:  {skipped_missing}")
    log(f"[Task {task_id}]   failed:           {failed}")
    log(f"[Task {task_id}] === run_microsam_batch.py finished ===")


if __name__ == "__main__":
    main()







