#!/usr/bin/env python3
# PURPOSE: Merge sweep params into UNet base YAML, pin 2×GPU per run,
# and direct all outputs under /ministorage/adina/pretrain_sweep_unet.

import os
import subprocess
import sys
import uuid
import yaml
from copy import deepcopy

BASE_CONFIG = os.environ.get(
    "BASE_CONFIG",
    "/home/ads4015/ssl_project/configs/all_datasets_clip_pretrain_2_config_unet.yaml"
)

TRAIN_SCRIPT = "/home/ads4015/ssl_project/src/all_datasets_clip_pretrain_unet.py"

BASE_OUT = os.environ.get("PRETRAIN_SWEEP_DIR", "/ministorage/adina/pretrain_sweep_unet")

for d in ("checkpoints", "configs", "wandb", "logs", "tmp"):
    os.makedirs(os.path.join(BASE_OUT, d), exist_ok=True)

os.environ.setdefault("WANDB_DIR", os.path.join(BASE_OUT, "wandb"))

def set_by_dotted(d, dotted_key, value):
    keys = dotted_key.split(".")
    ref = d
    for k in keys[:-1]:
        if k not in ref or not isinstance(ref[k], dict):
            ref[k] = {}
        ref = ref[k]
    ref[keys[-1]] = value

def parse_cli_overrides(argv):
    out = {}
    for arg in argv:
        if not arg.startswith("--"):
            continue
        if "=" in arg:
            k, v = arg[2:].split("=", 1)
            out[k] = v
    return out

if __name__ == "__main__":
    with open(BASE_CONFIG, "r") as f:
        base_cfg = yaml.safe_load(f)

    run_id = os.environ.get("WANDB_RUN_ID") or str(uuid.uuid4())

    merged = deepcopy(base_cfg)
    overrides = parse_cli_overrides(sys.argv[1:])
    for k, v in overrides.items():
        try:
            v_yaml = yaml.safe_load(v)
        except Exception:
            v_yaml = v
        set_by_dotted(merged, k, v_yaml)

    # invariants across the sweep (same spirit as Swin)
    merged.setdefault("training", {})
    merged["training"]["seed"] = 100

    merged.setdefault("data", {})
    merged["data"]["train_frac"] = 0.9
    merged["data"]["data_subset_frac"] = 1.0
    merged["data"]["use_sub_patches"] = False
    merged["data"]["base_patch_size"] = 96

    merged.setdefault("data", {}).setdefault("downsample", {})
    if merged["data"]["downsample"].get("enabled", True):
        merged["data"]["downsample"]["target_size"] = 64

    # force 2-GPU DDP per run
    merged.setdefault("dist", {})
    merged["dist"]["multi_gpu"] = True
    merged["dist"]["devices"] = 2
    merged["dist"]["num_nodes"] = 1
    merged["dist"]["strategy"] = "ddp"

    # per-run output dirs
    ckpt_dir = os.path.join(BASE_OUT, "checkpoints", run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    merged.setdefault("model", {})
    merged["model"]["save_dirpath"] = ckpt_dir
    merged["model"]["save_filename"] = (
        "all_datasets_clip_pretrained-unet-epoch{epoch:03d}-val-report{val_loss_report:.4f}-step{step}"
    )

    merged_cfg_path = os.path.join(BASE_OUT, "configs", f"{run_id}.yaml")
    with open(merged_cfg_path, "w") as f:
        yaml.safe_dump(merged, f, sort_keys=False)

    tmp_path = os.path.join(BASE_OUT, "tmp", f"merged_{run_id}.yaml")
    with open(tmp_path, "w") as f:
        yaml.safe_dump(merged, f, sort_keys=False)

    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        TRAIN_SCRIPT,
        "--config", tmp_path,
    ]
    print(f"[INFO] Launching: {' '.join(cmd)}", flush=True)
    rc = subprocess.call(cmd)
    sys.exit(rc)