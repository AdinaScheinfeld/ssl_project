# rewrite_prompts/run_pretrain_wandb_rewrites.py
# PURPOSE: Merge sweep params into base YAML, pin 2×GPU per run, and
#   direct all outputs (checkpoints, configs, local W&B files) under
#   /ministorage/adina/pretrain_sweep_rewrites.

# --- Setup ---

# imports
import os
import re
import subprocess
import sys
import tempfile
import time
import uuid
import yaml
import wandb

from copy import deepcopy


# base config path
BASE_CONFIG = os.environ.get(
    "BASE_CONFIG",
    "/home/ads4015/ssl_project/rewrite_prompts/all_datasets_clip_pretrain_2_config_rewrites.yaml"
)

# training script path
TRAIN_SCRIPT = "/home/ads4015/ssl_project/rewrite_prompts/all_datasets_clip_pretrain_rewrites.py"

# base output directory for all sweep runs
BASE_OUT = os.environ.get("PRETRAIN_SWEEP_DIR", "/ministorage/adina/pretrain_sweep_rewrites")

# ensure the output structure exists
for d in ("checkpoints", "configs", "wandb", "logs", "tmp"):
    os.makedirs(os.path.join(BASE_OUT, d), exist_ok=True)

# point W&B to store local run files under BASE_OUT/wandb
os.environ.setdefault("WANDB_DIR", os.path.join(BASE_OUT, "wandb"))

# --- Functions ---

# set value in nested dict by dotted key
def set_by_dotted(d, dotted_key, value):
    keys = dotted_key.split(".")
    ref = d
    for k in keys[:-1]:
        if k not in ref or not isinstance(ref[k], dict):
            ref[k] = {}
        ref = ref[k]
    ref[keys[-1]] = value

# deep-merge dictionaries (right overrides left)
def deep_merge(a, b):
    out = deepcopy(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out

def parse_cli_overrides(argv):
    """
    Converts ['--a.b=1','--c.d.e=True'] -> {'a.b':'1','c.d.e':'True'}
    Leaves types as strings; we'll pass them through to YAML (which can coerce).
    """
    out = {}
    for arg in argv:
        if not arg.startswith("--"):
            continue
        # accepts --key=value OR --key value
        if "=" in arg:
            k, v = arg[2:].split("=", 1)
            out[k] = v
        else:
            # grab the next token as the value if present
            # (agent uses --k=v form by default, so this is just defensive)
            pass
    return out

# --- Main ---
if __name__ == "__main__":

    # load base YAML
    with open(BASE_CONFIG, "r") as f:
        base_cfg = yaml.safe_load(f)

    # Do NOT init W&B here; let Lightning's WandbLogger own the run.
    # We still need a stable folder name for outputs:
    run_id = os.environ.get("WANDB_RUN_ID") or str(uuid.uuid4())

    # apply sweep overrides (supports both dotted keys and nested dicts)
    merged = deepcopy(base_cfg)

    overrides = parse_cli_overrides(sys.argv[1:])
    for k, v in overrides.items():
        # Let YAML coerce types (ints/bools/floats) from strings
        try:
            v_yaml = yaml.safe_load(v)
        except Exception:
            v_yaml = v
        set_by_dotted(merged, k, v_yaml)

    # enforce fixed invariants across the sweep
    merged["training"]["seed"] = 100
    merged["data"]["train_frac"] = 0.9
    merged["data"]["data_subset_frac"] = 1.0
    merged["data"]["use_sub_patches"] = False
    merged["data"]["base_patch_size"] = 96

    # if downsample is enabled, always target 64
    merged.setdefault("data", {}).setdefault("downsample", {})
    if merged["data"]["downsample"].get("enabled", True):
        merged["data"]["downsample"]["target_size"] = 64

    # force 2-GPU DDP per run so each agent consumes exactly two H100s
    merged.setdefault("dist", {})
    merged["dist"]["multi_gpu"] = True
    merged["dist"]["devices"] = 2
    merged["dist"]["num_nodes"] = 1
    merged["dist"]["strategy"] = "ddp"

    # make a unique subdir per run for checkpoints and the frozen merged YAML
    # run_id = run.id  # stable unique id from W&B
    ckpt_dir = os.path.join(BASE_OUT, "checkpoints", run_id)
    os.makedirs(ckpt_dir, exist_ok=True)

    # point ModelCheckpoint to that folder
    merged.setdefault("model", {})
    merged["model"]["save_dirpath"] = ckpt_dir
    merged["model"]["save_filename"] = (
        "all_datasets_clip_pretrained-rewrites-epoch{epoch:03d}-val{val_loss:.4f}-step{step}"
    )

    # persist the exact merged config used for this run (for reproducibility)
    merged_cfg_path = os.path.join(BASE_OUT, "configs", f"{run_id}.yaml")
    with open(merged_cfg_path, "w") as f:
        yaml.safe_dump(merged, f, sort_keys=False)

    # also write a transient copy for the trainer to read (Lightning only needs a path)
    tmp_path = os.path.join(BASE_OUT, "tmp", f"merged_{run_id}.yaml")
    with open(tmp_path, "w") as f:
        yaml.safe_dump(merged, f, sort_keys=False)

    # launch your existing trainer with the merged config
    cmd = [
        "torchrun",
        "--standalone",
        "--nproc_per_node=2",
        TRAIN_SCRIPT,
        "--config", tmp_path,
    ]
    print(f"[INFO] Launching: {' '.join(cmd)}")
    rc = subprocess.call(cmd)
    sys.exit(rc)






