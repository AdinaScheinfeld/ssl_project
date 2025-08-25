#!/bin/bash
#SBATCH --job-name=ng2nii
#SBATCH --output=logs/ng2nii_%A_%a.out
#SBATCH --error=logs/ng2nii_%A_%a.err
#SBATCH --time=01:00:00
#SBATCH --partition=minilab-cpu
#SBATCH --mem=100G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --array=1-9

set -euo pipefail

# indicate starting job
echo "Starting SLURM job ${SLURM_JOB_ID:-N/A} (array task ${SLURM_ARRAY_TASK_ID:-all})..."

# config
CONFIG_FILE="/midtier/paetzollab/scratch/ads4015/all_wu_brain_patches/all_wu_brain_patches_info.json"
SIZE="96,96,96"
NG2NII="/home/ads4015/ssl_project/preprocess_patches/src/ng2nii.py"

# activate conda
module load anaconda3/2022.10-34zllqw
source activate ng2nii-env1

# optional script argument: --brains "id1,id2" or "1,3-5"
# pass args to script after --
# ex: sbatch ng2nii_array.sh --brains "1-3,12"
BRAINS_ARG=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --brains) BRAINS_ARG="$2"; shift 2 ;;
        --) shift; break ;;
        *) shift ;;
    esac
done
if [[ -n "${BRAINS_ARG}" ]]; then export BRAINS="${BRAINS_ARG}"; fi

# expose config to python
export CONFIG_FILE SIZE NG2NII

python - <<'PY'
import json, os, re, subprocess, sys

config_path = os.environ["CONFIG_FILE"]
size = os.environ["SIZE"]
ng2nii = os.environ["NG2NII"]
array_task = os.environ.get("SLURM_ARRAY_TASK_ID")
brains_select = os.environ.get("BRAINS", "").strip()

with open(config_path, "r") as f:
    config = json.load(f)

# index brains 1-based
id_to_brain = {b["id"]: b for b in config}
idx_to_brain = {i+1: b for i, b in enumerate(config)}

# function to return ordered list of selected brains from a selection string
# selection supports exact IDs (ex: 01_AA1-PO-C-R45), indices (ex: 1,3,7), ranges (ex: 1-4)
# mixed is supported (ex: 1-2, 07_AZ...,13_AA1-...)
def parse_selection(sel: str):

    # if no selection, return all
    if not sel:
        return config[:]

    # create list and set of selections
    picked = []
    seen_ids = set()

    tokens = [t.strip() for t in sel.split(",") if t.strip()]
    for tok in tokens:

        # numeric or numeric range
        if re.fullmatch(r"\d+(-\d+)?", tok):
            if "-" in tok:
                a, b = map(int, tok.split("-"))
                if a > b:
                    a, b = b, a
                for i in range(a, b+1):
                    bobj = idx_to_brain.get(i)
                    if not bobj:
                        raise SystemExit(f"[ERROR] Index {i} out of range (1..{len(config)})")
                    if bobj["id"] not in seen_ids:
                        picked.append(bobj); seen_ids.add(bobj["id"])

            else:
                i = int(tok)
                bobj = idx_to_brain.get(i)
                if not bobj:
                    raise SystemExit(f"[ERROR] Index {i} out of range (1..{len(config)})")
                if bobj["id"] not in seen_ids:
                    picked.append(bobj); seen_ids.add(bobj["id"])

        # id
        else:
            bobj = id_to_brain.get(tok)
            if not bobj:
                candidates = [b["id"] for b in config if tok in b["id"]]
                if candidates:
                    raise SystemExit(f"[ERROR] ID '{tok}' not found. Did you mean one of: {candidates}")
                raise SystemExit(f"[ERROR] ID '{tok}' not found in config.")
            if bobj["id"] not in seen_ids:
                picked.append(bobj); seen_ids.add(bobj["id"])

    return picked

selected = parse_selection(brains_select)
print("[INFO] Selected brains:", [b["id"] for b in selected], flush=True)

def run_brain(brain):

    bid = brain["id"]
    vol_path = brain["vol_path"]
    folder = brain["folder"]
    coords = brain["coords"]
    os.makedirs(folder, exist_ok=True)

    print(f"[INFO] Processing brain: {bid} (#patches={len(coords)})", flush=True)
    for i, coord in enumerate(coords, start=1):
        suffix = f"_p{i:02d}"
        print(f"  -> {bid}: coord='{coord}' suffix='{suffix}'", flush=True)
        subprocess.run(
            [
                "python", ng2nii,
                "--vol_path", vol_path,
                "--coord_input", coord,
                "--folder", folder,
                "--size", size,
                "--suffix", suffix
            ],
            check=True
        )

# sequential mode over selection
if array_task is None:
    for brain in selected:
        run_brain(brain)

# array mode
else:
    idx = int(array_task)
    if idx < 1 or idx > len(selected):
        print(f"[INFO] Array task {idx} has no selection (selected={len(selected)}). Exiting.", flush=True)
        sys.exit(0)
    run_brain(selected[idx-1])

print("[INFO] Done.", flush=True)
PY

echo "ng2nii job complete."









