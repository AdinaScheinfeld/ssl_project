#!/bin/bash
#SBATCH --job-name=rewrite_prompts
#SBATCH --output=logs/rewrite_prompts_%j.out
#SBATCH --error=logs/rewrite_prompts_%j.err
#SBATCH --partition=minilab-gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1

# rewrite_text_prompts_job.sh - Script to run text prompt rewriting using an LLM

# indicate starting
echo "[INFO] Starting rewrite job at $(date)"

# activate conda env
module load anaconda3/2022.10-34zllqw
source activate monai-env1

# put HF cache somewhere with space
export HF_HOME=/ministorage/adina/huggingface
mkdir -p "$HF_HOME"

# get paths
OUT_DIR="/home/ads4015/ssl_project/rewrite_prompts"
mkdir -p "$OUT_DIR"

# input JSONs (from yaml data config)
INP_CONN="/home/ads4015/ssl_project/data/text_prompts_allen_connection.json"
INP_DEV="/home/ads4015/ssl_project/data/text_prompts_allen_dev_mouse.json"
INP_HUM2="/home/ads4015/ssl_project/data/text_prompts_allen_human2.json"
INP_SELMA="/home/ads4015/ssl_project/data/text_prompts_selma.json"
INP_WU="/home/ads4015/ssl_project/data/text_prompts_wu.json"

# output merged+expanded JSON
OUT_JSON="$OUT_DIR/text_prompts_expanded.json"

# LLM to use ## UP TO HERE
MODEL_ID="mistralai/Mistral-7B-Instruct-v0.2"

# Generation settings
K_REWRITES=6
SIM_THRESH=0.80
TEMP=0.2
TOPP=0.9
MAX_NEW=96

# run rewrite script
echo "[INFO] Running rewrite script with model: ${MODEL_ID}"

srun --label python "/home/ads4015/ssl_project/rewrite_prompts/rewrite_text_prompts.py" \
  --input_jsons "$INP_CONN" "$INP_DEV" "$INP_HUM2" "$INP_SELMA" "$INP_WU" \
  --output_json "$OUT_JSON" \
  --llm_name "$MODEL_ID" \
  --k "$K_REWRITES" \
  --sim_thresh "$SIM_THRESH" \
  --temperature "$TEMP" \
  --top_p "$TOPP" \
  --max_new_tokens "$MAX_NEW"

STATUS=$?
if [[ $STATUS -ne 0 ]]; then
  echo "[ERROR] rewrite_text_prompts.py failed with status $STATUS"
  exit $STATUS
fi

# print a few keys and 1–2 rewrites to stdout as sanity check
echo "[INFO] Sanity check on $OUT_JSON"

# pass the OUT_JSON path to the inline Python
export OUT_JSON_PATH="$OUT_JSON"

srun --label python - <<'PYCODE'
import json, random, os
p = os.environ.get("OUT_JSON_PATH", "/home/ads4015/ssl_project/rewrite_prompts/text_prompts_expanded.json")
with open(p, "r") as f:
    d = json.load(f)

keys = list(d.keys())[:8]  # show a few
print(f"[CHECK] Showing up to 8 keys from {p}:")
for k in keys:
    entry = d[k]
    orig = entry.get("orig", "")
    rws = entry.get("rewrites", [])
    print(f"\nKEY: {k}")
    print(f"  ORIG: {orig[:180]}{'...' if len(orig)>180 else ''}")
    if rws:
        print(f"  RW1 : {rws[0][:180]}{'...' if len(rws[0])>180 else ''}")
        if len(rws) > 1:
            print(f"  RW2 : {rws[1][:180]}{'...' if len(rws[1])>180 else ''}")
    else:
        print("  (no rewrites kept by filters)")
PYCODE

echo "[INFO] Done at $(date)"
















