# /home/ads4015/ssl_project/streamlit/segmentation_samples_streamlit_app.py - Streamlit app to show segmentation prediction samples for human eval

# --- Setup ---

# imports
import argparse
from collections import Counter
from datetime import datetime
import hashlib
from io import BytesIO
import json
# import matplotlib.pyplot as plt
# import nibabel as nib
import pandas as pd
from pathlib import Path
import random
import re
import requests
import streamlit as st


# --- Variable Definitions ---

MODELS = ["image_clip", "image_only", "random"]
LABELS = ["A", "B", "C"]

# map model name -> URL column name in the CSV
MODEL_TO_URLCOL = {
    "image_clip": "pred_image_clip_url",
    "image_only": "pred_image_only_url",
    "random": "pred_random_url",
}


# --- Helper Functions ---

# # function to load a 2D slice from a 3D volume
# @st.cache_data(show_spinner=False)
# def load_slice(path_str: str, z: int):
#     vol = nib.load(path_str).get_fdata()
#     return vol[:, :, z]

# # function to show a 2D slice in streamlit
# def show_slice(img2d, title: str):
#     fig, ax = plt.subplots()
#     ax.imshow(img2d.T, cmap="gray", origin="lower")
#     ax.set_title(title)
#     ax.axis("off")
#     st.pyplot(fig, clear_figure=True)




def _normalize_drive_url(url: str) -> str:
    """
    Accepts any of these:
      - https://drive.google.com/file/d/<ID>/view?...
      - https://drive.google.com/open?id=<ID>
      - https://drive.google.com/uc?id=<ID>...
    Returns a direct download URL that serves image bytes:
      - https://drive.google.com/uc?export=download&id=<ID>
    """
    url = (url or "").strip()
    if not url:
        return ""
    # Already a direct "uc" style
    m = re.search(r"[?&]id=([^&]+)", url)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    # /file/d/<ID>/...
    m = re.search(r"/file/d/([^/]+)/", url)
    if m:
        file_id = m.group(1)
        return f"https://drive.google.com/uc?export=download&id={file_id}"

    # If it's already some other public URL, return as-is
    return url

@st.cache_data(show_spinner=False)
def _fetch_image_bytes(url: str) -> bytes:
    """
    Fetch image bytes server-side. Works around Drive redirects and "view pages".
    """
    if not url:
        return b""
    direct = _normalize_drive_url(url)
    r = requests.get(direct, allow_redirects=True, timeout=30)
    r.raise_for_status()
    return r.content


def show_image_url(url: str, title: str):
    # Always coerce to string (prevents '0' / NaN weirdness)
    url = "" if url is None else str(url).strip()
    if not url or url.lower() in {"nan", "none", "0", "0.0"}:
        st.error(f"Missing URL for: {title}")
        return
    try:
        img_bytes = _fetch_image_bytes(url)
        if not img_bytes:
            st.error(f"Empty image bytes for: {title}")
            st.caption(url)
            return
        st.image(BytesIO(img_bytes), caption=title, use_container_width=True)
    except Exception as e:
        st.error(f"Failed to load {title}: {e}")
        st.caption(url)





# function to get deterministic mapping of labels to models per sample
def deterministic_mapping(sample_id: str, seed: int):
    h = hashlib.md5(f"{seed}:{sample_id}".encode("utf-8")).hexdigest()
    rng = random.Random(int(h[:8], 16))
    models = MODELS.copy()
    rng.shuffle(models)
    return dict(zip(LABELS, models))


# --- Main App ---

# main function
def main():

    # parse args
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=Path, default=Path("segmentation_samples_urls.csv"))
    ap.add_argument("--out_json", type=Path, default=Path("segmentation_results.json"))
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--user_id", type=str, default="anon")
    args = ap.parse_args()

    # set random seed
    random.seed(args.seed)

    # Important: keep URL columns as strings (avoid NaN -> float -> 0/0.0 issues)
    df = pd.read_csv(args.data_csv, dtype=str, keep_default_na=False)

    # app title
    st.title("Segmentation Prediction Ranking")

    # instructions to display in app
    st.caption("Rank A/B/C best → worst. Model identities are hidden.")

    # session state init
    if "idx" not in st.session_state:
        st.session_state.idx = 0
    if "results" not in st.session_state:
        st.session_state.results = []

    if st.session_state.idx >= len(df):
        st.success("Done — thank you!")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # If user provided "something.json", convert to "something_<user>_<stamp>.json"
        out_path = args.out_json
        if out_path.suffix.lower() == ".json":
            out_path = out_path.with_name(f"{out_path.stem}_{args.user_id}_{stamp}.json")
        else:
            # If they pass a directory or no suffix, create a file inside it
            out_path = out_path / f"segmentation_eval_results_{args.user_id}_{stamp}.json"

        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(st.session_state.results, f, indent=2)
        st.write("Saved results to:", str(out_path))
        st.stop()

    # get current row
    row = df.iloc[st.session_state.idx]

    # info display
    st.markdown(f"### Slice {st.session_state.idx + 1} / {len(df)}")
    st.write(f"**Datatype:** {row.datatype} | **Sample:** {row.sample_id} | **z:** {int(row.z)}")

    # --------------------
    # STABLE RANDOMIZE (per slice, deterministic)
    # --------------------
    if "mappings" not in st.session_state:
        st.session_state.mappings = {}

    if st.session_state.idx not in st.session_state.mappings:
        st.session_state.mappings[st.session_state.idx] = deterministic_mapping(
            str(row.sample_id), args.seed
        )

    mapping = st.session_state.mappings[st.session_state.idx]


    # --------------------
    # DISPLAY IMAGE + PREDS
    # --------------------
    st.markdown("### Reference image + predictions")

    # ref = load_slice(row["image_path"], int(row.z))

    c0, c1, c2, c3 = st.columns(4)

    with c0:
        # show_slice(ref, "Image")
        show_image_url(row["image_url"], "Image")

        if st.checkbox("Show ground truth", key=f"show_gt_{st.session_state.idx}"):
            # gt = load_slice(row["gt_path"], int(row.z))
            # show_slice(gt, "Ground truth")
            show_image_url(row["gt_url"], "Ground truth")

    for col, label in zip([c1, c2, c3], LABELS):
        model = mapping[label]
        # pred = load_slice(row[f"{model}_path"], int(row.z))
        url_col = MODEL_TO_URLCOL[model]
        pred_url = row[url_col]

        with col:
            # show_slice(pred, f"Prediction {label}")
            show_image_url(pred_url, f"Prediction {label}")




    # radio buttons for ranking preds
    st.markdown("### Rank each prediction (no ties allowed)")

    rank_options = ["Best", "Middle", "Worst"]

    rankA = st.radio("Prediction A", rank_options, key=f"rankA_{st.session_state.idx}", horizontal=True)
    rankB = st.radio("Prediction B", rank_options, key=f"rankB_{st.session_state.idx}", horizontal=True)
    rankC = st.radio("Prediction C", rank_options, key=f"rankC_{st.session_state.idx}", horizontal=True)

    label_to_rank = {"A": rankA, "B": rankB, "C": rankC}


    if st.button("Next"):

        # Enforce: exactly one Best, one Middle, one Worst (no ties)
        counts = Counter(label_to_rank.values())
        ok = (counts.get("Best", 0) == 1 and
              counts.get("Middle", 0) == 1 and
              counts.get("Worst", 0) == 1)
        if not ok:
            st.error("No ties allowed: assign exactly one Best, one Middle, and one Worst across A/B/C.")
            st.stop()

        # store result
        st.session_state.results.append({
            "user_id": args.user_id,
            "sample_id": row.sample_id,
            "datatype": row.datatype,
            "z": int(row.z),

            # per-pred label rank (no ties allowed)
            "ranking_labels": label_to_rank,

            # also store numeric form for convenience
            "ranking_numeric": {k: {"Best": 1, "Middle": 2, "Worst": 3}[v] for k, v in label_to_rank.items()},

            "model_map": mapping,  # A/B/C -> actual model name
        })
        st.session_state.idx += 1
        st.rerun()



# main app entry point
if __name__ == "__main__":
    main()







