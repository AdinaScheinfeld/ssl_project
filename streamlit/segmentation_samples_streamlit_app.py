# /home/ads4015/ssl_project/streamlit/segmentation_samples_streamlit_app.py - Streamlit app to show segmentation prediction samples for human eval

# --- Setup ---

# imports
import argparse
import hashlib
import json
import matplotlib.pyplot as plt
import nibabel as nib
import pandas as pd
from pathlib import Path
import random
import streamlit as st


# --- Variable Definitions ---

MODELS = ["image_clip", "image_only", "random"]
LABELS = ["A", "B", "C"]


# --- Helper Functions ---

# function to load a 2D slice from a 3D volume
@st.cache_data(show_spinner=False)
def load_slice(path_str: str, z: int):
    vol = nib.load(path_str).get_fdata()
    return vol[:, :, z]

# function to show a 2D slice in streamlit
def show_slice(img2d, title: str):
    fig, ax = plt.subplots()
    ax.imshow(img2d.T, cmap="gray", origin="lower")
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)

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
    ap.add_argument("--data_csv", type=Path, default=Path("segmentation_samples.csv"))
    ap.add_argument("--out_json", type=Path, default=Path("segmentation_results.json"))
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--user_id", type=str, default="anon")
    args = ap.parse_args()

    # set random seed
    random.seed(args.seed)

    # load data CSV
    df = pd.read_csv(args.data_csv)

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
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(st.session_state.results, f, indent=2)
        st.write("Saved results to:", str(args.out_json))
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

    ref = load_slice(row["image_path"], int(row.z))

    c0, c1, c2, c3 = st.columns(4)

    with c0:
        show_slice(ref, "Image")

    for col, label in zip([c1, c2, c3], LABELS):
        model = mapping[label]
        pred = load_slice(row[f"{model}_path"], int(row.z))
        with col:
            show_slice(pred, f"Prediction {label}")

    if st.checkbox("Show ground truth"):
        gt = load_slice(row["gt_path"], int(row.z))
        show_slice(gt, "Ground truth")


    # radio buttons for ranking preds
    st.markdown("### Rank the predictions")
    best = st.radio("Best", LABELS, key=f"best_{st.session_state.idx}")
    mid  = st.radio("Middle", LABELS, key=f"mid_{st.session_state.idx}")
    worst = st.radio("Worst", LABELS, key=f"worst_{st.session_state.idx}")

    if st.button("Next"):
        if len({best, mid, worst}) != 3:
            st.error("Each rank must be unique (A/B/C each used exactly once).")
        else:
            st.session_state.results.append({
                "user_id": args.user_id,
                "sample_id": row.sample_id,
                "datatype": row.datatype,
                "z": int(row.z),
                "ranking": {best: 1, mid: 2, worst: 3},
                "model_map": mapping,  # hidden mapping for analysis
            })
            st.session_state.idx += 1
            st.rerun()


# main app entry point
if __name__ == "__main__":
    main()







