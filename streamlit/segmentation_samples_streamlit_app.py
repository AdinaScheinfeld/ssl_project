#!/usr/bin/env python3
import argparse
import json
import random
from pathlib import Path

import streamlit as st
import pandas as pd
import nibabel as nib
import matplotlib.pyplot as plt

MODELS = ["image_clip", "image_only", "random"]
LABELS = ["A", "B", "C"]

def load_slice(path_str: str, z: int):
    vol = nib.load(path_str).get_fdata()
    return vol[:, :, z]

def show_slice(img2d, title: str):
    fig, ax = plt.subplots()
    ax.imshow(img2d.T, cmap="gray", origin="lower")
    ax.set_title(title)
    ax.axis("off")
    st.pyplot(fig, clear_figure=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_csv", type=Path, default=Path("human_eval_samples.csv"))
    ap.add_argument("--out_json", type=Path, default=Path("human_eval_results.json"))
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--user_id", type=str, default="anon")
    args = ap.parse_args()

    random.seed(args.seed)

    df = pd.read_csv(args.data_csv)

    st.title("Segmentation Prediction Ranking")
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

    row = df.iloc[st.session_state.idx]

    st.markdown(f"### Slice {st.session_state.idx + 1} / {len(df)}")
    st.write(f"**Datatype:** {row.datatype} | **Sample:** {row.sample_id} | **z:** {int(row.z)}")

    # --------------------
    # STABLE RANDOMIZE (per slice)
    # --------------------
    if "mappings" not in st.session_state:
        st.session_state.mappings = {}

    if st.session_state.idx not in st.session_state.mappings:
        models = MODELS.copy()
        random.shuffle(models)
        st.session_state.mappings[st.session_state.idx] = dict(zip(LABELS, models))

    mapping = st.session_state.mappings[st.session_state.idx]


    cols = st.columns(3)
    for col, label in zip(cols, LABELS):
        model = mapping[label]
        img = load_slice(row[f"{model}_path"], int(row.z))
        with col:
            show_slice(img, f"Prediction {label}")

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

if __name__ == "__main__":
    main()







