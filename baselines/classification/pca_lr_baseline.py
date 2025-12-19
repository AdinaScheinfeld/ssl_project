#!/usr/bin/env python
"""
/home/ads4015/ssl_project/baselines/classification/pca_lr_baseline.py

PCA + Logistic Regression baseline on the SAME folds as your SwinUNETR classifier.

- Uses fold JSONs created by get_classification_cross_val_folds.py
- Infers class labels from filepath patterns (Selma3D, mesoSPIM, Allen Human)
- Runs PCA (50 comps, whiten=True) + multinomial Logistic Regression (lbfgs)
- Computes Accuracy, Macro F1, per-class precision/recall/F1
- Writes:
    cls_metrics/classification/pca_lr_cvfold{F}_ntr{NTR}_ntest{NTEST}_fttr{FTTR}_ftval{FTVAL}_seed{SEED}/
        preds_*.csv
        confusion_matrix_*.csv
        per_class_metrics_*.csv


Notes:
PCA implemented here uses image embeddings. 
PCA is trained on training set only, then applied to test set. Val set is not used (since PCA has no tunable hyperparams).


"""

import argparse
import json
import os
from pathlib import Path

import nibabel as nib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


# -----------------------------
# Label inference from path
# -----------------------------

def infer_label_from_path(path: str) -> str:
    """
    Infer class label name from absolute file path using your conventions.

    Classes (from JSON):
      - TPH2
      - VIP_ASLM_off
      - VIP_ASLM_on
      - amyloid_plaque_patches
      - c_fos_positive_patches
      - cell_nucleus_patches
      - stain-CR
      - stain-LEC
      - stain-NN
      - stain-NPY
      - stain-YO
      - vessels_patches
    """
    p = str(path)
    base = os.path.basename(p)

    # Selma3D finetune patches: class is the folder name
    if "selma3d_finetune_patches" in p:
        for cname in (
            "amyloid_plaque_patches",
            "c_fos_positive_patches",
            "cell_nucleus_patches",
            "vessels_patches",
        ):
            if f"/{cname}/" in p:
                return cname

    # MesoSPIM classes
    if "VIP_ASLM_off" in base:
        return "VIP_ASLM_off"
    if "VIP_ASLM_on" in base:
        return "VIP_ASLM_on"
    if "TPH2" in base:
        return "TPH2"

    # Allen Human stains via tokens in filename
    if "_cr_" in base:
        return "stain-CR"
    if "_lec_" in base:
        return "stain-LEC"
    if "_nn_" in base:
        return "stain-NN"
    if "_npy_" in base:
        return "stain-NPY"
    if "_yo_" in base:
        return "stain-YO"

    raise ValueError(f"[ERROR] Could not infer class label from path: {path}")


# -----------------------------
# Utility: load + normalize
# -----------------------------

def load_and_flatten(path: str) -> np.ndarray:
    """Load NIfTI, min-max normalize to [0,1], flatten to 1D."""
    img = nib.load(path)
    vol = img.get_fdata(dtype=np.float32)
    vmin = float(vol.min())
    vmax = float(vol.max())
    if vmax <= vmin:
        vol = np.zeros_like(vol, dtype=np.float32)
    else:
        vol = (vol - vmin) / (vmax - vmin)
    return vol.reshape(-1)


# -----------------------------
# Metrics from confusion matrix
# -----------------------------

def metrics_from_counts(cm: np.ndarray):
    """
    Computes accuracy, macro F1, per-class precision/recall/f1/support
    from confusion matrix cm (rows=true, cols=pred).
    """
    K = cm.shape[0]
    support = cm.sum(axis=1)  # per-class counts

    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    with np.errstate(divide="ignore", invalid="ignore"):
        prec = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        rec = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1 = np.where(prec + rec > 0, 2 * prec * rec / (prec + rec), 0.0)

    acc = tp.sum() / max(1, cm.sum())
    macro_f1 = f1.mean() if K > 0 else 0.0

    return acc, macro_f1, prec, rec, f1, support


# -----------------------------
# Main baseline runner
# -----------------------------

def run_pca_lr_baseline(
    fold_json: str,
    fold_id: int,
    output_root: str,
    pca_components: int = 50,
    max_iter: int = 500,
    seed: int = 100,
    val_percent: float = 0.2,
):
    fold_json = Path(fold_json)
    output_root = Path(output_root)

    with open(fold_json, "r") as f:
        meta = json.load(f)

    class_names = meta["classes"]
    folds = meta["folds"]
    if fold_id < 0 or fold_id >= len(folds):
        raise ValueError(
            f"[ERROR] fold_id={fold_id} but JSON has {len(folds)} folds."
        )

    fold = folds[fold_id]
    train_paths = list(fold["train"])
    test_paths = list(fold["test"])

    ntr = len(train_paths)
    ntest = len(test_paths)

    # mimic fttr/ftval tag logic (like your classification script)
    if ntr <= 1:
        ftval = 0
        fttr = ntr
    else:
        n_val = int(round(ntr * val_percent))
        if n_val < 1:
            n_val = 1
        if n_val > ntr - 1:
            n_val = ntr - 1
        ftval = n_val
        fttr = ntr - n_val

    tag = f"pca_lr_cvfold{fold_id}_ntr{ntr}_ntest{ntest}_fttr{fttr}_ftval{ftval}_seed{seed}"
    print(f"[INFO] Running PCA+LR baseline for tag: {tag}")

    # map class name -> idx
    class_to_idx = {c: i for i, c in enumerate(class_names)}
    K = len(class_names)

    # Build X/y for train and test
    X_train, y_train = [], []
    for p in train_paths:
        lname = infer_label_from_path(p)
        if lname not in class_to_idx:
            raise ValueError(
                f"[ERROR] Inferred label '{lname}' not in classes list for path {p}"
            )
        y_train.append(class_to_idx[lname])
        X_train.append(load_and_flatten(p))

    X_test, y_test = [], []
    for p in test_paths:
        lname = infer_label_from_path(p)
        if lname not in class_to_idx:
            raise ValueError(
                f"[ERROR] Inferred label '{lname}' not in classes list for path {p}"
            )
        y_test.append(class_to_idx[lname])
        X_test.append(load_and_flatten(p))

    X_train = np.stack(X_train)
    X_test = np.stack(X_test)
    y_train = np.array(y_train, dtype=np.int64)
    y_test = np.array(y_test, dtype=np.int64)

    print(f"[INFO] Train samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    print(f"[INFO] Input voxel dim: {X_train.shape[1]}")

    # PCA
    print(f"[INFO] Fitting PCA (components={pca_components}, whiten=True)...")
    
    # Enforce PCA n_components ≤ number of training samples
    max_possible = min(pca_components, X_train.shape[0], X_train.shape[1])

    if max_possible < pca_components:
        print(f"[WARN] Reducing PCA components from {pca_components} → {max_possible} "
            f"(n_samples={X_train.shape[0]}, n_features={X_train.shape[1]})", flush=True)

    pca = PCA(n_components=max_possible, whiten=True, random_state=seed)


    X_train_p = pca.fit_transform(X_train)
    X_test_p = pca.transform(X_test)

    # Logistic Regression
    print(f"[INFO] Training multinomial Logistic Regression (lbfgs, max_iter={max_iter})...")
    clf = LogisticRegression(
        max_iter=max_iter,
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=-1,
        random_state=seed,
    )
    clf.fit(X_train_p, y_train)

    # Predictions
    probs = clf.predict_proba(X_test_p)
    preds = probs.argmax(axis=1)

    acc = accuracy_score(y_test, preds)
    macro_f1 = f1_score(y_test, preds, average="macro")
    print(f"[RESULT] Test Accuracy = {acc:.4f}, Macro F1 = {macro_f1:.4f}")

    # Confusion matrix
    cm = np.zeros((K, K), dtype=np.int64)
    for t, p in zip(y_test, preds):
        cm[t, p] += 1

    acc2, macro_f12, prec, rec, f1, support = metrics_from_counts(cm)
    assert abs(acc - acc2) < 1e-6

    # -----------------------------
    # Write outputs (mimic structure)
    # -----------------------------
    metrics_root = output_root / "cls_metrics" / "classification" / tag
    metrics_root.mkdir(parents=True, exist_ok=True)

    # 1) Predictions CSV
    preds_csv = metrics_root / f"preds_{tag}.csv"
    with preds_csv.open("w") as f:
        f.write("filename,true_label,predicted_label,pred_idx,true_idx,prob_pred\n")
        for path, t_idx, p_idx, prob_vec in zip(test_paths, y_test, preds, probs):
            fname = os.path.basename(path)
            true_label = class_names[t_idx]
            pred_label = class_names[p_idx]
            prob_pred = float(prob_vec[p_idx])
            f.write(
                f"{fname},{true_label},{pred_label},{p_idx},{t_idx},{prob_pred:.8f}\n"
            )
    print(f"[INFO] Saved predictions to: {preds_csv}")

    # 2) Confusion matrix CSV
    cm_csv = metrics_root / f"confusion_matrix_{tag}.csv"
    with cm_csv.open("w") as f:
        f.write("," + ",".join(class_names) + "\n")
        for i, cname in enumerate(class_names):
            row = ",".join(str(int(v)) for v in cm[i].tolist())
            f.write(f"{cname},{row}\n")
    print(f"[INFO] Saved confusion matrix to: {cm_csv}")

    # 3) Per-class metrics CSV
    per_class_csv = metrics_root / f"per_class_metrics_{tag}.csv"
    with per_class_csv.open("w") as f:
        f.write("class_name,support,precision,recall,f1_score\n")
        for i, cname in enumerate(class_names):
            f.write(
                f"{cname},{int(support[i])},{prec[i]:.6f},{rec[i]:.6f},{f1[i]:.6f}\n"
            )
        f.write(
            f"__OVERALL__,{int(cm.sum())},ACC={acc:.6f},MACRO_F1={macro_f1:.6f},\n"
        )
    print(f"[INFO] Saved per-class metrics to: {per_class_csv}")

    return {
        "tag": tag,
        "accuracy": acc,
        "macro_f1": macro_f1,
        "preds_csv": str(preds_csv),
        "cm_csv": str(cm_csv),
        "per_class_csv": str(per_class_csv),
    }


# -----------------------------
# CLI
# -----------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="PCA+Logistic Regression baseline on classification folds."
    )
    p.add_argument(
        "--fold_json", type=str, required=True,
        help="Path to cls_folds_*.json created by get_classification_cross_val_folds.py"
    )
    p.add_argument(
        "--fold_id", type=int, required=True,
        help="Fold index (0-based) within the JSON 'folds' list."
    )
    p.add_argument(
        "--output_root", type=str, default="/midtier/paetzollab/scratch/ads4015/temp_selma_classification_preds_pca_lr",
        help="Root directory to write cls_metrics/classification/... (default: temp_selma_classification_preds_pca_lr)"
    )
    p.add_argument("--pca_components", type=int, default=50)
    p.add_argument("--max_iter", type=int, default=500)
    p.add_argument("--seed", type=int, default=100)
    p.add_argument("--val_percent", type=float, default=0.2)
    return p.parse_args()


def main():
    args = parse_args()
    run_pca_lr_baseline(
        fold_json=args.fold_json,
        fold_id=args.fold_id,
        output_root=args.output_root,
        pca_components=args.pca_components,
        max_iter=args.max_iter,
        seed=args.seed,
        val_percent=args.val_percent,
    )


if __name__ == "__main__":
    main()













