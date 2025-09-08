#!/usr/bin/env python3
"""
Universal SVM trainer for DAIC-WOZ features (audio or text).

Usage:
  python scripts/train_eval_svm.py \
    --index_csv /path/to/daic_woz_feature_index.csv \
    --labels_csv /nfs/scratch/jtan/labels.csv \
    --use mean+std
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def _pick_feature_block(rec: dict) -> str:
    """Return 'audio_features' or 'text_features' depending on what's in the .pkl."""
    if "audio_features" in rec:
        return "audio_features"
    if "text_features" in rec:
        return "text_features"
    raise KeyError("Neither 'audio_features' nor 'text_features' found in the record.")


def load_features(index_csv: str, use: str = "mean+std"):
    """
    Load participant-level features from index CSV (works for audio or text).
    `use`: 'mean' | 'std' | 'mean+std'
    """
    df = pd.read_csv(index_csv)
    X, pids, ftypes = [], [], []

    for _, row in df.iterrows():
        with open(row["feature_file"], "rb") as f:
            rec = pickle.load(f)

        block = _pick_feature_block(rec)  # 'audio_features' or 'text_features'
        feats = rec[block]["aggregated"]  # {'mean': .., 'std': ..}

        if use == "mean":
            vec = np.array(feats["mean"])
        elif use == "std":
            vec = np.array(feats["std"])
        else:  # mean+std
            vec = np.concatenate([feats["mean"], feats["std"]], axis=0)

        X.append(vec)
        pids.append(rec["participant_id"])
        ftypes.append(block)

    X = np.stack(X, axis=0)
    pids = np.asarray(pids)
    return X, pids, ftypes, df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index_csv", required=True, help="Feature index CSV (audio or text).")
    ap.add_argument("--labels_csv", required=True, help="CSV with columns: participant_id,label")
    ap.add_argument("--use", default="mean+std", choices=["mean", "std", "mean+std"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # Load features
    X, pids, ftypes, df = load_features(args.index_csv, use=args.use)

    # Load labels
    labels_df = pd.read_csv(args.labels_csv)
    if "participant_id" not in labels_df.columns or "label" not in labels_df.columns:
        raise ValueError("labels_csv must contain columns: participant_id,label")
    lab_map = labels_df.set_index("participant_id")["label"].to_dict()

    y = np.array([lab_map.get(int(pid), None) for pid in pids])
    mask = y != None
    X, y, pids = X[mask], y[mask], pids[mask]

    if len(X) == 0:
        raise ValueError("No samples matched between index_csv and labels_csv.")

    # Pipeline: Standardize -> LinearSVC
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LinearSVC())
    ])
    param = {"clf__C": [1e-3, 1e-2, 1e-1, 1, 10, 100]}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    accs, f1s, baccs, pras = [], [], [], []
    for tr, va in skf.split(X, y):
        gs = GridSearchCV(pipe, param, scoring="f1", cv=3, n_jobs=-1)
        gs.fit(X[tr], y[tr])

        pred = gs.predict(X[va])
        # 用 decision_function 计算 PR-AUC（如果报错则降级为 0/1）
        try:
            score = gs.decision_function(X[va])
            pr_auc = average_precision_score(y[va], score)
        except Exception:
            pr_auc = average_precision_score(y[va], pred)

        accs.append(accuracy_score(y[va], pred))
        f1s.append(f1_score(y[va], pred))
        baccs.append(balanced_accuracy_score(y[va], pred))
        pras.append(pr_auc)

    print(f"n={len(y)} | use={args.use} | index={Path(args.index_csv).name}")
    print(f"ACC   : {np.mean(accs):.3f} ± {np.std(accs):.3f}")
    print(f"F1    : {np.mean(f1s):.3f} ± {np.std(f1s):.3f}")
    print(f"BAcc  : {np.mean(baccs):.3f} ± {np.std(baccs):.3f}")
    print(f"PR-AUC: {np.mean(pras):.3f} ± {np.std(pras):.3f}")


if __name__ == "__main__":
    main()
