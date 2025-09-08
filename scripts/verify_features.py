#!/usr/bin/env python3
import argparse, pickle
from pathlib import Path
import random, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features_dir", required=True)
    args = ap.parse_args()
    feat_dir = Path(args.features_dir)
    files = list(feat_dir.glob("participant_*_*_features.pkl"))
    if not files:
        print("No .pkl found."); return
    f = random.choice(files)
    print("Sample file:", f)
    data = pickle.load(open(f, "rb"))

    # 自动选择 audio_features 或 text_features
    if "audio_features" in data:
        feats = data["audio_features"]
        ftype = "audio"
    elif "text_features" in data:
        feats = data["text_features"]
        ftype = "text"
    else:
        print("No audio_features or text_features found in file")
        return

    agg = feats["aggregated"]
    print("participant:", data["participant_id"])
    print("feature_type:", ftype)
    print("n_segments:", feats["n_segments"])
    print("mean shape:", np.array(agg["mean"]).shape)
    print("std  shape:", np.array(agg["std"]).shape)
    seg = feats["segment_features"]
    print("segment_features:", np.array(seg).shape)

if __name__ == "__main__":
    main()
