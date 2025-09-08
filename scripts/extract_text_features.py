#!/usr/bin/env python3
"""
从 DAIC-WOZ 自带 TRANSCRIPT.csv 提取文本特征（RoBERTa），不做 Whisper。
输出：
  participant_XXX_text_features.pkl
  daic_woz_text_feature_index.csv
"""
import os
import argparse
from pathlib import Path
import pandas as pd
import pickle
from tqdm import tqdm

from src.models.text_feature_extractor import HFTextFeatureExtractor

def read_transcript(csv_path: Path) -> str:
    df = pd.read_csv(csv_path)
    # 自适应列名
    spk_col, txt_col = None, None
    for c in df.columns:
        cl = c.lower().strip()
        if cl in ["speaker", "participant", "speaker_type"]:
            spk_col = c
        if cl in ["value", "text", "utterance"]:
            txt_col = c
    # 兜底：最后一列当文本列
    if txt_col is None:
        txt_col = df.columns[-1]
    if spk_col is None:
        # 没有说话人列，就全拼
        return " ".join(df[txt_col].astype(str).tolist())

    # 只取“Participant”的发言
    mask = df[spk_col].astype(str).str.lower().str.contains("participant")
    lines = df.loc[mask, txt_col].astype(str).tolist()
    return " ".join(lines)

def build_index(data_root: str):
    base = Path(data_root) / "dmamontov" / "daicwoz"
    items = []
    for pid in range(300, 493):
        folder = base / f"{pid}_P"
        if not folder.exists():
            continue
        audio = folder / f"{pid}_AUDIO.wav"
        trans = folder / f"{pid}_TRANSCRIPT.csv"
        if audio.exists() and trans.exists():
            items.append({"participant_id": pid, "audio": audio, "transcript": trans})
    return items

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/nfs/scratch")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--roberta_repo", default="rafalposwiata/roberta-large-depression")
    ap.add_argument("--pooling", default="cls", choices=["cls", "mean"])
    ap.add_argument("--max_words_per_chunk", type=int, default=200)
    args = ap.parse_args()

    # 缓存放可写位置
    os.environ.setdefault("HF_HOME", "/nfs/scratch/jtan/.hf_home")
    os.environ.setdefault("TRANSFORMERS_CACHE", "/nfs/scratch/jtan/.hf_cache")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    extractor = HFTextFeatureExtractor(
        repo_id=args.roberta_repo,
        pooling=args.pooling,
        max_words=args.max_words_per_chunk
    )

    index = build_index(args.data_root)
    rows, failed = [], []

    for it in tqdm(index, desc="DAIC-WOZ Text"):
        pid = it["participant_id"]
        trans = it["transcript"]
        try:
            text = read_transcript(trans)
            feats = extractor.extract_features_from_text(text)
            if feats is None:
                print(f"[empty] P{pid} transcript")
                failed.append(pid); continue

            pkl_path = out_dir / f"participant_{pid}_text_features.pkl"
            with open(pkl_path, "wb") as f:
                pickle.dump({
                    "participant_id": pid,
                    "audio_path": str(it["audio"]),
                    "transcript_path": str(trans),
                    "n_segments": feats["n_segments"],
                    "text_features": feats
                }, f)

            rows.append({
                "participant_id": pid,
                "n_segments": feats["n_segments"],
                "feature_file": str(pkl_path),
                "has_transcript": True
            })
        except Exception as e:
            print(f"[err] P{pid}: {e}")
            failed.append(pid)

    # 索引与汇总
    pd.DataFrame(rows).to_csv(out_dir / "daic_woz_text_feature_index.csv", index=False)
    import datetime as dt, pickle as pkl
    summary = {
        "total": len(index),
        "successful": len(rows),
        "failed": len(failed),
        "failed_ids": failed,
        "processing_date": dt.datetime.now().isoformat(timespec="seconds")
    }
    with open(out_dir / "daic_woz_text_processing_summary.pkl", "wb") as f:
        pickle.dump(summary, f)

    print(f"\nDone. OK {len(rows)}/{len(index)}; failed {failed}")
    print(f"Saved to: {out_dir}")

if __name__ == "__main__":
    main()
