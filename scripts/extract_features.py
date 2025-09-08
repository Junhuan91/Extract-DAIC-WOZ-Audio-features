#!/usr/bin/env python3
"""
DAIC-WOZ audio feature extraction script
usage:
  python scripts/extract_features.py \
    --data_root /nfs/scratch \
    --output_dir /nfs/scratch/jtan/features/test \
    --start_id 300 --end_id 300 \
    --repo audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim
"""

import os
import sys
import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle
from typing import List, Dict

from src.data.audio_processor import AudioProcessor
# 需要你已更新 src/models/feature_extractors.py，里面包含两个类
from src.models.feature_extractors import EmotionFeatureExtractor, HFAudioFeatureExtractor


# ---- 推荐：把 HF 缓存指到你可写目录（在线加载时更稳） ----
os.environ.setdefault("HF_HOME", "/nfs/scratch/jtan/.hf_home")
os.environ.setdefault("TRANSFORMERS_CACHE", "/nfs/scratch/jtan/.hf_cache")


def load_config(config_path: str = "config/model_config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def get_daic_woz_files(data_root: str) -> List[Dict]:
    """Retrieve DAIC-WOZ data file paths and metadata
       data path: /nfs/scratch/dmamontov/daicwoz/
    """
    data_path = Path(data_root) / "dmamontov" / "daicwoz"
    daic_files = []

    for participant_id in range(300, 493):  # include 492
        folder_name = f"{participant_id}_P"
        participant_folder = data_path / folder_name

        if participant_folder.exists():
            audio_file = participant_folder / f"{participant_id}_AUDIO.wav"
            transcript_file = participant_folder / f"{participant_id}_TRANSCRIPT.csv"

            if audio_file.exists():
                daic_files.append(
                    {
                        "participant_id": participant_id,
                        "folder_path": participant_folder,
                        "audio_path": audio_file,
                        "transcript_path": transcript_file if transcript_file.exists() else None,
                    }
                )
            else:
                print(f"Warning: Audio file not found for participant {participant_id}")

    print(f"Found {len(daic_files)} valid DAIC-WOZ participants")
    return sorted(daic_files, key=lambda x: x["participant_id"])


def process_single_participant(
    participant_info: Dict, audio_processor: AudioProcessor, feature_extractor
) -> Dict | None:
    """Process data for a single participant"""
    participant_id = participant_info["participant_id"]
    audio_path = participant_info["audio_path"]
    transcript_path = participant_info["transcript_path"]

    print(f"Processing Participant {participant_id}: {audio_path.name}")

    # 1) 切分音频
    segments = audio_processor.process_long_audio(str(audio_path))
    if not segments:
        print(f"Failed to process audio for participant {participant_id}")
        return None

    print(f"Created {len(segments)} segments for participant {participant_id}")

    # 2) 提特征（内部做 mean+std pooling 或已为段级向量）
    features = feature_extractor.extract_features_from_audio(segments)
    if features is None:
        print(f"Failed to extract features for participant {participant_id}")
        return None

    # 3) 读转写（如存在）
    transcript_data = None
    if transcript_path and transcript_path.exists():
        try:
            transcript_data = pd.read_csv(transcript_path)
            print(f"Loaded transcript for participant {participant_id}")
        except Exception as e:
            print(f"Warning: Could not load transcript for {participant_id}: {e}")

    # 4) 汇总
    result = {
        "participant_id": participant_id,
        "folder_path": str(participant_info["folder_path"]),
        "audio_path": str(audio_path),
        "transcript_path": str(transcript_path) if transcript_path else None,
        "n_segments": features["n_segments"],
        "audio_features": features,
        "transcript_data": transcript_data,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description="Extract emotion features from DAIC-WOZ audio")
    parser.add_argument("--data_root", default="/nfs/scratch",
                        help="Root path to scratch directory (default: /nfs/scratch)")
    parser.add_argument("--output_dir", default="./data/processed/emotion_features/",
                        help="Output directory for features")
    parser.add_argument("--config", default="config/model_config.yaml", help="Config file path")
    parser.add_argument("--start_id", type=int, default=300, help="Starting participant ID")
    parser.add_argument("--end_id", type=int, default=492, help="Ending participant ID")
    parser.add_argument("--repo", default=None,
                        help="HF repo id for online extraction (e.g., audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim). "
                             "If not set, use SpeechBrain model from config.")
    args = parser.parse_args()

    # 读配置
    config = load_config(args.config)

    # 输出目录
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 音频切分器
    audio_processor = AudioProcessor(
        sample_rate=config["audio_processing"]["sample_rate"],
        segment_length=config["audio_processing"]["segment_length"],
        overlap=config["audio_processing"]["overlap"],
    )

    # 选择提特征器：
    #   优先用命令行 --repo（HF 在线加载）；
    #   没给的话：走 SpeechBrain（用 config['emotion_model']['huggingface_id']）
    if args.repo:
        print(f"Using HF model online: {args.repo}")
        feature_extractor = HFAudioFeatureExtractor(
            repo_id=args.repo,
            sample_rate=config["audio_processing"]["sample_rate"],
        )
    else:
        sb_id = config["emotion_model"]["huggingface_id"]
        print(f"Using SpeechBrain model: {sb_id}")
        # 把 savedir 放到你可写目录，避免默认缓存路径权限问题
        feature_extractor = EmotionFeatureExtractor(
            model_id=sb_id,
            savedir="/nfs/scratch/jtan/models/emotion-recognition-wav2vec2-IEMOCAP",
            sample_rate=config["audio_processing"]["sample_rate"],
        )

    # 构建文件清单
    participant_files = get_daic_woz_files(args.data_root)
    filtered_files = [p for p in participant_files if args.start_id <= p["participant_id"] <= args.end_id]
    print(f"Processing {len(filtered_files)} participants (ID: {args.start_id}-{args.end_id})")

    # 逐个处理
    all_results = []
    failed_participants = []

    for participant_info in tqdm(filtered_files, desc="Processing DAIC-WOZ participants"):
        try:
            result = process_single_participant(participant_info, audio_processor, feature_extractor)
            if result:
                feature_file = output_path / f"participant_{result['participant_id']}_emotion_features.pkl"
                with open(feature_file, "wb") as f:
                    pickle.dump(result, f)

                all_results.append(
                    {
                        "participant_id": result["participant_id"],
                        "n_segments": result["n_segments"],
                        "feature_file": str(feature_file),
                        "has_transcript": result["transcript_path"] is not None,
                    }
                )
            else:
                failed_participants.append(participant_info["participant_id"])

        except Exception as e:
            print(f"Error processing participant {participant_info['participant_id']}: {e}")
            failed_participants.append(participant_info["participant_id"])

    # 汇总
    summary = {
        "total_participants": len(filtered_files),
        "successful": len(all_results),
        "failed": len(failed_participants),
        "failed_participant_ids": failed_participants,
        "results": all_results,
        "data_path": str(Path(args.data_root) / "dmamontov" / "daicwoz"),
        "processing_date": str(pd.Timestamp.now()),
    }

    summary_file = output_path / "daic_woz_processing_summary.pkl"
    with open(summary_file, "wb") as f:
        pickle.dump(summary, f)

    # CSV 索引
    pd.DataFrame(all_results).to_csv(output_path / "daic_woz_feature_index.csv", index=False)

    print("\nDAIC-WOZ Processing complete!")
    print(f"Successful: {len(all_results)}/{len(filtered_files)}")
    print(f"Failed participants: {failed_participants}")
    print(f"Results saved to: {output_path}")


if __name__ == "__main__":
    main()
