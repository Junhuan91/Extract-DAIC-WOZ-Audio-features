#!/usr/bin/env python3
"""
DAIC-WOZ audio feature extraction script
usage: python scripts/extract_features.py --data_dir /path/to/daic_woz --output_dir ./data/processed
"""

import argparse
import yaml
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import pickle

from typing import List, Dict
from src.data.audio_processor import AudioProcessor
from src.models.feature_extractors import EmotionFeatureExtractor

def load_config(config_path: str = "config/model_config.yaml"):
    """load config file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_daic_woz_files(data_root: str) -> List[Dict]:
    """Retrieve DAIC-WOZ data file paths and metadata"""
    # data path: /nfs/scratch/dmamontov/daicwoz/
    data_path = Path(data_root) / "dmamontov" / "daicwoz"
    
    daic_files = []
    
    # Iterate through 300_P to 492_P folder
    for participant_id in range(300, 493):  # 300 to 492 (include 492)
        folder_name = f"{participant_id}_P"
        participant_folder = data_path / folder_name
        
        if participant_folder.exists():
            # Search for audio and transcription files
            audio_file = participant_folder / f"{participant_id}_AUDIO.wav"
            transcript_file = participant_folder / f"{participant_id}_TRANSCRIPT.csv"
            
            if audio_file.exists():
                daic_files.append({
                    'participant_id': participant_id,
                    'folder_path': participant_folder,
                    'audio_path': audio_file,
                    'transcript_path': transcript_file if transcript_file.exists() else None
                })
            else:
                print(f"Warning: Audio file not found for participant {participant_id}")
    
    print(f"Found {len(daic_files)} valid DAIC-WOZ participants")
    return sorted(daic_files, key=lambda x: x['participant_id'])

def process_single_participant(participant_info: Dict, audio_processor: AudioProcessor, 
                             feature_extractor: EmotionFeatureExtractor) -> Dict | None:
    """Process data for a single participant"""
    participant_id = participant_info['participant_id']
    audio_path = participant_info['audio_path']
    transcript_path = participant_info['transcript_path']
    
    print(f"Processing Participant {participant_id}: {audio_path.name}")
    
    # 1. Process audio in segments
    segments = audio_processor.process_long_audio(str(audio_path))
    if not segments:
        print(f"Failed to process audio for participant {participant_id}")
        return None
    
    print(f"Created {len(segments)} segments for participant {participant_id}")
    
    # 2. feature extraction
    features = feature_extractor.extract_features_from_audio(segments)
    if features is None:
        print(f"Failed to extract features for participant {participant_id}")
        return None
    
    # 3. Read transcription file (if it exists)
    transcript_data = None
    if transcript_path and transcript_path.exists():
        try:
            transcript_data = pd.read_csv(transcript_path)
            print(f"Loaded transcript for participant {participant_id}")
        except Exception as e:
            print(f"Warning: Could not load transcript for {participant_id}: {e}")
    
    # 4. Integrate results
    result = {
        'participant_id': participant_id,
        'folder_path': str(participant_info['folder_path']),
        'audio_path': str(audio_path),
        'transcript_path': str(transcript_path) if transcript_path else None,
        'n_segments': features['n_segments'],
        'audio_features': features,
        'transcript_data': transcript_data
    }
    
    return result

def main():
    parser = argparse.ArgumentParser(description='Extract emotion features from DAIC-WOZ audio')
    parser.add_argument('--data_root', default='/nfs/scratch', 
                       help='Root path to scratch directory (default: /nfs/scratch)')
    parser.add_argument('--output_dir', default='./data/processed/emotion_features/', 
                       help='Output directory for features')
    parser.add_argument('--config', default='config/model_config.yaml', help='Config file path')
    parser.add_argument('--start_id', type=int, default=300, help='Starting participant ID')
    parser.add_argument('--end_id', type=int, default=492, help='Ending participant ID')
    
    args = parser.parse_args()
    
    # load config
    config = load_config(args.config)
    
    # create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize processor
    audio_processor = AudioProcessor(
        sample_rate=config['audio_processing']['sample_rate'],
        segment_length=config['audio_processing']['segment_length'],
        overlap=config['audio_processing']['overlap']
    )
    
    feature_extractor = EmotionFeatureExtractor(
        model_id="/nfs/scratch/jtan/models/emotion-recognition-wav2vec2-IEMOCAP"
    )
    
    # Get all DAIC-WOZ participant files
    participant_files = get_daic_woz_files(args.data_root)
    
    # Filter participants in the specified range
    filtered_files = [
        p for p in participant_files 
        if args.start_id <= p['participant_id'] <= args.end_id
    ]
    
    print(f"Processing {len(filtered_files)} participants (ID: {args.start_id}-{args.end_id})")
    
    # Process all participants
    all_results = []
    failed_participants = []
    
    for participant_info in tqdm(filtered_files, desc="Processing DAIC-WOZ participants"):
        try:
            result = process_single_participant(participant_info, audio_processor, feature_extractor)
            
            if result:
                # Save features of a single participant
                feature_file = output_path / f"participant_{result['participant_id']}_emotion_features.pkl"
                with open(feature_file, 'wb') as f:
                    pickle.dump(result, f)
                
                all_results.append({
                    'participant_id': result['participant_id'],
                    'n_segments': result['n_segments'],
                    'feature_file': str(feature_file),
                    'has_transcript': result['transcript_path'] is not None
                })
            else:
                failed_participants.append(participant_info['participant_id'])
                
        except Exception as e:
            print(f"Error processing participant {participant_info['participant_id']}: {e}")
            failed_participants.append(participant_info['participant_id'])
    
    # Save processing summary
    summary = {
        'total_participants': len(filtered_files),
        'successful': len(all_results),
        'failed': len(failed_participants),
        'failed_participant_ids': failed_participants,
        'results': all_results,
        'data_path': str(Path(args.data_root) / "dmamontov" / "daicwoz"),
        'processing_date': str(pd.Timestamp.now())
    }
    
    summary_file = output_path / "daic_woz_processing_summary.pkl"
    with open(summary_file, 'wb') as f:
        pickle.dump(summary, f)
    
    # create csv index files
    df = pd.DataFrame(all_results)
    df.to_csv(output_path / "daic_woz_feature_index.csv", index=False)
    
    print(f"\nDAIC-WOZ Processing complete!")
    print(f"Successful: {len(all_results)}/{len(filtered_files)}")
    print(f"Failed participants: {failed_participants}")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    main()
