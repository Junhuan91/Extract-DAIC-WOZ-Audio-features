#!/usr/bin/env python3
"""
DAIC-WOZ Data Exploration Script
Usage: python scripts/explore_data.py
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
import librosa

# Add src to path for imports
sys.path.append('.')
sys.path.append('./src')

def load_config(config_path: str = "config/data_config.yaml"):
    """Load configuration file"""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"Config file not found: {config_path}")
        return None
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def check_data_paths(config):
    """Check if data paths exist and are accessible"""
    print("Checking Data Paths")
    print("=" * 50)
    
    data_root = Path(config['daic_woz']['data_root'])
    data_path = data_root / config['daic_woz']['data_path']
    
    print(f"Data root: {data_root}")
    print(f"DAIC-WOZ path: {data_path}")
    print(f"Data root exists: {data_root.exists()}")
    print(f"DAIC-WOZ path exists: {data_path.exists()}")
    
    if not data_path.exists():
        print("DAIC-WOZ data path not found!")
        print("Please check the path in config/data_config.yaml")
        return False
    
    return True, data_path

def check_labels_file(data_path, config):
    """Check labels file (meta_info.csv)"""
    print("\nChecking Labels File")
    print("=" * 50)
    
    labels_file = data_path / config['labels']['file_path']
    print(f"Labels file path: {labels_file}")
    print(f"Labels file exists: {labels_file.exists()}")
    
    if not labels_file.exists():
        print("Labels file not found!")
        return None
    
    try:
        labels_df = pd.read_csv(labels_file)
        print(f"Labels loaded successfully")
        print(f"Shape: {labels_df.shape}")
        print(f"Columns: {list(labels_df.columns)}")
        
        # Show first few rows
        print("\nFirst 5 rows:")
        print(labels_df.head())
        
        # Check for depression-related columns
        depression_cols = [col for col in labels_df.columns 
                          if any(keyword in col.lower() for keyword in ['phq', 'depression', 'score'])]
        if depression_cols:
            print(f"\nDepression-related columns found: {depression_cols}")
            
            # Show value counts for binary labels
            for col in depression_cols:
                if 'binary' in col.lower() or labels_df[col].nunique() <= 5:
                    print(f"\n {col} value counts:")
                    print(labels_df[col].value_counts())
        
        return labels_df
        
    except Exception as e:
        print(f"Error reading labels file: {e}")
        return None

def check_participant_folders(data_path, config):
    """Check participant folders and files"""
    print("\nChecking Participant Folders")
    print("=" * 50)
    
    start_id = config['daic_woz']['participant_range']['start']
    end_id = config['daic_woz']['participant_range']['end']
    
    audio_pattern = config['daic_woz']['file_patterns']['audio']
    transcript_pattern = config['daic_woz']['file_patterns']['transcript']
    folder_pattern = config['daic_woz']['folder_pattern']
    
    valid_participants = []
    missing_audio = []
    missing_transcript = []
    
    # Check first 10 participants as sample
    sample_range = range(start_id, min(start_id + 10, end_id + 1))
    
    print(f"Checking sample participants {start_id} to {start_id + 9}:")
    
    for participant_id in sample_range:
        folder_name = folder_pattern.format(participant_id=participant_id)
        participant_folder = data_path / folder_name
        
        audio_name = audio_pattern.format(participant_id=participant_id)
        transcript_name = transcript_pattern.format(participant_id=participant_id)
        
        audio_path = participant_folder / audio_name
        transcript_path = participant_folder / transcript_name
        
        print(f"\nParticipant {participant_id}:")
        print(f"Folder: {participant_folder.exists()}")
        print(f"Audio: {audio_path.exists()}")
        print(f"Transcript: {transcript_path.exists()}")
        
        if audio_path.exists():
            valid_participants.append(participant_id)
            
            # Check audio file size and duration
            try:
                file_size_mb = audio_path.stat().st_size / (1024 * 1024)
                print(f"Audio size: {file_size_mb:.1f} MB")
                
                # Quick duration check (optional, can be slow)
                # duration = librosa.get_duration(filename=str(audio_path))
                # print(f"Duration: {duration:.1f} seconds")
                
            except Exception as e:
                print(f"Error checking audio: {e}")
        else:
            missing_audio.append(participant_id)
            
        if not transcript_path.exists():
            missing_transcript.append(participant_id)
    
    print(f"\nSample Check Summary:")
    print(f"Valid participants (with audio): {len(valid_participants)}")
    print(f"Missing audio: {len(missing_audio)}")
    print(f"Missing transcript: {len(missing_transcript)}")
    
    return valid_participants

def check_full_dataset(data_path, config):
    """Quick check of full dataset"""
    print("\nFull Dataset Overview")
    print("=" * 50)
    
    start_id = config['daic_woz']['participant_range']['start']
    end_id = config['daic_woz']['participant_range']['end']
    folder_pattern = config['daic_woz']['folder_pattern']
    audio_pattern = config['daic_woz']['file_patterns']['audio']
    
    total_expected = end_id - start_id + 1
    audio_found = 0
    transcript_found = 0
    
    print(f"Expected participants: {total_expected} (ID: {start_id}-{end_id})")
    
    for participant_id in range(start_id, end_id + 1):
        folder_name = folder_pattern.format(participant_id=participant_id)
        participant_folder = data_path / folder_name
        
        if participant_folder.exists():
            audio_name = audio_pattern.format(participant_id=participant_id)
            transcript_name = config['daic_woz']['file_patterns']['transcript'].format(participant_id=participant_id)
            
            if (participant_folder / audio_name).exists():
                audio_found += 1
            if (participant_folder / transcript_name).exists():
                transcript_found += 1
    
    print(f"Audio files found: {audio_found}/{total_expected}")
    print(f"Transcript files found: {transcript_found}/{total_expected}")
    print(f"Audio coverage: {audio_found/total_expected*100:.1f}%")
    
    return audio_found

def main():
    """Main exploration function"""
    print("DAIC-WOZ Data Exploration")
    print("=" * 60)
    
    # Load configuration
    config = load_config()
    if config is None:
        print("Cannot proceed without configuration")
        return
    
    print("Configuration loaded successfully")
    
    # Check data paths
    path_check = check_data_paths(config)
    if not path_check[0]:
        return
    
    data_path = path_check[1]
    
    # Check labels file
    labels_df = check_labels_file(data_path, config)
    
    # Check participant folders (sample)
    valid_participants = check_participant_folders(data_path, config)
    
    # Full dataset overview
    total_audio_files = check_full_dataset(data_path, config)
    
    # Final summary
    print("\nExploration Summary")
    print("=" * 50)
    
    if labels_df is not None:
        print(f"Labels file: Available ({len(labels_df)} entries)")
    else:
        print("Labels file: Not available")
    
    print(f"Audio files: {total_audio_files} found")
    print(f"Sample check: {len(valid_participants)} participants verified")
    
    if total_audio_files >= 100:  # Reasonable threshold
        print("\nReady for feature extraction!")
        print("Next step: python scripts/extract_features.py --test_mode")
    else:
        print("\nDataset seems incomplete. Please check data paths.")
        print("You may need to adjust paths in config/data_config.yaml")

if __name__ == "__main__":
    main()
