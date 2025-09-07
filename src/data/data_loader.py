import pandas as pd
import pickle
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple

class DAICWOZDataLoader:
    """DAIC-WOZ data loader"""
    
    def __init__(self, data_root: str, feature_dir: str, labels_file: str):
        self.data_root = Path(data_root)
        self.feature_dir = Path(feature_dir)
        self.labels_file = labels_file
        self.labels_df = None
        
    def load_labels(self) -> pd.DataFrame:
        """Load depression labels"""
        if self.labels_df is None:
            self.labels_df = pd.read_csv(self.labels_file)
            
        return self.labels_df
    
    def load_single_feature(self, file_id: str) -> Dict:
        """load feature of single file"""
        feature_file = self.feature_dir / f"{file_id}_emotion_features.pkl"
        
        if not feature_file.exists():
            print(f"Feature file not found: {feature_file}")
            return None
            
        with open(feature_file, 'rb') as f:
            return pickle.load(f)
    
    def load_all_features(self) -> Tuple[List[np.ndarray], List[int]]:
        """load all features and labels"""
        labels_df = self.load_labels()
        
        features = []
        labels = []
        valid_ids = []
        
        for _, row in labels_df.iterrows():
            file_id = row['participant_id']  # Adjust according to actual column names
            depression_label = row['depression_label']  # Adjust according to actual column names
            
            feature_data = self.load_single_feature(file_id)
            if feature_data:
                # Use aggregated features
                aggregated = feature_data['features']['aggregated']
                # Concatenate all statistical features
                feature_vector = np.concatenate([
                    aggregated['mean'],
                    aggregated['std'],
                    aggregated['max'],
                    aggregated['min']
                ])
                
                features.append(feature_vector)
                labels.append(depression_label)
                valid_ids.append(file_id)
        
        return np.array(features), np.array(labels), valid_ids
