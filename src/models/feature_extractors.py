import torch
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier
import numpy as np
from typing import List, Dict
from pathlib import Path

class EmotionFeatureExtractor:
    """Use SpeechBrain emotion recognition model to extract features"""
    
    def __init__(self, model_id: str = "/nfs/scratch/jtan/models/emotion-recognition-wav2vec2-IEMOCAP"):
        self.model_id = model_id
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """load pre-trained model"""
        try:
            self.model = EncoderClassifier.from_hparams(
                source=self.model_id,
                savedir="tmp_model",
                run_opts={"device": self.device}
            )
            print(f"Model loaded successfully on {self.device}")
        except Exception as e:
            print(f"Error loading model: {e}")
            
    def extract_features_from_segment(self, audio_segment: np.ndarray) -> np.ndarray:
        """Extract features from individual audio segment"""
        if self.model is None:
            self.load_model()
            
        try:
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio_segment).unsqueeze(0).to(self.device)
            
            # Extract features (not classification probabilities, but intermediate features)
            with torch.no_grad():
                # Get encoder output (features) rather than classification results
                embeddings = self.model.encode_batch(audio_tensor)
                
            return embeddings.cpu().numpy().squeeze()
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def extract_features_from_audio(self, audio_segments: List[np.ndarray]) -> Dict:
        """Extract features from all audio segments and aggregate"""
        segment_features = []
        
        for i, segment in enumerate(audio_segments):
            features = self.extract_features_from_segment(segment)
            if features is not None:
                segment_features.append(features)
                print(f"Processed segment {i+1}/{len(audio_segments)}")
        
        if not segment_features:
            return None
            
        segment_features = np.array(segment_features)
        
        # Aggregation strategy: statistical features
        aggregated_features = {
            'mean': np.mean(segment_features, axis=0),
            'std': np.std(segment_features, axis=0),
            'max': np.max(segment_features, axis=0),
            'min': np.min(segment_features, axis=0),
            'median': np.median(segment_features, axis=0)
        }
        
        return {
            'segment_features': segment_features,  # Features of each segment
            'aggregated': aggregated_features,     # Aggregated features
            'n_segments': len(segment_features)
        }
