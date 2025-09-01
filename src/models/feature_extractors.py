import torch
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier
import numpy as np
from typing import List, Dict
from pathlib import Path

class EmotionFeatureExtractor:
    """使用SpeechBrain情感识别模型提取特征"""
    
    def __init__(self, model_id: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"):
        self.model_id = model_id
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self):
        """加载预训练模型"""
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
        """从单个音频片段提取特征"""
        if self.model is None:
            self.load_model()
            
        try:
            # 转换为tensor
            audio_tensor = torch.FloatTensor(audio_segment).unsqueeze(0).to(self.device)
            
            # 提取特征（不是分类概率，而是中间特征）
            with torch.no_grad():
                # 获取编码器的输出（特征）而不是分类结果
                embeddings = self.model.encode_batch(audio_tensor)
                
            return embeddings.cpu().numpy().squeeze()
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            return None
    
    def extract_features_from_audio(self, audio_segments: List[np.ndarray]) -> Dict:
        """从所有音频片段提取特征并聚合"""
        segment_features = []
        
        for i, segment in enumerate(audio_segments):
            features = self.extract_features_from_segment(segment)
            if features is not None:
                segment_features.append(features)
                print(f"Processed segment {i+1}/{len(audio_segments)}")
        
        if not segment_features:
            return None
            
        segment_features = np.array(segment_features)
        
        # 聚合策略：统计特征
        aggregated_features = {
            'mean': np.mean(segment_features, axis=0),
            'std': np.std(segment_features, axis=0),
            'max': np.max(segment_features, axis=0),
            'min': np.min(segment_features, axis=0),
            'median': np.median(segment_features, axis=0)
        }
        
        return {
            'segment_features': segment_features,  # 每个片段的特征
            'aggregated': aggregated_features,     # 聚合后的特征
            'n_segments': len(segment_features)
        }
