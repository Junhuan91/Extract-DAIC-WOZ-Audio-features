import librosa
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple
import soundfile as sf

class AudioProcessor:
    """处理DAIC-WOZ长音频的分段和预处理"""
    
    def __init__(self, sample_rate: int = 16000, segment_length: int = 30, overlap: int = 15):
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.overlap = overlap
        
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """加载音频文件"""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate)
            return audio, sr
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None, None
    
    def segment_audio(self, audio: np.ndarray, sr: int) -> List[np.ndarray]:
        """将长音频分割成重叠的片段"""
        segment_samples = self.segment_length * sr
        hop_samples = (self.segment_length - self.overlap) * sr
        
        segments = []
        start = 0
        
        while start + segment_samples <= len(audio):
            segment = audio[start:start + segment_samples]
            segments.append(segment)
            start += hop_samples
            
        # 处理最后一个片段
        if start < len(audio):
            remaining = audio[start:]
            if len(remaining) >= self.sample_rate * 5:  # 至少5秒
                segments.append(remaining)
                
        return segments
    
    def preprocess_segment(self, segment: np.ndarray) -> np.ndarray:
        """预处理单个音频片段"""
        # 归一化
        if np.max(np.abs(segment)) > 0:
            segment = segment / np.max(np.abs(segment))
        
        # 确保长度一致（padding或truncation）
        target_length = self.segment_length * self.sample_rate
        if len(segment) < target_length:
            segment = np.pad(segment, (0, target_length - len(segment)))
        elif len(segment) > target_length:
            segment = segment[:target_length]
            
        return segment
    
    def process_long_audio(self, audio_path: str) -> List[np.ndarray]:
        """处理完整的长音频文件"""
        audio, sr = self.load_audio(audio_path)
        if audio is None:
            return []
            
        segments = self.segment_audio(audio, sr)
        processed_segments = [self.preprocess_segment(seg) for seg in segments]
        
        return processed_segments
