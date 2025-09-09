import os
from pathlib import Path
from typing import List, Dict, Optional, Union

import numpy as np
import torch

# ---- SpeechBrain ----
import speechbrain as sb
from speechbrain.pretrained import EncoderClassifier

# ---- Transformers ----
from transformers import AutoProcessor, AutoModel


def mean_std_pool(hidden_last: torch.Tensor) -> torch.Tensor:
    """
    hidden_last: (B, T, C)
    return: (B, 2C)  = [mean, std]
    """
    h = hidden_last.mean(dim=1)
    s = hidden_last.std(dim=1, unbiased=False)
    return torch.cat([h, s], dim=-1)


class BaseFeatureExtractor:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_features_from_segment(self, audio_segment: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def extract_features_from_audio(self, audio_segments: List[np.ndarray]) -> Dict:
        if not audio_segments:
            return None

        # 批量提取，速度更快
        seg_feats = self._embed_batch(audio_segments)  # (N, C')  或 (N, 2C)
        seg_feats_np = seg_feats.cpu().numpy()

        aggregated = {
            "mean": seg_feats_np.mean(axis=0),
            "std": seg_feats_np.std(axis=0),
        }
        return {
            "segment_features": seg_feats_np,
            "aggregated": aggregated,
            "n_segments": int(seg_feats_np.shape[0]),
        }

    # 默认逐段；子类可重写为真正的 batch
    def _embed_batch(self, segments: List[np.ndarray]) -> torch.Tensor:
        xs = []
        for seg in segments:
            xs.append(torch.from_numpy(seg.astype("float32")))
        outs = []
        for x in xs:
            v = torch.from_numpy(self.extract_features_from_segment(x.numpy()))
            outs.append(v)
        return torch.stack(outs, dim=0)


# ============ SpeechBrain (.ckpt) ============
class EmotionFeatureExtractor(BaseFeatureExtractor):
    """
    SpeechBrain emotion-recognition-wav2vec2-IEMOCAP
    支持在线加载：source 可是 "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
    也可指向本地目录；关键是 savedir 指向一个可写目录（建议与 source 同目录或你自己的 models 目录）
    """
    def __init__(self,
                 model_id: str = "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
                 savedir: Optional[str] = "/nfs/scratch/jtan/models/emotion-recognition-wav2vec2-IEMOCAP",
                 sample_rate: int = 16000):
        super().__init__(sample_rate)
        self.model_id = model_id
        self.savedir = str(Path(savedir).resolve()) if savedir else None
        self.model: Optional[EncoderClassifier] = None

    def load_model(self):
        if self.model is not None:
            return
        # 关键：savedir 用一个你可写的固定目录，避免默认缓存或联网问题
        self.model = EncoderClassifier.from_hparams(
            source=self.model_id,
            savedir=self.savedir or self.model_id,
            run_opts={"device": str(self.device)}
        )
        self.model.eval()

    @torch.inference_mode()
    def _embed_batch(self, segments: List[np.ndarray]) -> torch.Tensor:
        self.load_model()
        B = len(segments)
        T = max(len(s) for s in segments)
        batch = torch.zeros(B, T, dtype=torch.float32)
        for i, seg in enumerate(segments):
            t = len(seg)
            batch[i, :t] = torch.from_numpy(seg.astype("float32"))
        batch = batch.to(self.device)  # (B, T)
        emb: torch.Tensor = self.model.encode_batch(batch)  # (B, C)
        # 这里返回的已是段级 embedding（SpeechBrain 内部已做了池化）
        return emb.detach().to("cpu")

    def extract_features_from_segment(self, audio_segment: np.ndarray) -> np.ndarray:
        # 单段兜底（不会走到这，除非父类没重写 _embed_batch）
        segs = [audio_segment]
        return self._embed_batch(segs).squeeze(0).numpy()


# ============ HuggingFace (Transformers) ============
class HFAudioFeatureExtractor(BaseFeatureExtractor):
    """
    任意 HF 音频模型：WavLM / HuBERT / W2V2 / Whisper encoder …
    在线加载 + mean+std pooling → 段级向量
    """
    def __init__(self,
                 repo_id: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
                 sample_rate: int = 16000):
        super().__init__(sample_rate)
        self.repo_id = repo_id
        self.processor = None
        self.model = None

        # 推荐把缓存放到你可写目录
        os.environ.setdefault("HF_HOME", "/nfs/scratch/jtan/.hf_home")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/nfs/scratch/jtan/.hf_cache")

    def load_model(self):
        if self.model is not None:
            return
        self.processor = AutoProcessor.from_pretrained(self.repo_id)
        self.model = AutoModel.from_pretrained(self.repo_id).to(self.device).eval()

    @torch.inference_mode()
    def _embed_batch(self, segments: List[np.ndarray]) -> torch.Tensor:
        self.load_model()
        # HF 处理：可以直接传 list 的 1D numpy，processor 会做 padding
        inputs = self.processor(
            segments,
            sampling_rate=self.sample_rate,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs, output_hidden_states=True)
        hidden = out.last_hidden_state  # (B, T, C)
        pooled = mean_std_pool(hidden)  # (B, 2C)
        return pooled.detach().to("cpu")

    def extract_features_from_segment(self, audio_segment: np.ndarray) -> np.ndarray:
        segs = [audio_segment]
        return self._embed_batch(segs).squeeze(0).numpy()
