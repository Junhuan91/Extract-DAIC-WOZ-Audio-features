import os
from pathlib import Path
from typing import List, Dict, Optional
import numpy as np
import torch

# SpeechBrain
from speechbrain.pretrained import EncoderClassifier

# HF
from transformers import AutoProcessor, AutoModel

def mean_std_pool(hidden_last: torch.Tensor) -> torch.Tensor:
    # hidden_last: (B, T, C) -> (B, 2C)
    h = hidden_last.mean(dim=1)
    s = hidden_last.std(dim=1, unbiased=False)
    return torch.cat([h, s], dim=-1)

class BaseFeatureExtractor:
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def extract_features_from_audio(self, audio_segments: List[np.ndarray]) -> Dict:
        if not audio_segments:
            return None
        seg_emb = self._embed_batch(audio_segments)      # (N, D)
        seg_np = seg_emb.cpu().numpy()
        return {
            "segment_features": seg_np,                  # (N, D)
            "aggregated": {
                "mean": seg_np.mean(axis=0),
                "std": seg_np.std(axis=0),
            },
            "n_segments": int(seg_np.shape[0]),
        }

    def _embed_batch(self, segments: List[np.ndarray]) -> torch.Tensor:
        raise NotImplementedError

# ---- SpeechBrain (.ckpt) ----
class EmotionFeatureExtractor(BaseFeatureExtractor):
    """
    emotion-recognition-wav2vec2-IEMOCAP（SpeechBrain）
    support：source=repository_name or local_directory; 
    savedir points to writable directory (avoid network connectivity issues)
    encode_batch already outputs segment-level embeddings (C), 
    this class performs additional mean/std aggregation at the outer level
    """
    def __init__(self, model_id: str, savedir: Optional[str] = None, sample_rate: int = 16000):
        super().__init__(sample_rate)
        self.model_id = model_id
        self.savedir = str(Path(savedir).resolve()) if savedir else None
        self.model: Optional[EncoderClassifier] = None

    def _load(self):
        if self.model is None:
            self.model = EncoderClassifier.from_hparams(
                source=self.model_id,
                savedir=self.savedir or self.model_id,
                run_opts={"device": str(self.device)}
            ).eval()

    @torch.inference_mode()
    def _embed_batch(self, segments: List[np.ndarray]) -> torch.Tensor:
        self._load()
        B = len(segments)
        T = max(len(s) for s in segments)
        batch = torch.zeros(B, T, dtype=torch.float32)
        for i, seg in enumerate(segments):
            t = len(seg)
            batch[i, :t] = torch.from_numpy(seg.astype("float32"))
        batch = batch.to(self.device)  # (B, T)
        emb: torch.Tensor = self.model.encode_batch(batch)  # (B, C)
        return emb.detach().to("cpu")

# ---- HF audio model ----
class HFAudioFeatureExtractor(BaseFeatureExtractor):
    """
    Any HF audio model (WavLM / HuBERT / W2V2 / Whisper encoder)
    Online loading (first run downloads cache), 
    segment-level output mean+std pooling -> (2C)
    """
    def __init__(self, repo_id: str, sample_rate: int = 16000,
                 segments_per_batch: int = 8, fp16: bool = True):
        super().__init__(sample_rate)
        self.repo_id = repo_id
        self.segs_per_batch = max(1, int(segments_per_batch))
        self.use_fp16 = fp16 and self.device.type == "cuda"

        os.environ.setdefault("HF_HOME", "/nfs/scratch/jtan/.hf_home")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/nfs/scratch/jtan/.hf_cache")

        self.processor = None
        self.model = None

    def _load(self):
        if self.model is not None:
            return
        torch_dtype = torch.float16 if self.use_fp16 else None
        self.processor = AutoProcessor.from_pretrained(self.repo_id)
        self.model = AutoModel.from_pretrained(self.repo_id, torch_dtype=torch_dtype)
        try:
            self.model.config.attn_implementation = "sdpa"
        except Exception:
            pass
        self.model = self.model.to(self.device).eval()

    @torch.inference_mode()
    def _embed_batch(self, segments: List[np.ndarray]) -> torch.Tensor:
        self._load()
        outs = []
        # 分批送，省显存
        for i in range(0, len(segments), self.segs_per_batch):
            batch = segments[i:i+self.segs_per_batch]
            inputs = self.processor(batch, sampling_rate=self.sample_rate,
                                    return_tensors="pt", padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=self.use_fp16):
                out = self.model(**inputs, output_hidden_states=False)
                hidden = out.last_hidden_state        # (B, T, C)
                pooled = mean_std_pool(hidden)        # (B, 2C)
            outs.append(pooled.detach().to("cpu"))
        return torch.cat(outs, dim=0)
