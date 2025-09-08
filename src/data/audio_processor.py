from pathlib import Path
import numpy as np
import soundfile as sf
import torchaudio

class AudioProcessor:
    def __init__(self, sample_rate=16000, segment_length=30, overlap=15):
        self.sr = sample_rate
        self.seg_len = int(segment_length * sample_rate)
        hop = segment_length - overlap
        self.hop_len = int(max(1, hop) * sample_rate)

    def _load_mono_16k(self, path: str) -> np.ndarray:
        # torchaudio more stable（large file/multi audio channel）,but soundfile is fine
        wav, sr = torchaudio.load(path)  # (C, T)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav.squeeze(0).cpu().numpy()

    def process_long_audio(self, path: str):
        x = self._load_mono_16k(path)
        n = len(x)
        segs = []
        i = 0
        while i + self.seg_len <= n:
            segs.append(x[i:i+self.seg_len])
            i += self.hop_len
        if not segs and n > 0:
            # short audio: complete segment
            segs.append(x)
        return segs
