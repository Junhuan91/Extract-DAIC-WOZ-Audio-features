import os
import re
from typing import List, Dict, Optional
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel

def _clean_transcript_text(t: str) -> str:
    # 只做“最低限度”清洗：去掉 [LAUGH] [SIGH] (xxx) 之类标注，保留大小写/标点
    t = re.sub(r"\[.*?\]", " ", t)
    t = re.sub(r"\(.*?\)", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _sentence_split(text: str) -> List[str]:
    # 轻量句子切分：按 . ! ? 加空白/行末 ；尽量不强行清洗标点
    # 需要更强可换 nltk，但这里保持零依赖
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]

def _make_chunks_by_words(sentences: List[str], max_words=200, overlap_words=40) -> List[str]:
    # 句子为单位装箱，超出就开新块；相邻块做少量词重叠，保留上下文
    chunks, cur, cur_n = [], [], 0
    for s in sentences:
        w = s.split()
        if cur_n + len(w) > max_words and cur:
            chunks.append(" ".join(cur))
            # overlap：从当前块尾部取 overlap_words 词作为下一块开头
            tail = " ".join(" ".join(cur).split()[-overlap_words:])
            cur = [tail] if tail else []
            cur_n = len(tail.split()) if tail else 0
        cur.append(s); cur_n += len(w)
    if cur:
        chunks.append(" ".join(cur))
    return chunks

def _mask_mean(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    # last_hidden: (B, L, C); attention_mask: (B, L)
    mask = attention_mask.unsqueeze(-1)  # (B, L, 1)
    summed = (last_hidden * mask).sum(dim=1)      # (B, C)
    denom = mask.sum(dim=1).clamp_min(1e-6)      # (B, 1)
    return summed / denom

class HFTextFeatureExtractor:
    """
    RoBERTa（或任意 HF 文本模型）特征提取：
    - 句子切分 + 词数装箱 + 轻重叠（默认 200 词、40 词 overlap）
    - 池化：'cls' | 'mean' | 'last4_cls' | 'last4_mean'
    - 批量编码，提高吞吐
    - 返回：{segment_features, aggregated{mean,std}, n_segments}
    """
    def __init__(self,
                 repo_id: str = "rafalposwiata/roberta-large-depression",
                 pooling: str = "last4_cls",
                 max_words: int = 200,
                 overlap_words: int = 40,
                 batch_size: int = 16,
                 l2_normalize: bool = True):
        assert pooling in {"cls", "mean", "last4_cls", "last4_mean"}
        self.repo_id = repo_id
        self.pooling = pooling
        self.max_words = max_words
        self.overlap_words = overlap_words
        self.batch_size = batch_size
        self.l2_normalize = l2_normalize

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        os.environ.setdefault("HF_HOME", "/nfs/scratch/jtan/.hf_home")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/nfs/scratch/jtan/.hf_cache")

        # roberta 系列需要 fast tokenizer；Auto 会处理 add_prefix_space 等细节
        self.tok = AutoTokenizer.from_pretrained(self.repo_id, use_fast=True)
        # 文本模型 FP16 收益有限，保持 float32 稳定性
        self.model = AutoModel.from_pretrained(self.repo_id).to(self.device).eval()

    @torch.inference_mode()
    def _encode_batch(self, texts: List[str]) -> torch.Tensor:
        enc = self.tok(texts, return_tensors="pt", truncation=True, max_length=512, padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}

        need_hidden = self.pooling.startswith("last4")
        out = self.model(**enc, output_hidden_states=need_hidden)

        if self.pooling == "cls":
            vec = out.last_hidden_state[:, 0, :]  # (B, C)
        elif self.pooling == "mean":
            vec = _mask_mean(out.last_hidden_state, enc["attention_mask"])
        else:
            # last4_*: 取最后四层平均再做 pool
            hs = out.hidden_states[-4:]  # 4 × (B, L, C)
            stack = torch.stack(hs, dim=0).mean(dim=0)  # (B, L, C)
            if self.pooling == "last4_cls":
                vec = stack[:, 0, :]
            else:  # last4_mean
                vec = _mask_mean(stack, enc["attention_mask"])

        if self.l2_normalize:
            vec = torch.nn.functional.normalize(vec, p=2, dim=-1)
        return vec.detach().cpu()  # (B, C)

    @torch.inference_mode()
    def extract_features(self, texts: List[str]) -> Optional[Dict]:
        """
        texts: 可以是若干文本块；也可以传一条长文本（建议上游先拼好一条长字符串）
        """
        if not texts:
            return None
        # 如果传进来的是“一条长文本”，做一次轻清洗+句子切分+装箱
        if len(texts) == 1:
            long_text = _clean_transcript_text(texts[0])
            sents = _sentence_split(long_text)
            chunks = _make_chunks_by_words(sents, self.max_words, self.overlap_words)
        else:
            # 已经是块列表：也做最小清洗
            chunks = [_clean_transcript_text(t) for t in texts]

        chunks = [c for c in chunks if c]
        if not chunks:
            return None

        embs = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i:i + self.batch_size]
            embs.append(self._encode_batch(batch))
        seg = torch.cat(embs, dim=0).numpy()  # (N, C)

        aggregated = {
            "mean": seg.mean(axis=0),
            "std":  seg.std(axis=0),
        }
        return {
            "segment_features": seg,
            "aggregated": aggregated,
            "n_segments": int(seg.shape[0]),
        }
