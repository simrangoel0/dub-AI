from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

import numpy as np


class EmbeddingBackend(ABC):
    """
    Abstract interface so we can swap between:
      - DummyEmbeddingBackend (for local dev)
      - Real embedding providers (Bedrock, OpenAI, etc.)
    """

    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        ...

    def embed_many(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_text(t) for t in texts]


class DummyEmbeddingBackend(EmbeddingBackend):
    """
    Deterministic pseudo-embedding based on Python's hash.

    NOTE:
      - This is NOT semantically meaningful.
      - It is only used to reduce the number of candidates
        passed to the LLM.
    """

    def embed_text(self, text: str) -> List[float]:
        h = hash(text)
        rng = np.random.default_rng(h & 0xFFFFFFFF)
        vec = rng.normal(size=128)
        norm = np.linalg.norm(vec) + 1e-8
        vec = vec / norm
        return vec.tolist()


def cosine_similarity(a: List[float], b: List[float]) -> float:
    va = np.array(a, dtype=float)
    vb = np.array(b, dtype=float)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-8)
    return float(va.dot(vb) / denom)