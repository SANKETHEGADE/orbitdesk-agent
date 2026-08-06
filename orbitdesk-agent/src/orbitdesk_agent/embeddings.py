"""Embedding backends used for retrieval.

Both backends implement the same tiny interface (`encode(list[str]) ->
np.ndarray`), so the rest of the pipeline (retrieval node, tests) never
needs to know which one is in use. `RealEmbedder` is the actual Hugging
Face model used for the graded run. `MockEmbedder` is a deterministic,
network-free stand-in used by the automated tests and by anyone without
model weights downloaded yet, so the graph's routing logic can be verified
without downloading anything.
"""

from __future__ import annotations

import hashlib
import re
import time
from abc import ABC, abstractmethod
from typing import List

import numpy as np

from .config import MODEL_CONFIG


class BaseEmbedder(ABC):
    name: str = "base"

    @abstractmethod
    def encode(self, texts: List[str]) -> np.ndarray:
        ...


class RealEmbedder(BaseEmbedder):
    """Wraps sentence-transformers/all-MiniLM-L6-v2 (or config override)."""

    def __init__(self) -> None:
        from sentence_transformers import SentenceTransformer 

        self.name = MODEL_CONFIG.embedding_model_name
        t0 = time.time()
        self._model = SentenceTransformer(
            MODEL_CONFIG.embedding_model_name,
            revision=MODEL_CONFIG.embedding_model_revision,
        )
        self.load_time_seconds = time.time() - t0

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.asarray(self._model.encode(texts, normalize_embeddings=True))


class MockEmbedder(BaseEmbedder):
    """Deterministic bag-of-words-ish embedding with no external calls or
    downloads. Good enough to make semantically-similar OrbitDesk sentences
    score higher than unrelated ones, which is all the routing tests need.
    """

    name = "mock-hashing-embedder"
    _DIM = 256

    def __init__(self) -> None:
        self.load_time_seconds = 0.0

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self._DIM, dtype=np.float32)
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        for tok in tokens:
            idx = int(hashlib.md5(tok.encode()).hexdigest(), 16) % self._DIM
            vec[idx] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def encode(self, texts: List[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts])


def cosine_sim_matrix(query_vecs: np.ndarray, corpus_vecs: np.ndarray) -> np.ndarray:
    """Both inputs are assumed to already be L2-normalised."""
    return query_vecs @ corpus_vecs.T


def get_embedder(mock: bool) -> BaseEmbedder:
    if mock:
        return MockEmbedder()
    return RealEmbedder()
