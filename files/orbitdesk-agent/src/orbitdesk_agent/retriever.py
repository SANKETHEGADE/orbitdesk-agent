"""Wraps an embedder + the loaded chunks into a tiny local vector index.

No managed vector database is used, per the assignment -- just numpy cosine
similarity over a small, in-memory matrix, which is more than adequate for
~40 short chunks.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

from .config import RETRIEVAL_CONFIG
from .embeddings import BaseEmbedder, cosine_sim_matrix
from .kb import Chunk, load_all_chunks
from .state import RetrievedChunk


class Retriever:
    def __init__(self, data_dir: Path, embedder: BaseEmbedder) -> None:
        self.embedder = embedder
        self.chunks: List[Chunk] = load_all_chunks(data_dir)
        corpus_texts = [f"{c.title}\n{c.text}" for c in self.chunks]
        self.corpus_vectors = embedder.encode(corpus_texts)

    def search(self, query: str, top_k: int | None = None) -> List[RetrievedChunk]:
        top_k = top_k or RETRIEVAL_CONFIG.top_k
        query_vec = self.embedder.encode([query])
        sims = cosine_sim_matrix(query_vec, self.corpus_vectors)[0]
        order = sims.argsort()[::-1][:top_k]
        results: List[RetrievedChunk] = []
        for idx in order:
            chunk = self.chunks[idx]
            results.append(
                RetrievedChunk(
                    source_id=chunk.source_id,
                    title=chunk.title,
                    text=chunk.text,
                    score=float(sims[idx]),
                    superseded=chunk.superseded,
                )
            )
        return results
