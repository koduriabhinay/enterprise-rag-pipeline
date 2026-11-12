"""Lightweight reranker: no cross-encoder needed.

Each candidate chunk is scored on three explainable signals, each normalized
to [0, 1], then combined with configurable weights:

  * tfidf_cosine  - cosine similarity between query and chunk vectors
  * term_overlap  - fraction of query content-words present in the chunk
  * position      - how early the first query-term match appears in the chunk
                    (answers near the start of a chunk tend to be on-topic)

A production deployment would replace this with a cross-encoder; the
interface (rerank(query, candidates) -> scored list) is deliberately the same
shape so the swap is one class.
"""

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from ingest.models import Chunk
from retrieval.text import tokenize
from retrieval.tfidf import HashedTfidfVectorizer

DEFAULT_WEIGHTS = {
    "tfidf_cosine": 0.45,
    "term_overlap": 0.35,
    "position": 0.20,
}


@dataclass
class RerankedChunk:
    chunk: Chunk
    score: float
    components: Dict[str, float]


class HeuristicReranker:
    def __init__(self, weights: Dict[str, float] | None = None):
        self.weights = dict(DEFAULT_WEIGHTS)
        if weights:
            self.weights.update(weights)

    def rerank(
        self,
        query: str,
        candidates: List[Chunk],
        vectorizer: HashedTfidfVectorizer,
        chunk_vectors: np.ndarray,
        chunk_index: Dict[str, int],
    ) -> List[RerankedChunk]:
        """Score and sort candidate chunks for the query, best first."""
        q_tokens = tokenize(query)
        q_set = set(q_tokens)
        q_vec = vectorizer.transform_one(query)

        scored: List[RerankedChunk] = []
        for chunk in candidates:
            c_tokens = tokenize(chunk.text)
            c_set = set(c_tokens)

            tfidf_cosine = float(np.dot(q_vec, chunk_vectors[chunk_index[chunk.id]]))
            tfidf_cosine = max(0.0, min(1.0, tfidf_cosine))

            overlap = len(q_set & c_set) / max(len(q_set), 1)

            lowered = chunk.text.lower()
            first_hit = min(
                (lowered.find(t) for t in q_set if lowered.find(t) != -1),
                default=-1,
            )
            if first_hit == -1 or not lowered:
                position = 0.0
            else:
                position = 1.0 / (1.0 + first_hit / max(len(lowered), 1))

            components = {
                "tfidf_cosine": round(tfidf_cosine, 4),
                "term_overlap": round(overlap, 4),
                "position": round(position, 4),
            }
            score = sum(self.weights[k] * components[k] for k in self.weights)
            scored.append(RerankedChunk(chunk=chunk, score=score, components=components))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored
