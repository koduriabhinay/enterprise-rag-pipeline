"""Hybrid retriever: BM25 + hashed TF-IDF, fused with RRF, then reranked.

Pipeline per query:
  1. Optional metadata filtering narrows the candidate corpus.
  2. BM25 and hashed TF-IDF each retrieve their top-N.
  3. Reciprocal Rank Fusion merges the two ranked lists.
  4. The heuristic reranker re-scores the fused shortlist.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List

import numpy as np

from ingest.models import Chunk
from retrieval.bm25_index import Bm25Index
from retrieval.filters import apply_filters
from retrieval.fusion import reciprocal_rank_fusion, top_fused
from retrieval.reranker import HeuristicReranker
from retrieval.tfidf import HashedTfidfVectorizer


@dataclass
class RetrievalResult:
    chunk: Chunk
    score: float  # final reranked score
    rrf_score: float  # fused score before reranking
    components: Dict[str, float] = field(default_factory=dict)


class HybridRetriever:
    def __init__(
        self,
        chunks: List[Chunk],
        n_features: int = 4096,
        candidate_k: int = 25,
        rrf_k: int = 60,
        rerank_pool: int = 15,
    ):
        if not chunks:
            raise ValueError("cannot build a retriever over zero chunks")
        self.chunks = chunks
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.rerank_pool = rerank_pool
        self._index_of = {c.id: i for i, c in enumerate(chunks)}

        texts = [c.text for c in chunks]
        self.vectorizer = HashedTfidfVectorizer(n_features=n_features).fit(texts)
        self.chunk_vectors = self.vectorizer.transform(texts)
        self.bm25 = Bm25Index(texts)
        self.reranker = HeuristicReranker()

    def _search_subset(
        self, query: str, chunks: List[Chunk], top_k: int
    ) -> List[RetrievalResult]:
        """Run the hybrid pipeline over an explicit chunk subset."""
        if not chunks:
            return []
        idx = [self._index_of[c.id] for c in chunks]
        sub_vectors = self.chunk_vectors[idx]

        bm25_hits = self.bm25.search(query, top_n=self.candidate_k)
        bm25_ranked = [i for i, _ in bm25_hits if i in set(idx)]

        q_vec = self.vectorizer.transform_one(query)
        cos = HashedTfidfVectorizer.cosine_similarity(q_vec, sub_vectors)
        tfidf_order = np.argsort(-cos)[: self.candidate_k]
        tfidf_ranked = [idx[i] for i in tfidf_order if cos[i] > 0.0]

        fused = reciprocal_rank_fusion([bm25_ranked, tfidf_ranked], k=self.rrf_k)
        # Restrict fusion output to this subset (paranoia: bm25 searched all).
        fused = {i: s for i, s in fused.items() if i in set(idx)}
        shortlist_idx = top_fused(fused, self.rerank_pool)
        shortlist = [self.chunks[i] for i in shortlist_idx]

        reranked = self.reranker.rerank(
            query, shortlist, self.vectorizer, self.chunk_vectors, self._index_of
        )
        return [
            RetrievalResult(
                chunk=r.chunk,
                score=r.score,
                rrf_score=fused[self._index_of[r.chunk.id]],
                components=r.components,
            )
            for r in reranked[:top_k]
        ]

    def search(
        self,
        query: str,
        top_k: int = 5,
        filters: Dict[str, Any] | None = None,
    ) -> List[RetrievalResult]:
        """Hybrid search with optional metadata filters."""
        candidates = apply_filters(self.chunks, filters)
        return self._search_subset(query, candidates, top_k)

    @property
    def corpus_size(self) -> int:
        return len(self.chunks)
