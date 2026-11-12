"""Thin wrapper around rank-bm25 with our shared tokenizer."""

from typing import List, Tuple

from rank_bm25 import BM25Okapi

from retrieval.text import tokenize


class Bm25Index:
    def __init__(self, documents: List[str]):
        self._tokenized = [tokenize(d) for d in documents]
        self._bm25 = BM25Okapi(self._tokenized)

    def search(self, query: str, top_n: int = 20) -> List[Tuple[int, float]]:
        """Return (doc_index, score) for the top_n documents, best first."""
        q_tokens = tokenize(query)
        if not q_tokens:
            return []
        scores = self._bm25.get_scores(q_tokens)
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(i, float(scores[i])) for i in order[:top_n] if scores[i] > 0.0]

    def __len__(self) -> int:
        return len(self._tokenized)
