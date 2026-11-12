"""Hashed TF-IDF vectors with numpy (no embedding model required).

The hashing trick maps every token to one of ``n_features`` buckets with
md5, so the vocabulary is unbounded and no model download is needed. IDF is
computed from the indexed corpus. Vectors are L2-normalized, so the dot
product equals cosine similarity.

This is intentionally a *lexical* dense-style signal: it captures term
importance weighting (unlike raw BM25 scores it is normalized and directly
comparable), and it pairs well with BM25 in the hybrid fusion below. See the
README's "production upgrade path" for swapping in sentence-transformers.
"""

import hashlib
from typing import List

import numpy as np

from retrieval.text import tokenize


def _hash_token(token: str, n_features: int) -> int:
    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % n_features


class HashedTfidfVectorizer:
    def __init__(self, n_features: int = 4096):
        self.n_features = n_features
        self.idf_: np.ndarray | None = None
        self.n_docs_: int = 0

    def fit(self, documents: List[str]) -> "HashedTfidfVectorizer":
        """Compute IDF weights from a corpus of raw texts."""
        tokenized = [tokenize(d) for d in documents]
        df = np.zeros(self.n_features, dtype=np.float64)
        for tokens in tokenized:
            for bucket in {_hash_token(t, self.n_features) for t in tokens}:
                df[bucket] += 1
        n = max(len(documents), 1)
        # Smoothed IDF, always positive.
        self.idf_ = np.log((1.0 + n) / (1.0 + df)) + 1.0
        self.n_docs_ = n
        return self

    def transform(self, documents: List[str]) -> np.ndarray:
        """Return L2-normalized TF-IDF vectors, shape (n_docs, n_features)."""
        if self.idf_ is None:
            raise RuntimeError("vectorizer is not fitted; call fit() first")
        mat = np.zeros((len(documents), self.n_features), dtype=np.float32)
        for i, text in enumerate(documents):
            tokens = tokenize(text)
            if not tokens:
                continue
            counts: dict[int, int] = {}
            for t in tokens:
                b = _hash_token(t, self.n_features)
                counts[b] = counts.get(b, 0) + 1
            for bucket, c in counts.items():
                # Sublinear TF scaling dampens very frequent terms.
                mat[i, bucket] = (1.0 + np.log(c)) * self.idf_[bucket]
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return (mat / norms).astype(np.float32)

    def transform_one(self, text: str) -> np.ndarray:
        return self.transform([text])[0]

    @staticmethod
    def cosine_similarity(query_vec: np.ndarray, doc_matrix: np.ndarray) -> np.ndarray:
        """Cosine similarity for L2-normalized vectors (dot product)."""
        return doc_matrix @ query_vec
