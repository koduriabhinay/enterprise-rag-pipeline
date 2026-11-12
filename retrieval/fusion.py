"""Reciprocal Rank Fusion for combining ranked lists from multiple retrievers.

RRF needs no score normalization between systems with incompatible scales
(BM25 vs cosine), which is exactly our situation:

    rrf_score(d) = sum over systems of 1 / (k + rank(d))

Documents missing from a system's list simply contribute nothing from it.
"""

from typing import Dict, List


def reciprocal_rank_fusion(
    ranked_lists: List[List[int]],
    k: int = 60,
) -> Dict[int, float]:
    """Fuse ranked doc-index lists into {doc_index: fused_score}."""
    fused: Dict[int, float] = {}
    for ranked in ranked_lists:
        for rank, doc_idx in enumerate(ranked, start=1):
            fused[doc_idx] = fused.get(doc_idx, 0.0) + 1.0 / (k + rank)
    return fused


def top_fused(fused: Dict[int, float], limit: int) -> List[int]:
    """Doc indices sorted by fused score, best first, truncated to limit."""
    return sorted(fused, key=lambda i: fused[i], reverse=True)[:limit]
