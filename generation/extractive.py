"""Extractive answer generation with citations.

The default mode needs no API key: it selects the highest-scoring sentences
from the retrieved chunks (query term overlap weighted by the chunk's
retrieval score), deduplicates near-identical sentences, and attaches
numbered citations that map to source documents and sections.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List

from ingest.chunker import split_sentences
from retrieval.pipeline import RetrievalResult
from retrieval.text import tokenize


@dataclass
class Citation:
    label: str  # "[1]", "[2]", ...
    source: str
    section: str
    title: str


@dataclass
class GeneratedAnswer:
    text: str
    citations: List[Citation] = field(default_factory=list)


def _sentence_score(sentence: str, query_terms: set, chunk_score: float) -> float:
    tokens = set(tokenize(sentence))
    if not tokens or not query_terms:
        return 0.0
    overlap = len(tokens & query_terms) / len(query_terms)
    coverage = len(tokens & query_terms) / len(tokens)
    # Blend recall-style overlap with precision-style coverage, scaled by
    # how well the parent chunk matched.
    return (0.7 * overlap + 0.3 * coverage) * (0.5 + 0.5 * chunk_score)


def _near_duplicate(a: str, b: str, threshold: float = 0.8) -> bool:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return False
    jaccard = len(ta & tb) / len(ta | tb)
    return jaccard >= threshold


def generate_extractive(
    question: str,
    results: List[RetrievalResult],
    max_sentences: int = 4,
) -> GeneratedAnswer:
    """Build an extractive answer with citations from retrieval results."""
    if not results:
        return GeneratedAnswer(
            text="I could not find relevant information in the indexed documents."
        )

    query_terms = set(tokenize(question))
    scored: List[tuple[float, str, RetrievalResult, str]] = []
    for res in results:
        # Prefer the chunker's sentence/section pairs so each cited sentence
        # carries its true section; fall back to re-splitting for ad-hoc chunks.
        pairs = res.chunk.sentences or [
            (s, res.chunk.section) for s in split_sentences(res.chunk.text)
        ]
        for sent, sec in pairs:
            if len(sent) < 25:  # skip fragments and headings
                continue
            scored.append((_sentence_score(sent, query_terms, res.score), sent, res, sec))
    scored.sort(key=lambda t: t[0], reverse=True)

    # Deduplicate near-identical sentences (overlap between chunks is common).
    picked: List[tuple[str, RetrievalResult, str]] = []
    for score, sent, res, sec in scored:
        if score <= 0.0:
            continue
        if any(_near_duplicate(sent, p[0]) for p in picked):
            continue
        picked.append((sent, res, sec))
        if len(picked) >= max_sentences:
            break

    if not picked:
        return GeneratedAnswer(
            text="I could not find relevant information in the indexed documents."
        )

    # Number citations in order of first appearance; a chunk cited from two
    # different sections gets one citation per section.
    citation_order: List[tuple[RetrievalResult, str]] = []
    for _, res, sec in picked:
        key = (res.chunk.id, sec)
        if key not in {(r.chunk.id, s) for r, s in citation_order}:
            citation_order.append((res, sec))
    label_of: Dict[tuple[int, str], str] = {
        (id(r), s): f"[{i + 1}]" for i, (r, s) in enumerate(citation_order)
    }

    sentences = [f"{sent} {label_of[(id(res), sec)]}" for sent, res, sec in picked]
    citations = [
        Citation(
            label=f"[{i + 1}]",
            source=res.chunk.source,
            section=sec,
            title=res.chunk.title,
        )
        for i, (res, sec) in enumerate(citation_order)
    ]
    return GeneratedAnswer(text=" ".join(sentences), citations=citations)
