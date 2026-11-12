"""Tests for hybrid retrieval: TF-IDF, BM25, RRF fusion, reranking, filters."""

import numpy as np
import pytest

from ingest.chunker import chunk_document
from ingest.models import Document
from retrieval.filters import apply_filters
from retrieval.fusion import reciprocal_rank_fusion
from retrieval.pipeline import HybridRetriever
from retrieval.tfidf import HashedTfidfVectorizer


def _chunks():
    docs = [
        Document(
            source="cats.md",
            title="Cats",
            text="## Care\n\nCats need fresh water daily. Cats enjoy warm sleeping spots. "
            "Veterinary checkups keep cats healthy every year.",
        ),
        Document(
            source="dogs.md",
            title="Dogs",
            text="## Care\n\nDogs need long walks every day. Dogs enjoy playing fetch in parks. "
            "Training classes help dogs learn obedience.",
        ),
        Document(
            source="fish.md",
            title="Fish",
            text="## Care\n\nFish tanks require clean filtered water. Tropical fish need warm water. "
            "Feed fish small portions twice daily.",
        ),
    ]
    out = []
    for d in docs:
        out.extend(chunk_document(d, max_tokens=100, overlap_sentences=0))
    return out


@pytest.fixture()
def retriever():
    return HybridRetriever(_chunks())


def test_relevant_doc_ranked_first(retriever):
    results = retriever.search("How often should cats visit the vet?", top_k=3)
    assert results
    assert results[0].chunk.source == "cats.md"


def test_scores_descend(retriever):
    results = retriever.search("dogs playing fetch", top_k=3)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_metadata_filter_restricts_corpus(retriever):
    results = retriever.search("daily care", top_k=5, filters={"source": "fish.md"})
    assert results
    assert all(r.chunk.source == "fish.md" for r in results)


def test_metadata_filter_list_value(retriever):
    results = retriever.search("daily care", top_k=5, filters={"source": ["cats.md", "dogs.md"]})
    assert all(r.chunk.source in {"cats.md", "dogs.md"} for r in results)


def test_filter_no_match_returns_empty(retriever):
    assert retriever.search("daily care", filters={"source": "nope.md"}) == []


def test_rrf_prefers_consensus():
    # doc 0 is ranked first by both systems, so it must win the fusion.
    fused = reciprocal_rank_fusion([[0, 1, 2], [0, 2, 1]])
    assert max(fused, key=fused.get) == 0
    assert fused[0] > fused[1]
    assert fused[0] > fused[2]


def test_tfidf_vectors_are_normalized():
    vec = HashedTfidfVectorizer(n_features=256).fit(["cats dogs", "fish tanks"])
    mat = vec.transform(["cats dogs", "fish tanks"])
    assert mat.shape == (2, 256)
    np.testing.assert_allclose(np.linalg.norm(mat, axis=1), 1.0, atol=1e-5)


def test_empty_query_returns_no_results(retriever):
    assert retriever.search("the and of", top_k=3) == []


def test_retriever_rejects_empty_corpus():
    with pytest.raises(ValueError):
        HybridRetriever([])


def test_apply_filters_unknown_key_excludes():
    chunks = _chunks()
    assert apply_filters(chunks, {"nonexistent": "x"}) == []
