"""Tests for document chunking."""

from ingest.chunker import chunk_document, estimate_tokens, split_sentences
from ingest.models import Document


def _doc(text: str, source: str = "policy.md") -> Document:
    return Document(source=source, title="Policy", text=text)


def test_sentences_split_on_punctuation():
    sents = split_sentences("First sentence. Second sentence! Is this third?")
    assert len(sents) == 3


def test_chunk_respects_max_tokens():
    text = " ".join(f"Sentence number {i} with some filler words." for i in range(40))
    chunks = chunk_document(_doc(text), max_tokens=30, overlap_sentences=0)
    assert len(chunks) > 1
    for c in chunks:
        assert c.token_estimate <= 30


def test_overlap_carries_sentences_forward():
    text = " ".join(f"Sentence number {i}." for i in range(10))
    chunks = chunk_document(_doc(text), max_tokens=12, overlap_sentences=2)
    assert len(chunks) >= 2
    # The tail of chunk 0 should appear at the head of chunk 1.
    assert "Sentence number 3." in chunks[0].text
    assert chunks[1].text.startswith("Sentence number 2.")


def test_section_tracking_from_markdown_headings():
    elig = " ".join(f"Eligibility detail sentence {i}." for i in range(12))
    equip = " ".join(f"Equipment detail sentence {i}." for i in range(12))
    text = f"# Title\n\n## Eligibility\n\n{elig}\n\n## Equipment\n\n{equip}"
    chunks = chunk_document(_doc(text), max_tokens=15, overlap_sentences=0)
    assert len(chunks) > 2
    sections = {c.section for c in chunks}
    assert "Eligibility" in sections
    assert "Equipment" in sections


def test_chunk_ids_are_unique_and_ordered():
    chunks = chunk_document(_doc("One. Two. Three."), max_tokens=50)
    ids = [c.id for c in chunks]
    assert len(ids) == len(set(ids))
    assert all(c.index == i for i, c in enumerate(chunks))


def test_estimate_tokens_counts_words():
    assert estimate_tokens("one two three") == 3
