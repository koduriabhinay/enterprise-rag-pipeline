"""Semantic-aware chunking with overlap.

Strategy:
  1. Split the document into paragraphs (blank-line separated blocks).
  2. Track the current markdown heading so every chunk knows its section.
  3. Split long paragraphs into sentences and accumulate sentences into
     chunks up to ``max_tokens``.
  4. Carry the last ``overlap_sentences`` sentences into the next chunk so
     context is never hard-cut at a boundary.

Token counts are whitespace-based estimates, which is consistent enough for
sizing chunks without pulling in a tokenizer dependency.
"""

import re
from typing import List

from ingest.models import Document, Chunk

_HEADING_RE = re.compile(r"^(#{1,3})\s+(.*)")
_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
_WORD_RE = re.compile(r"\S+")


def estimate_tokens(text: str) -> int:
    """Rough token estimate: one whitespace-separated word counts as one token."""
    return len(_WORD_RE.findall(text))


def split_sentences(text: str) -> List[str]:
    """Split text into sentences on terminal punctuation followed by a capital."""
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    return [p.strip() for p in parts if p.strip()]


def _split_long_sentence(sentence: str, max_tokens: int) -> List[str]:
    """Hard-split an over-long sentence on word boundaries."""
    words = sentence.split()
    out, current = [], []
    for w in words:
        current.append(w)
        if len(current) >= max_tokens:
            out.append(" ".join(current))
            current = []
    if current:
        out.append(" ".join(current))
    return out


def chunk_document(
    doc: Document,
    max_tokens: int = 220,
    overlap_sentences: int = 2,
) -> List[Chunk]:
    """Chunk a document into overlapping, section-aware chunks."""
    # First pass: paragraphs annotated with their section.
    paragraphs: List[tuple[str, str]] = []  # (section, paragraph_text)
    current_section = "Introduction"
    for block in re.split(r"\n\s*\n", doc.text):
        block = block.strip()
        if not block:
            continue
        heading = _HEADING_RE.match(block.splitlines()[0])
        if heading:
            # h1 is the document title (already captured by the loader);
            # only h2/h3 headings define retrievable sections.
            if len(heading.group(1)) >= 2:
                current_section = heading.group(2).strip()
            # Keep any body text that follows the heading line on the same block.
            rest = "\n".join(block.splitlines()[1:]).strip()
            if rest:
                paragraphs.append((current_section, rest))
        else:
            paragraphs.append((current_section, block))

    # Second pass: accumulate sentences into sized chunks.
    # Second pass: accumulate sentences into sized chunks. Each chunk is
    # labeled with the section contributing the most sentences, so citations
    # point at the section the chunk is actually about.
    chunks: List[Chunk] = []
    carry: List[tuple[str, str]] = []  # (sentence, section) overlap tail

    def flush(pending: List[tuple[str, str]]) -> List[tuple[str, str]]:
        if not pending:
            return []
        sentences = [s for s, _ in pending]
        sections = [sec for _, sec in pending]
        section = max(set(sections), key=sections.count)
        text = " ".join(sentences)
        chunks.append(
            Chunk(
                id=f"{doc.source}::chunk-{len(chunks):03d}",
                text=text,
                source=doc.source,
                section=section,
                title=doc.title,
                index=len(chunks),
                token_estimate=estimate_tokens(text),
                metadata=dict(doc.metadata),
                sentences=list(pending),
            )
        )
        # Overlap: the tail of this chunk seeds the next one.
        return pending[-overlap_sentences:] if overlap_sentences > 0 else []

    pending: List[tuple[str, str]] = []
    pending_tokens = 0

    for section, paragraph in paragraphs:
        for sentence in split_sentences(paragraph):
            for piece in _split_long_sentence(sentence, max_tokens):
                piece_tokens = estimate_tokens(piece)
                if pending and pending_tokens + piece_tokens > max_tokens:
                    carry = flush(pending)
                    pending = list(carry)
                    pending_tokens = sum(estimate_tokens(s) for s, _ in pending)
                pending.append((piece, section))
                pending_tokens += piece_tokens
    flush(pending)
    return chunks


def chunk_documents(
    docs: List[Document],
    max_tokens: int = 220,
    overlap_sentences: int = 2,
) -> List[Chunk]:
    """Chunk a list of documents, preserving document order."""
    out: List[Chunk] = []
    for doc in docs:
        out.extend(chunk_document(doc, max_tokens, overlap_sentences))
    return out
