"""Metadata filtering for retrieval (e.g. restrict to one policy document)."""

from typing import Any, Dict, List

from ingest.models import Chunk


def _chunk_field(chunk: Chunk, key: str) -> Any:
    if key == "source":
        return chunk.source
    if key == "section":
        return chunk.section
    if key == "title":
        return chunk.title
    return chunk.metadata.get(key)


def apply_filters(chunks: List[Chunk], filters: Dict[str, Any] | None) -> List[Chunk]:
    """Keep chunks whose fields match all filters.

    A filter value may be a scalar (exact match) or a list (match any).
    Unknown filter keys match against chunk metadata and simply exclude
    chunks that do not carry the key.
    """
    if not filters:
        return list(chunks)
    out = []
    for chunk in chunks:
        keep = True
        for key, wanted in filters.items():
            actual = _chunk_field(chunk, key)
            if isinstance(wanted, (list, tuple, set)):
                if actual not in wanted:
                    keep = False
                    break
            elif actual != wanted:
                keep = False
                break
        if keep:
            out.append(chunk)
    return out
