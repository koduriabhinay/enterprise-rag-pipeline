"""Core data models for the ingestion stage."""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Tuple


@dataclass
class Document:
    """A single source document before chunking."""

    source: str  # filename or identifier, e.g. "remote-work-policy.md"
    title: str  # human readable title
    text: str  # full raw text
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """A retrievable unit of text with provenance metadata."""

    id: str  # e.g. "remote-work-policy.md::chunk-003"
    text: str
    source: str  # originating document filename
    section: str  # markdown heading or "page N" for PDFs
    title: str  # document title
    index: int  # position within the document
    token_estimate: int
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Every sentence paired with the section it came from, so citations can
    # point at the exact section even when a chunk spans several.
    sentences: List[Tuple[str, str]] = field(default_factory=list)

    def cite(self) -> str:
        """Short human readable citation for this chunk."""
        return f"{self.title} ({self.section})"
