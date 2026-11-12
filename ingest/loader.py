"""Document loaders for txt, markdown, and PDF files."""

import re
from pathlib import Path
from typing import List

from ingest.models import Document

SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf"}


def _title_from_markdown(text: str, fallback: str) -> str:
    """Use the first top-level heading as the document title."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            return stripped[2:].strip()
    return fallback


def _load_text_file(path: Path) -> Document:
    text = path.read_text(encoding="utf-8")
    title = (
        _title_from_markdown(text, path.stem.replace("-", " ").replace("_", " ").title())
        if path.suffix == ".md"
        else path.stem.replace("-", " ").replace("_", " ").title()
    )
    return Document(
        source=path.name,
        title=title,
        text=text,
        metadata={"filetype": path.suffix.lstrip("."), "path": str(path)},
    )


def _load_pdf(path: Path) -> Document:
    from pypdf import PdfReader  # imported lazily so txt/md never need pypdf

    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        # Prefix a synthetic section marker so the chunker can track provenance.
        pages.append(f"## Page {i}\n\n{page_text.strip()}")
    text = "\n\n".join(pages)
    return Document(
        source=path.name,
        title=path.stem.replace("-", " ").replace("_", " ").title(),
        text=text,
        metadata={"filetype": "pdf", "path": str(path), "pages": len(reader.pages)},
    )


def load_document(path: str | Path) -> Document:
    """Load a single file into a Document. Raises ValueError for unsupported types."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"no such file: {p}")
    suffix = p.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(p)
    if suffix in {".txt", ".md"}:
        return _load_text_file(p)
    raise ValueError(f"unsupported file type: {suffix} (supported: {sorted(SUPPORTED_SUFFIXES)})")


def load_directory(directory: str | Path) -> List[Document]:
    """Load every supported file in a directory (non-recursive), sorted by name."""
    d = Path(directory)
    if not d.is_dir():
        raise NotADirectoryError(f"no such directory: {d}")
    docs = []
    for p in sorted(d.iterdir()):
        if p.is_file() and p.suffix.lower() in SUPPORTED_SUFFIXES:
            docs.append(load_document(p))
    return docs
