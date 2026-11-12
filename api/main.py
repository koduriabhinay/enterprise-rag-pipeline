"""FastAPI service exposing the RAG pipeline.

Endpoints:
  GET  /health   liveness probe
  GET  /stats    indexed document/chunk counts
  POST /ask      question answering with citations
  POST /ingest   add a document (inline text or a directory path)
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from generation.answer import RAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DOCS = PROJECT_ROOT / "data_samples" / "policies"

app = FastAPI(
    title="Enterprise RAG Pipeline",
    description="Hybrid retrieval (BM25 + hashed TF-IDF) Q&A over enterprise documents.",
    version="1.0.0",
)

pipeline = RAGPipeline(DEFAULT_DOCS if DEFAULT_DOCS.is_dir() else None)


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The question to answer")
    top_k: int = Field(5, ge=1, le=20, description="Chunks to retrieve")
    filters: Optional[Dict[str, Any]] = Field(
        None, description='Metadata filters, e.g. {"source": "expense-policy.md"}'
    )
    mode: str = Field(
        "extractive",
        description='"extractive" (default, offline) or "abstractive" (needs LLM env config)',
    )


class CitationOut(BaseModel):
    label: str
    source: str
    section: str
    title: str


class AskResponse(BaseModel):
    question: str
    answer: str
    citations: List[CitationOut]
    retrieved: List[Dict[str, Any]]
    blocked: bool
    block_reasons: List[str]
    pii_findings: List[Dict[str, int]]
    latency_ms: float
    mode: str


class IngestRequest(BaseModel):
    text: Optional[str] = Field(None, description="Inline document text to index")
    source: Optional[str] = Field(None, description="Filename for inline text")
    title: Optional[str] = Field(None, description="Title for inline text")
    directory: Optional[str] = Field(None, description="Directory of documents to index")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/stats")
def stats() -> Dict[str, Any]:
    return pipeline.stats


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if req.mode not in {"extractive", "abstractive"}:
        raise HTTPException(status_code=400, detail="mode must be extractive or abstractive")
    try:
        ans = pipeline.ask(req.question, top_k=req.top_k, filters=req.filters, mode=req.mode)
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return AskResponse(
        question=ans.question,
        answer=ans.answer_text,
        citations=[CitationOut(**c.__dict__) for c in ans.citations],
        retrieved=ans.retrieved,
        blocked=ans.blocked,
        block_reasons=ans.block_reasons,
        pii_findings=ans.pii_findings,
        latency_ms=round(ans.latency_ms, 2),
        mode=ans.mode,
    )


@app.post("/ingest")
def ingest(req: IngestRequest) -> Dict[str, Any]:
    if req.directory:
        added = pipeline.ingest_directory(req.directory)
        return {"ingested_chunks": added, **pipeline.stats}
    if req.text and req.source:
        added = pipeline.add_text(req.text, source=req.source, title=req.title)
        return {"ingested_chunks": added, **pipeline.stats}
    raise HTTPException(
        status_code=400,
        detail='provide either "directory" or both "text" and "source"',
    )
