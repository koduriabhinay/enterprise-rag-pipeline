"""End-to-end RAG pipeline: guardrails -> retrieval -> generation."""

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

from generation.extractive import Citation, generate_extractive
from generation.llm import LlmConfig, generate_abstractive
from guardrails.injection import check_injection
from guardrails.pii import redact_pii
from ingest.chunker import chunk_document, chunk_documents, split_sentences
from ingest.loader import load_directory, load_document
from ingest.models import Chunk, Document
from retrieval.pipeline import HybridRetriever, RetrievalResult


@dataclass
class Answer:
    question: str
    answer_text: str
    citations: List[Citation] = field(default_factory=list)
    retrieved: List[Dict[str, Any]] = field(default_factory=list)
    blocked: bool = False
    block_reasons: List[str] = field(default_factory=list)
    pii_findings: List[Dict[str, int]] = field(default_factory=list)
    latency_ms: float = 0.0
    mode: str = "extractive"


class RAGPipeline:
    """Assemble ingestion, retrieval, guardrails, and generation."""

    def __init__(self, docs_dir: str | Path | None = None):
        self.chunks: List[Chunk] = []
        self.retriever: HybridRetriever | None = None
        if docs_dir is not None:
            self.ingest_directory(docs_dir)

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest_directory(self, directory: str | Path) -> int:
        docs = load_directory(directory)
        new_chunks = chunk_documents(docs)
        self.chunks.extend(new_chunks)
        self._rebuild_index()
        return len(new_chunks)

    def ingest_file(self, path: str | Path) -> int:
        doc = load_document(path)
        new_chunks = chunk_document(doc)
        self.chunks.extend(new_chunks)
        self._rebuild_index()
        return len(new_chunks)

    def add_text(self, text: str, source: str, title: str | None = None) -> int:
        doc = Document(
            source=source,
            title=title or source,
            text=text,
            metadata={"filetype": "inline"},
        )
        new_chunks = chunk_document(doc)
        self.chunks.extend(new_chunks)
        self._rebuild_index()
        return len(new_chunks)

    def _rebuild_index(self) -> None:
        if self.chunks:
            self.retriever = HybridRetriever(self.chunks)

    # ------------------------------------------------------------------ #
    # Question answering
    # ------------------------------------------------------------------ #
    def ask(
        self,
        question: str,
        top_k: int = 5,
        filters: Dict[str, Any] | None = None,
        mode: str = "extractive",
    ) -> Answer:
        started = time.perf_counter()

        # 1. Guardrail: prompt-injection screening on the raw question.
        verdict = check_injection(question)
        if verdict.blocked:
            return Answer(
                question=question,
                answer_text=(
                    "This question was blocked by the input guardrail because it "
                    "looks like a prompt-injection attempt."
                ),
                blocked=True,
                block_reasons=verdict.reasons,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

        if self.retriever is None:
            raise RuntimeError("no documents indexed; ingest something first")

        # 2. Retrieval.
        results = self.retriever.search(question, top_k=top_k, filters=filters)

        # 3. Guardrail: redact PII from retrieved context before generation.
        # Redaction runs per sentence because extractive generation reads
        # chunk.sentences, not chunk.text.
        all_findings: List[Dict[str, int]] = []
        safe_results: List[RetrievalResult] = []
        for res in results:
            pairs = res.chunk.sentences or [
                (s, res.chunk.section) for s in split_sentences(res.chunk.text)
            ]
            safe_sentences: List[Tuple[str, str]] = []
            for sent, sec in pairs:
                rep = redact_pii(sent)
                all_findings.extend(rep.findings)
                safe_sentences.append((rep.redacted_text, sec))
            safe_chunk = Chunk(
                id=res.chunk.id,
                text=" ".join(s for s, _ in safe_sentences),
                source=res.chunk.source,
                section=res.chunk.section,
                title=res.chunk.title,
                index=res.chunk.index,
                token_estimate=res.chunk.token_estimate,
                metadata=res.chunk.metadata,
                sentences=safe_sentences,
            )
            safe_results.append(
                RetrievalResult(
                    chunk=safe_chunk,
                    score=res.score,
                    rrf_score=res.rrf_score,
                    components=res.components,
                )
            )

        # 4. Generation.
        used_mode = "extractive"
        generated = generate_extractive(question, safe_results)
        if mode == "abstractive":
            llm_config = LlmConfig.from_env()
            if llm_config is not None and safe_results:
                passages = [
                    f"[{i}] ({r.chunk.source} / {r.chunk.section}): {r.chunk.text}"
                    for i, r in enumerate(safe_results, start=1)
                ]
                passage_citations = [
                    Citation(
                        label=f"[{i}]",
                        source=r.chunk.source,
                        section=r.chunk.section,
                        title=r.chunk.title,
                    )
                    for i, r in enumerate(safe_results, start=1)
                ]
                try:
                    answer_text = generate_abstractive(
                        question, "\n\n".join(passages), passage_citations, llm_config
                    )
                    citations = passage_citations
                    used_mode = "abstractive"
                except Exception:
                    # Any LLM failure degrades gracefully to extractive.
                    answer_text, citations = generated.text, generated.citations
            else:
                answer_text, citations = generated.text, generated.citations
        else:
            answer_text, citations = generated.text, generated.citations

        # 5. Guardrail: redact PII that may appear in the final answer.
        final_report = redact_pii(answer_text)
        all_findings.extend(final_report.findings)

        retrieved_summary = [
            {
                "source": r.chunk.source,
                "section": r.chunk.section,
                "score": round(r.score, 4),
            }
            for r in safe_results
        ]
        return Answer(
            question=question,
            answer_text=final_report.redacted_text,
            citations=citations,
            retrieved=retrieved_summary,
            pii_findings=all_findings,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            mode=used_mode,
        )

    @property
    def stats(self) -> Dict[str, Any]:
        sources = sorted({c.source for c in self.chunks})
        return {
            "documents": len(sources),
            "chunks": len(self.chunks),
            "sources": sources,
        }
