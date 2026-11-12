"""Lightweight, embedding-free evaluation metrics for the RAG pipeline.

  * faithfulness      - fraction of answer sentences supported by the
                        retrieved context (content-word overlap heuristic)
  * answer_relevance  - cosine similarity between question and answer using
                        the pipeline's hashed TF-IDF vectorizer
  * context_precision - fraction of retrieved chunks that are relevant:
                        a chunk counts as relevant if its source is in the
                        gold sources, or (when no gold is given) if it shares
                        at least two content words with the question

All scores are in [0, 1]. These are proxies, not LLM-judge scores; they run
offline in milliseconds and are honest about what they measure.
"""

from dataclasses import dataclass, field
from typing import Dict, List

from generation.answer import Answer, RAGPipeline
from ingest.chunker import split_sentences
from retrieval.text import content_words


def faithfulness(answer_text: str, context_texts: List[str]) -> float:
    """Share of answer sentences with substantive support in the context."""
    sentences = [s for s in split_sentences(answer_text) if len(s) >= 25]
    if not sentences:
        return 0.0
    context_vocab = set()
    for ctx in context_texts:
        context_vocab.update(content_words(ctx))
    if not context_vocab:
        return 0.0
    supported = 0
    for sent in sentences:
        words = set(content_words(sent))
        if not words:
            continue
        if len(words & context_vocab) / len(words) >= 0.5:
            supported += 1
    return supported / len(sentences)


def answer_relevance(question: str, answer_text: str, pipeline: RAGPipeline) -> float:
    """Cosine similarity between question and answer in TF-IDF space."""
    if pipeline.retriever is None:
        raise RuntimeError("pipeline has no retriever")
    vec = pipeline.retriever.vectorizer
    q = vec.transform_one(question)
    a = vec.transform_one(answer_text)
    return float(max(0.0, min(1.0, q @ a)))


def context_precision(
    question: str,
    retrieved_sources: List[str],
    retrieved_texts: List[str],
    gold_sources: List[str] | None = None,
) -> float:
    """Share of retrieved chunks deemed relevant to the question."""
    if not retrieved_sources:
        return 0.0
    q_words = set(content_words(question))
    gold = set(gold_sources or [])
    relevant = 0
    for source, text in zip(retrieved_sources, retrieved_texts):
        if gold and source in gold:
            relevant += 1
        elif len(q_words & set(content_words(text))) >= 2:
            relevant += 1
    return relevant / len(retrieved_sources)


@dataclass
class EvalItemResult:
    question: str
    faithfulness: float
    answer_relevance: float
    context_precision: float
    latency_ms: float
    blocked: bool = False


@dataclass
class EvalReport:
    items: List[EvalItemResult] = field(default_factory=list)

    def means(self) -> Dict[str, float]:
        ok = [i for i in self.items if not i.blocked]
        if not ok:
            return {
                "faithfulness": 0.0,
                "answer_relevance": 0.0,
                "context_precision": 0.0,
                "latency_ms": 0.0,
            }
        return {
            "faithfulness": sum(i.faithfulness for i in ok) / len(ok),
            "answer_relevance": sum(i.answer_relevance for i in ok) / len(ok),
            "context_precision": sum(i.context_precision for i in ok) / len(ok),
            "latency_ms": sum(i.latency_ms for i in ok) / len(ok),
        }


def evaluate_item(
    pipeline: RAGPipeline,
    question: str,
    gold_sources: List[str] | None = None,
    top_k: int = 5,
) -> EvalItemResult:
    answer: Answer = pipeline.ask(question, top_k=top_k)
    if answer.blocked:
        return EvalItemResult(
            question=question,
            faithfulness=0.0,
            answer_relevance=0.0,
            context_precision=0.0,
            latency_ms=answer.latency_ms,
            blocked=True,
        )
    # Raw chunk texts for the faithfulness check. Retrieval is deterministic,
    # so re-running the same query returns the same chunks used for the answer.
    chunks = pipeline.retriever.search(question, top_k=top_k)
    context_texts = [c.chunk.text for c in chunks]
    return EvalItemResult(
        question=question,
        faithfulness=faithfulness(answer.answer_text, context_texts),
        answer_relevance=answer_relevance(question, answer.answer_text, pipeline),
        context_precision=context_precision(
            question,
            [c.chunk.source for c in chunks],
            context_texts,
            gold_sources,
        ),
        latency_ms=answer.latency_ms,
    )


def run_evaluation(
    pipeline: RAGPipeline,
    dataset: List[Dict],
    top_k: int = 5,
) -> EvalReport:
    report = EvalReport()
    for item in dataset:
        report.items.append(
            evaluate_item(
                pipeline,
                item["question"],
                item.get("gold_sources"),
                top_k=top_k,
            )
        )
    return report
