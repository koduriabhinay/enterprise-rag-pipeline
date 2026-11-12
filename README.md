# Enterprise RAG Pipeline

A production-style retrieval-augmented generation pipeline over enterprise documents, built to run fully offline with zero API keys and no heavy ML dependencies. It answers questions from a corpus of company policies with cited sources, guarded by PII redaction and prompt-injection screening.

The interesting design choice: there are no embedding models here. Retrieval is hybrid lexical (BM25 + hashed TF-IDF over numpy), fused with reciprocal rank fusion and a heuristic reranker. That keeps `pip install` under a minute and inference at single-digit milliseconds, while the architecture mirrors what a dense-retrieval system looks like, so upgrading to embeddings later is a clean swap (see Production upgrade path).

## Architecture

```
                        +------------------+
                        |  txt / md / pdf  |
                        +--------+---------+
                                 |
                        +--------v---------+
                        |  ingest/loader   |  title + section metadata
                        +--------+---------+
                                 |
                        +--------v---------+
                        |  ingest/chunker  |  semantic-aware chunking,
                        |                  |  sentence overlap, section tracking
                        +--------+---------+
                                 |
              +------------------+------------------+
              |                                     |
     +--------v---------+                +----------v----------+
     |  hashed TF-IDF   |                |       BM25          |
     |  (numpy, 4k hash |                |   (rank-bm25)       |
     |   buckets)       |                |                     |
     +--------+---------+                +----------+----------+
              |                                     |
              +-----------+----------------+--------+
                          |
                 +--------v---------+
                 |  RRF fusion      |  1 / (k + rank), k=60
                 +--------+---------+
                          |
                 +--------v---------+
                 |  heuristic       |  tfidf cosine + term overlap
                 |  reranker        |  + match position
                 +--------+---------+
                          |
                 +--------v---------+
                 |  guardrails      |  injection screen, then
                 |                  |  PII redaction on context
                 +--------+---------+
                          |
                 +--------v---------+
                 |  generation      |  extractive Q&A with [n]
                 |                  |  citations (default), or
                 |                  |  optional LLM abstractive
                 +------------------+
```

Request flow for `POST /ask`:

1. The question is screened for prompt-injection patterns; matches are blocked with reasons.
2. Hybrid retrieval runs over metadata-filtered chunks (BM25 and TF-IDF top-25 each, RRF-fused, reranked to top-k).
3. Retrieved context is PII-redacted before generation, and the final answer is redacted again.
4. Extractive generation selects and deduplicates the best sentences and attaches numbered citations that resolve to document, section, and source file.

## 60-second quickstart

```bash
git clone <this-repo> && cd enterprise-rag-pipeline
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Ask a question over the bundled sample policies
python cli.py ask "What is the meal per diem for business travel?"

# Run the evaluation suite
python cli.py eval

# Start the API
uvicorn api.main:app --reload
# POST /ask, POST /ingest, GET /stats, GET /health  (docs at /docs)
```

Example:

```
$ python cli.py ask "What is the acknowledgement time for a SEV1 incident?"

A (extractive, 3ms): A SEV1 incident requires acknowledgement within 15 minutes
and an incident commander assigned within 30 minutes. [1]

Sources:
  [1] Incident Response Policy - Response Times [incident-response-policy.md]
```

Index your own documents:

```bash
python cli.py ingest ./my-docs        # txt, md, pdf
```

Or via the API:

```bash
curl -X POST localhost:8000/ingest \
  -H 'Content-Type: application/json' \
  -d '{"directory": "./my-docs"}'

curl -X POST localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question": "How long are logs retained?",
       "filters": {"source": "data-retention-policy.md"}}'
```

## Evaluation

`python cli.py eval` (or `python -m eval.run_eval`) builds the pipeline over the bundled policies and scores 10 sample questions from `data_samples/eval_qa.json`:

| metric | what it measures | mean |
|---|---|---|
| faithfulness | share of answer sentences supported by retrieved context | 1.000 |
| answer relevance | cosine(question, answer) in TF-IDF space | 0.465 |
| context precision | share of retrieved chunks relevant to the question | 0.567 |
| latency | end-to-end ask time | 2.3 ms |

Read these for what they are: honest proxies, not LLM-judge scores. Faithfulness is 1.000 because extractive answers quote retrieved sentences by construction. Relevance and precision reflect lexical retrieval on a tiny 5-document corpus; they move up with real embeddings (see below). Every metric runs offline in milliseconds, and the harness prints per-question scores so regressions are visible.

The test suite (`pytest tests/`, 35 tests) covers chunking, RRF fusion, reranking, metadata filters, PII redaction, injection detection, and the metrics themselves.

## Configuration

| variable | purpose | default |
|---|---|---|
| `RAG_LLM_BASE_URL` | OpenAI-compatible endpoint, e.g. `https://api.openai.com/v1` | unset (extractive mode) |
| `RAG_LLM_API_KEY` | bearer token for the endpoint | unset |
| `RAG_LLM_MODEL` | model name, e.g. `gpt-4o-mini` | unset |

Abstractive mode is only attempted when all three are set (`python cli.py ask "..." --mode abstractive`); any failure degrades gracefully back to extractive. Retrieval tuning lives in `retrieval/pipeline.py` (`candidate_k`, `rrf_k`, `rerank_pool`) and `retrieval/reranker.py` (signal weights).

## Project structure

```
enterprise-rag-pipeline/
  ingest/        loader.py (txt/md/pdf), chunker.py (semantic-aware,
                 overlapping, section-tracked chunks), models.py
  retrieval/     text.py (tokenize + stem), tfidf.py (hashed TF-IDF/numpy),
                 bm25_index.py, fusion.py (RRF), reranker.py,
                 filters.py, pipeline.py (HybridRetriever)
  generation/    extractive.py (cited answers), llm.py (optional OpenAI-
                 compatible abstractive), answer.py (RAGPipeline orchestration)
  guardrails/    pii.py (regex redaction + Luhn-validated card detection),
                 injection.py (two-tier prompt-injection heuristics)
  eval/          metrics.py (faithfulness, relevance, context precision),
                 run_eval.py (results table)
  api/           main.py (FastAPI: /ask, /ingest, /stats, /health)
  cli.py         ask / ingest / eval / stats commands
  data_samples/  5 fictional company policies + eval_qa.json (10 questions)
  tests/         35 pytest tests, all offline
```

## Production upgrade path

The interfaces are shaped so each lexical component swaps for a neural one without touching the pipeline:

- `retrieval/tfidf.py` (`HashedTfidfVectorizer.fit/transform`) can be replaced by sentence-transformers bi-encoders; cosine similarity and L2 normalization already match.
- `retrieval/reranker.py` (`HeuristicReranker.rerank`) takes the same `(query, candidates)` in and scored chunks out as a cross-encoder would.
- `retrieval/pipeline.py` (`HybridRetriever`) would then fuse dense + BM25 exactly as it does today.
- For scale, the in-memory numpy index moves to FAISS or pgvector behind the same `search()` signature; chunking, guardrails, eval, and the API stay untouched.

## Limitations

- Lexical retrieval has no true semantic understanding: paraphrases with no term overlap can miss ("password requirements" vs "passwords must be..."). The bundled stemmer and hybrid fusion mitigate this, but embeddings remain the real fix.
- The injection detector is heuristic pattern matching, a first line of defense, not a substitute for a trained classifier.
- PII patterns target common US formats; they are not exhaustive.
- The eval metrics are proxies. Faithfulness via word overlap is generous to extractive answers and would need an NLI or LLM judge to be meaningful for abstractive ones.
- The sample corpus is five short fictional policies; treat the reported numbers as a smoke test, not a benchmark.
