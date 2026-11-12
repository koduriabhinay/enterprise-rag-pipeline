"""Command-line evaluation harness: builds the pipeline, runs the sample
question set, and prints a results table with per-question and mean scores."""

import json
from pathlib import Path

from eval.metrics import run_evaluation
from generation.answer import RAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_dataset(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _print_table(report, dataset):
    q_width = max(len(i["question"]) for i in dataset)
    q_width = min(max(q_width, 8), 60)
    header = (
        f"{'question':<{q_width}}  {'faithful':>8}  {'relevant':>8}  "
        f"{'ctx_prec':>8}  {'latency':>9}"
    )
    print(header)
    print("-" * len(header))
    for item, res in zip(dataset, report.items):
        q = item["question"]
        q = q if len(q) <= q_width else q[: q_width - 3] + "..."
        flag = "  [BLOCKED]" if res.blocked else ""
        print(
            f"{q:<{q_width}}  {res.faithfulness:>8.3f}  {res.answer_relevance:>8.3f}  "
            f"{res.context_precision:>8.3f}  {res.latency_ms:>7.1f}ms{flag}"
        )
    print("-" * len(header))
    m = report.means()
    print(
        f"{'MEAN':<{q_width}}  {m['faithfulness']:>8.3f}  {m['answer_relevance']:>8.3f}  "
        f"{m['context_precision']:>8.3f}  {m['latency_ms']:>7.1f}ms"
    )


def main() -> int:
    docs_dir = PROJECT_ROOT / "data_samples" / "policies"
    dataset_path = PROJECT_ROOT / "data_samples" / "eval_qa.json"
    if not docs_dir.is_dir():
        print(f"sample documents not found: {docs_dir}")
        return 1
    pipeline = RAGPipeline(docs_dir)
    dataset = _load_dataset(dataset_path)
    print(f"indexed {pipeline.stats['chunks']} chunks from {pipeline.stats['documents']} documents")
    print(f"running {len(dataset)} eval questions...\n")
    report = run_evaluation(pipeline, dataset)
    _print_table(report, dataset)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
