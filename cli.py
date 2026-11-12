"""Terminal interface for the RAG pipeline.

Usage:
    python cli.py ask "What is the meal per diem?"
    python cli.py ask "Which documents mention SEV1?" --filter source=incident-response-policy.md
    python cli.py ingest ./my-docs
    python cli.py eval
    python cli.py stats
"""

import argparse
import json
from pathlib import Path

from generation.answer import RAGPipeline

PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DOCS = PROJECT_ROOT / "data_samples" / "policies"


def _parse_filters(pairs) -> dict:
    filters = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"filter must look like key=value, got {pair!r}")
        key, value = pair.split("=", 1)
        # Support comma-separated lists: section=a,b
        filters[key] = value.split(",") if "," in value else value
    return filters


def _print_answer(ans) -> None:
    print(f"\nQ: {ans.question}\n")
    if ans.blocked:
        print("BLOCKED by input guardrail:")
        for reason in ans.block_reasons:
            print(f"  - {reason}")
        return
    print(f"A ({ans.mode}, {ans.latency_ms:.0f}ms): {ans.answer_text}\n")
    if ans.citations:
        print("Sources:")
        for c in ans.citations:
            print(f"  {c.label} {c.title} - {c.section} [{c.source}]")
    if ans.pii_findings:
        print("\nPII redacted:", json.dumps(ans.pii_findings))


def cmd_ask(args) -> int:
    pipeline = RAGPipeline(DEFAULT_DOCS if DEFAULT_DOCS.is_dir() else None)
    ans = pipeline.ask(
        args.question,
        top_k=args.top_k,
        filters=_parse_filters(args.filter),
        mode=args.mode,
    )
    _print_answer(ans)
    return 0


def cmd_ingest(args) -> int:
    pipeline = RAGPipeline(DEFAULT_DOCS if DEFAULT_DOCS.is_dir() else None)
    added = pipeline.ingest_directory(args.directory)
    print(f"ingested {added} chunks from {args.directory}")
    print(json.dumps(pipeline.stats, indent=2))
    return 0


def cmd_eval(_args) -> int:
    from eval.run_eval import main as eval_main

    return eval_main()


def cmd_stats(_args) -> int:
    pipeline = RAGPipeline(DEFAULT_DOCS if DEFAULT_DOCS.is_dir() else None)
    print(json.dumps(pipeline.stats, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Enterprise RAG pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ask = sub.add_parser("ask", help="Ask a question over the indexed documents")
    p_ask.add_argument("question", help="The question to answer")
    p_ask.add_argument("--top-k", type=int, default=5, help="Chunks to retrieve")
    p_ask.add_argument("--filter", action="append", help="Metadata filter key=value")
    p_ask.add_argument("--mode", default="extractive", choices=["extractive", "abstractive"])
    p_ask.set_defaults(func=cmd_ask)

    p_ing = sub.add_parser("ingest", help="Index a directory of documents")
    p_ing.add_argument("directory", help="Directory containing txt/md/pdf files")
    p_ing.set_defaults(func=cmd_ingest)

    p_eval = sub.add_parser("eval", help="Run the bundled evaluation question set")
    p_eval.set_defaults(func=cmd_eval)

    p_stats = sub.add_parser("stats", help="Show index statistics")
    p_stats.set_defaults(func=cmd_stats)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
