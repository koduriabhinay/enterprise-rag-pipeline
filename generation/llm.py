"""Optional abstractive generation via any OpenAI-compatible chat endpoint.

Configuration is entirely environment-driven; when the variables are absent
the pipeline silently stays in extractive mode:

    RAG_LLM_BASE_URL  e.g. https://api.openai.com/v1 (no trailing slash)
    RAG_LLM_API_KEY   bearer token
    RAG_LLM_MODEL     e.g. gpt-4o-mini

The model is instructed to answer only from the provided context and to
cite sources with the [n] markers produced by the extractive stage.
"""

import os
from dataclasses import dataclass
from typing import List

from generation.extractive import Citation


@dataclass
class LlmConfig:
    base_url: str
    api_key: str
    model: str

    @classmethod
    def from_env(cls) -> "LlmConfig | None":
        base_url = os.environ.get("RAG_LLM_BASE_URL", "").rstrip("/")
        api_key = os.environ.get("RAG_LLM_API_KEY", "")
        model = os.environ.get("RAG_LLM_MODEL", "")
        if not (base_url and api_key and model):
            return None
        return cls(base_url=base_url, api_key=api_key, model=model)


_SYSTEM_PROMPT = (
    "You are an enterprise Q&A assistant. Answer the user's question using "
    "ONLY the context below. If the context does not contain the answer, say "
    "so plainly. Cite every factual claim with the provided [n] markers, "
    "matching the sources they came from. Be concise."
)


def generate_abstractive(
    question: str,
    context: str,
    citations: List[Citation],
    config: LlmConfig,
    timeout_s: float = 30.0,
) -> str:
    """Call the chat completions endpoint; raises on transport/API errors."""
    import httpx  # local import: only needed when LLM mode is active

    source_lines = "\n".join(
        f"{c.label} {c.title} - {c.section} (source file: {c.source})"
        for c in citations
    )
    user_prompt = (
        f"Context:\n{context}\n\n"
        f"Sources:\n{source_lines}\n\n"
        f"Question: {question}\n\nAnswer:"
    )
    resp = httpx.post(
        f"{config.base_url}/chat/completions",
        headers={"Authorization": f"Bearer {config.api_key}"},
        json={
            "model": config.model,
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0.1,
        },
        timeout=timeout_s,
    )
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"].strip()
