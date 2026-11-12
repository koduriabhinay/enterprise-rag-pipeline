"""Heuristic prompt-injection detection for incoming questions.

Two tiers:
  * high-severity phrases block immediately (instruction override attempts,
    system-prompt extraction, jailbreak markers)
  * medium-severity phrases accumulate a score; the query is blocked when the
    score reaches ``block_threshold`` (default 2)

This is a first line of defense, not a complete solution: a production
system would layer a classifier on top. Every decision ships with reasons so
blocks are auditable.
"""

import re
from dataclasses import dataclass, field
from typing import List

_HIGH_SEVERITY = [
    r"ignore\s+(all\s+)?(previous|prior|earlier)\s+instructions?",
    r"disregard\s+(all\s+)?(previous|prior|your)\s+instructions?",
    r"forget\s+(all\s+)?(previous|prior|your)\s+instructions?",
    r"reveal\s+(your\s+)?(system|hidden|secret)\s+(prompt|instructions?)",
    r"show\s+me\s+(your\s+)?(system|hidden|secret)\s+(prompt|instructions?)",
    r"you\s+are\s+now\s+",
    r"\bdo\s+anything\s+now\b",
    r"\bjailbreak\b",
    r"\bdeveloper\s+mode\b",
    r"override\s+(your\s+)?(safety|system|core)\s+(instructions?|rules?)",
    r"bypass\s+(your\s+)?(safety|content|system)\s+(filters?|rules?|guardrails?)",
]

_MEDIUM_SEVERITY = [
    r"\bpretend\s+(you\s+are|to\s+be)\b",
    r"\bact\s+as\s+if\b",
    r"\bno\s+restrictions\b",
    r"\bwithout\s+restrictions\b",
    r"\bhypothetical(ly)?\b.*\b(ignore|disregard)\b",
    r"\bsecret(ly)?\b.*\binstructions?\b",
    r"\bnew\s+instructions?\b.*\bfollow\b",
]

_HIGH_RE = [re.compile(p, re.IGNORECASE) for p in _HIGH_SEVERITY]
_MEDIUM_RE = [re.compile(p, re.IGNORECASE) for p in _MEDIUM_SEVERITY]


@dataclass
class InjectionVerdict:
    blocked: bool
    reasons: List[str] = field(default_factory=list)
    score: int = 0


def check_injection(text: str, block_threshold: int = 2) -> InjectionVerdict:
    """Inspect a user query for prompt-injection attempts."""
    reasons: List[str] = []
    score = 0

    for pattern in _HIGH_RE:
        m = pattern.search(text)
        if m:
            reasons.append(f"high-severity instruction-override pattern: {m.group(0)!r}")
            return InjectionVerdict(blocked=True, reasons=reasons, score=score + 10)

    for pattern in _MEDIUM_RE:
        for m in pattern.finditer(text):
            score += 1
            reasons.append(f"suspicious phrasing: {m.group(0)!r}")

    blocked = score >= block_threshold
    return InjectionVerdict(blocked=blocked, reasons=reasons, score=score)
