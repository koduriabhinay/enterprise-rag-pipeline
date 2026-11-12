"""Regex-based PII redaction for retrieved context and generated answers.

Detected types: email addresses, phone numbers, SSNs, and credit card
numbers (validated with a Luhn check to avoid flagging random digit runs).
Findings report only types and counts, never the matched values themselves.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

_PATTERNS: Dict[str, re.Pattern] = {
    # Order matters: credit cards are checked before phones so a 16-digit
    # card number is never partially eaten by the phone pattern.
    "email": re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"),
    "credit_card": re.compile(r"\b(?:\d[ -]?){13,16}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    # Phone numbers require separators so bare digit runs are not matched.
    "phone": re.compile(
        r"\b(?:\+?1[-.\s]?)?(?:\(\d{3}\)|\d{3})[-.\s]\d{3}[-.\s]\d{4}\b"
        r"|\b\d{3}[-.]\d{4}\b"
    ),
}

_REDACTION_LABELS = {
    "email": "[EMAIL REDACTED]",
    "phone": "[PHONE REDACTED]",
    "ssn": "[SSN REDACTED]",
    "credit_card": "[CARD REDACTED]",
}


def _luhn_ok(digits: str) -> bool:
    nums = [int(d) for d in digits if d.isdigit()]
    if not (13 <= len(nums) <= 16):
        return False
    checksum = 0
    for i, d in enumerate(reversed(nums)):
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        checksum += d
    return checksum % 10 == 0


@dataclass
class RedactionReport:
    redacted_text: str
    findings: List[Dict[str, int]] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(f["count"] for f in self.findings)


def redact_pii(text: str) -> RedactionReport:
    """Redact PII in text. Returns redacted text plus type/count findings."""
    findings: List[Dict[str, int]] = []
    redacted = text
    for kind, pattern in _PATTERNS.items():
        matches = list(pattern.finditer(redacted))
        count = 0
        # Iterate in reverse so earlier replacements do not shift offsets.
        for m in reversed(matches):
            raw = m.group(0)
            if kind == "credit_card" and not _luhn_ok(raw):
                continue
            if kind == "phone" and not any(ch.isdigit() for ch in raw):
                continue
            redacted = redacted[: m.start()] + _REDACTION_LABELS[kind] + redacted[m.end():]
            count += 1
        if count:
            findings.append({"type": kind, "count": count})
    return RedactionReport(redacted_text=redacted, findings=findings)


def contains_pii(text: str) -> bool:
    """Quick boolean check without building the redacted string."""
    return redact_pii(text).total > 0
