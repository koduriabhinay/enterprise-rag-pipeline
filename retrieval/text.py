"""Shared text utilities: tokenization and stopwords."""

import re
from typing import List

# A compact English stopword list so common words do not dominate similarity.
STOPWORDS = frozenset(
    """
    a an the and or but if then else when while of at by for with about into
    through during before after above below to from up down in out on off over
    under again further once here there all any both each few more most other
    some such no nor not only own same so than too very can will just don
    should now is are was were be been being have has had having do does did
    doing would could ought i you he she it we they them his her its our their
    this that these those as s t ll m re ve d
    """.split()
)

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")


def _stem(token: str) -> str:
    """Tiny rule-based stemmer for plural/verb forms.

    Only the most predictable suffixes are stripped, and the same function
    runs on queries and documents, so matching stays consistent. This is a
    heuristic, not a linguistic stemmer: "business" becomes "busines" on both
    sides, which is all retrieval needs.
    """
    if token.endswith("ies") and len(token) > 4:
        return token[:-3] + "y"
    if token.endswith("s") and not token.endswith("ss") and len(token) > 3:
        return token[:-1]
    return token


def tokenize(text: str, keep_stopwords: bool = False) -> List[str]:
    """Lowercase alphanumeric tokenization, stopword removal, and stemming."""
    tokens = _TOKEN_RE.findall(text.lower())
    tokens = [t for t in tokens if len(t) >= 2]
    if not keep_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]
    return [_stem(t) for t in tokens]


def content_words(text: str) -> List[str]:
    """Tokens with stopwords removed, used for overlap-style heuristics."""
    return tokenize(text, keep_stopwords=False)
