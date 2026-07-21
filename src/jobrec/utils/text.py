"""Small text normalisation helpers used by extraction and matching."""

from __future__ import annotations

import re

_WS = re.compile(r"\s+")
_NON_ALNUM = re.compile(r"[^a-z0-9+#. ]")


def normalize_token(value: str) -> str:
    """Lower-case and strip a token while preserving display-relevant chars."""
    value = value.strip().lower()
    value = _NON_ALNUM.sub(" ", value)
    return _WS.sub(" ", value).strip()


def normalize_skill(value: str) -> str:
    """Normalise a skill string to its canonical comparison form."""
    return normalize_token(value)


def tokenize(text: str) -> list[str]:
    """Very small whitespace tokenizer over normalised text."""
    return [t for t in normalize_token(text).split(" ") if t]
