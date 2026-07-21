"""Bounded retry helper for LLM calls."""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar

from .provider import LLMError

T = TypeVar("T")


def retry_call(fn: Callable[[], T], max_retries: int) -> T:
    """Call ``fn`` up to ``max_retries + 1`` times, re-raising the last error."""
    last: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            return fn()
        except LLMError as exc:  # only retry known LLM failures
            last = exc
    assert last is not None
    raise last
