"""Bounded retry helper for LLM calls."""
from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

from .provider import LLMError

T = TypeVar("T")

logger = logging.getLogger(__name__)

#: Seconds to wait before each retry, indexed by attempt number (0-based). Fixed, not
#: exponential and not jittered, because the point is reproducibility: an experiment has to be
#: able to state its retry policy exactly, and a randomised schedule makes two runs of the same
#: batch differ in how hard they tried.
#:
#: Why a delay at all: measured against the thesis endpoint, retrying with NO delay barely
#: retried. The server drops connections in bursts (``RemoteProtocolError`` at ~14% of
#: attempts), and three attempts fired within milliseconds all land inside the same burst --
#: which is why the observed failure rate was 15.4% of logical calls where independent
#: failures would have predicted 0.3%. Spacing the attempts is what makes a retry a second
#: chance rather than a second symptom.
DEFAULT_BACKOFF_SECONDS: tuple[float, ...] = (2.0, 5.0, 10.0, 20.0)


def backoff_for(attempt: int, schedule: tuple[float, ...] = DEFAULT_BACKOFF_SECONDS) -> float:
    """Seconds to wait before retry number ``attempt`` (0-based).

    Past the end of the schedule the last value repeats, so a longer ``max_retries`` cannot
    silently turn into an unbounded wait.
    """
    if not schedule:
        return 0.0
    return schedule[min(attempt, len(schedule) - 1)]


def retry_call(
    fn: Callable[[], T],
    max_retries: int,
    *,
    backoff: tuple[float, ...] = DEFAULT_BACKOFF_SECONDS,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call ``fn`` up to ``max_retries + 1`` times with fixed backoff, re-raising the last error.

    Bounded by construction: the loop is a ``range``, so no failure mode can extend it. Only
    :class:`LLMError` is retried -- an unexpected exception is a bug rather than a transient
    condition and must not be smoothed over by trying again.

    ``sleep`` is injected so tests can assert the schedule without waiting for it.
    """
    last: Exception | None = None
    for attempt in range(max_retries + 1):
        if attempt:
            delay = backoff_for(attempt - 1, backoff)
            if delay:
                logger.warning("llm call failed; retry %s of %s after %.1fs",
                               attempt, max_retries, delay)
                sleep(delay)
        try:
            return fn()
        except LLMError as exc:  # only retry known LLM failures
            last = exc
    assert last is not None
    raise last
