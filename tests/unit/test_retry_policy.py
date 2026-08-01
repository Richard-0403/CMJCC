"""The retry policy is bounded, spaced, declared, and part of the experiment's identity.

Why spacing matters, measured rather than assumed
------------------------------------------------
``retry_call`` used to retry in a tight loop with no delay. Against the thesis endpoint that
barely retried at all: the server drops connections in bursts (``RemoteProtocolError`` on ~14%
of attempts, confirmed by ``scripts/probe_endpoint_prompt_size.py`` at every prompt size from
1.7k to 33k characters), so three attempts fired within milliseconds all landed inside the
same burst. The observed failure rate was 15.4% of logical calls where independent failures
would have predicted 0.3% -- the gap between those two numbers is what a missing backoff looks
like.

Three things ruled out before landing on this, each with data:

* prompt size -- failures occur at every size and the LARGEST size passed 3/3;
* native structured output -- the endpoint accepts ``response_format`` on every call, 0 of 32
  records had to drop it;
* the client timeout -- raising it from 60s to 240s did not help, it only changed the failure
  from ``ReadTimeout`` into ``RemoteProtocolError``.

What is asserted here
---------------------
The schedule is fixed and unjittered, so two runs of one batch try equally hard; it is bounded,
so no failure mode can extend it; and it lives in the config, so it enters ``config_hash`` and
therefore the experiment id -- two batches that tried different numbers of times are not the
same experiment.
"""

from __future__ import annotations

import pytest

from jobrec.config import AppConfig
from jobrec.llm.provider import LLMError
from jobrec.llm.retry import DEFAULT_BACKOFF_SECONDS, backoff_for, retry_call


def test_a_successful_call_never_waits():
    waits: list[float] = []
    assert retry_call(lambda: "ok", 4, sleep=waits.append) == "ok"
    assert waits == []


def test_each_retry_waits_the_declared_amount():
    waits: list[float] = []
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise LLMError("dropped")
        return "ok"

    assert retry_call(flaky, 4, backoff=(2.0, 5.0, 10.0), sleep=waits.append) == "ok"
    assert calls["n"] == 3
    # Two failures -> two waits, taken in schedule order.
    assert waits == [2.0, 5.0]


def test_the_schedule_is_bounded_and_its_last_value_repeats():
    """A longer ``max_retries`` must not silently become an unbounded wait."""
    schedule = (1.0, 2.0)
    assert [backoff_for(i, schedule) for i in range(5)] == [1.0, 2.0, 2.0, 2.0, 2.0]
    assert backoff_for(0, ()) == 0.0


def test_retries_are_bounded_by_max_retries():
    waits: list[float] = []
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise LLMError("dropped")

    with pytest.raises(LLMError):
        retry_call(always_fails, 4, backoff=(1.0,), sleep=waits.append)

    assert calls["n"] == 5, "max_retries + 1 attempts, no more and no fewer"
    assert len(waits) == 4


def test_an_unexpected_exception_is_not_retried():
    """A bug is not a transient condition and must not be smoothed over by trying again."""
    calls = {"n": 0}

    def broken():
        calls["n"] += 1
        raise ValueError("a real bug")

    with pytest.raises(ValueError):
        retry_call(broken, 4, sleep=lambda _s: None)
    assert calls["n"] == 1


def test_the_schedule_is_fixed_and_not_random():
    """Two runs of one batch must try equally hard, so the schedule cannot be jittered."""
    assert [backoff_for(i) for i in range(4)] == list(DEFAULT_BACKOFF_SECONDS)
    assert [backoff_for(i) for i in range(4)] == list(DEFAULT_BACKOFF_SECONDS)
    assert all(isinstance(x, float) and x > 0 for x in DEFAULT_BACKOFF_SECONDS)
    assert list(DEFAULT_BACKOFF_SECONDS) == sorted(DEFAULT_BACKOFF_SECONDS), (
        "a schedule that shrinks would retry hardest when the endpoint is worst")


# ------------------------------------------------- the policy is part of the identity
def test_the_policy_lives_in_the_config_and_moves_the_config_hash():
    """Two batches that tried different numbers of times are not the same experiment."""
    base = AppConfig()
    assert base.llm.retry_backoff_seconds == list(DEFAULT_BACKOFF_SECONDS)

    more_retries = AppConfig()
    more_retries.llm.max_retries = base.llm.max_retries + 2
    slower = AppConfig()
    slower.llm.retry_backoff_seconds = [1.0, 1.0]

    assert base.config_hash() != more_retries.config_hash()
    assert base.config_hash() != slower.config_hash()


def test_the_resolved_config_records_the_policy():
    """It has to be readable from the artifact, not inferred from the code version."""
    dumped = AppConfig().model_dump(mode="json")["llm"]
    assert dumped["max_retries"] == 2
    assert dumped["retry_backoff_seconds"] == list(DEFAULT_BACKOFF_SECONDS)
    assert "timeout_seconds" in dumped
