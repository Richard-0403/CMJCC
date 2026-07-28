"""Unit tests for model-call accounting in hybrid runs (R11.1, R26.1).

Checklist item 9 asks a hybrid run to record what a real remote call cost and how
it was issued, and asks the recorded outputs to be replayable. These tests cover
the three pieces that make that true, against the real provider, the real bundle
writer and the real replay provider (no mocks beyond a fake transport):

* :class:`~jobrec.llm.remote_provider.RemoteLLMProvider` populates
  ``LLMCallRecord.metadata`` with token usage (normalised across both the classic
  ``prompt_tokens`` and the newer ``input_tokens`` spellings), the request
  parameters *as actually sent* and the retry/fallback trace;
* :func:`~jobrec.evaluation.exporters._model_call_row` surfaces those as
  ``request_params``/``response_metadata`` and carries a redacted ``raw_response``
  only while ``config.llm.save_raw_responses`` is on;
* :class:`~jobrec.llm.replay.ReplayProvider` serves a row that has ``call_id`` +
  ``raw_response`` and no prompt, which is the shape a run bundle writes.

No network call is ever made: ``_post`` is replaced with a function returning a
fabricated OpenAI-compatible payload. Every key used below is an obvious fake.
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from jobrec.app_service import AppService
from jobrec.config import AppConfig, load_config
from jobrec.evaluation.exporters import _model_call_row, write_run_bundle
from jobrec.llm.provider import LLMCallRecord, LLMError
from jobrec.llm.remote_provider import API_KEY_ENV, RemoteLLMProvider
from jobrec.llm.replay import ReplayProvider
from jobrec.utils.hashing import content_id
from jobrec.utils.redaction import REDACTED, REDACTED_KEY
from tests.conftest import CATALOG_PATH

FAKE_KEY = "sk-zzfakekeyvalue0000000001"
PROMPT = "Utterance: I want a data analyst role in Kuala Lumpur."
JSON_BODY = '{"role": "data analyst", "location": "Kuala Lumpur"}'

#: The classic OpenAI spelling and the newer one; both must normalise identically.
CLASSIC_USAGE = {"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150}
NEW_USAGE = {"input_tokens": 120, "output_tokens": 30}


# --------------------------------------------------------------------- helpers
def _response(content: str = JSON_BODY, *, usage: dict | None = None,
              finish_reason: str | None = "stop") -> dict:
    """A fabricated OpenAI-compatible chat-completions payload."""
    choice: dict = {"message": {"role": "assistant", "content": content}}
    if finish_reason is not None:
        choice["finish_reason"] = finish_reason
    payload: dict = {"id": "chatcmpl-fake", "choices": [choice]}
    if usage is not None:
        payload["usage"] = usage
    return payload


def _provider(monkeypatch, model: str = "gpt-4o-mini") -> RemoteLLMProvider:
    monkeypatch.setenv(API_KEY_ENV, FAKE_KEY)
    return RemoteLLMProvider(model=model, base_url="https://example.invalid/v1")


def _status_error() -> httpx.HTTPStatusError:
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")
    response = httpx.Response(400, request=request, json={"error": "unsupported"})
    return httpx.HTTPStatusError("400 Bad Request", request=request, response=response)


def _row(record: LLMCallRecord, config: AppConfig | None = None) -> dict:
    """The exported ``model_calls.jsonl`` row for one call record."""
    return _model_call_row(record, config)


# ------------------------------------------------------------------ token usage
@pytest.mark.parametrize(
    ("usage", "case"),
    [(CLASSIC_USAGE, "prompt/completion"), (NEW_USAGE, "input/output")],
    ids=["classic_spelling", "new_spelling"],
)
def test_token_usage_is_normalised_from_both_spellings(monkeypatch, usage, case) -> None:
    """Either upstream spelling lands on the same flat token keys (R11.1)."""
    provider = _provider(monkeypatch)
    monkeypatch.setattr(provider, "_post", lambda _body: _response(usage=usage))

    _payload, record = provider.complete_json(PROMPT, purpose="extraction")

    assert record.metadata["prompt_tokens"] == 120, case
    assert record.metadata["completion_tokens"] == 30, case
    assert record.metadata["total_tokens"] == 150, case
    # The server's own object is kept verbatim alongside the normalised counts.
    assert record.metadata["usage"] == usage

    response_metadata = _row(record)["response_metadata"]
    assert response_metadata["prompt_tokens"] == 120
    assert response_metadata["completion_tokens"] == 30
    assert response_metadata["total_tokens"] == 150
    assert response_metadata["finish_reason"] == "stop"


def test_a_response_without_usage_does_not_crash_and_reports_no_tokens(
    monkeypatch,
) -> None:
    """Some proxies omit ``usage``; absent must stay absent, never a fake zero."""
    provider = _provider(monkeypatch)
    monkeypatch.setattr(provider, "_post", lambda _body: _response(usage=None))

    _payload, record = provider.complete_json(PROMPT, purpose="extraction")

    assert "usage" not in record.metadata
    assert "total_tokens" not in record.metadata
    response_metadata = _row(record)["response_metadata"]
    assert "total_tokens" not in response_metadata
    # The rest of the accounting is still recorded.
    assert response_metadata["attempts"] == 1


def test_a_partial_usage_block_is_recorded_without_inventing_the_missing_half(
    monkeypatch,
) -> None:
    """A server reporting only one half gets that half recorded, and no total."""
    provider = _provider(monkeypatch)
    monkeypatch.setattr(
        provider, "_post", lambda _body: _response(usage={"prompt_tokens": 7}))

    _payload, record = provider.complete_json(PROMPT, purpose="extraction")

    assert record.metadata["prompt_tokens"] == 7
    assert "completion_tokens" not in record.metadata
    assert "total_tokens" not in record.metadata


# ----------------------------------------------------------- request parameters
def test_temperature_is_absent_for_gpt_5_because_it_was_never_sent(monkeypatch) -> None:
    """The code omits temperature for the gpt-5 family; the artifact must agree."""
    provider = _provider(monkeypatch, model="gpt-5-mini")
    sent: list[dict] = []

    def post(body: dict) -> dict:
        sent.append(dict(body))
        return _response(usage=CLASSIC_USAGE)

    monkeypatch.setattr(provider, "_post", post)

    _payload, record = provider.complete_json(PROMPT, purpose="extraction")

    assert "temperature" not in sent[0], "fixture guard: nothing was sent"
    assert "temperature" not in record.metadata
    assert "temperature" not in _row(record)["request_params"]


def test_temperature_is_recorded_for_a_model_that_accepts_it(monkeypatch) -> None:
    """For every other model the value actually sent is the value recorded."""
    provider = _provider(monkeypatch)
    provider.extraction_temperature = 0.0
    sent: list[dict] = []

    def post(body: dict) -> dict:
        sent.append(dict(body))
        return _response(usage=CLASSIC_USAGE)

    monkeypatch.setattr(provider, "_post", post)

    _payload, record = provider.complete_json(PROMPT, purpose="extraction")

    assert sent[0]["temperature"] == 0.0
    request_params = _row(record)["request_params"]
    assert request_params["temperature"] == 0.0
    assert request_params["json_mode"] is True
    assert request_params["response_format"] == {"type": "json_object"}
    assert request_params["timeout_seconds"] == provider.timeout
    # The pre-existing descriptive fields are untouched.
    assert request_params["model"] == "gpt-4o-mini"
    assert request_params["purpose"] == "extraction"


def test_text_calls_record_that_json_mode_was_not_requested(monkeypatch) -> None:
    """Phrasing calls do not ask for JSON mode, and the row says so."""
    provider = _provider(monkeypatch)
    monkeypatch.setattr(
        provider, "_post", lambda _body: _response("Here are three roles.",
                                                   usage=CLASSIC_USAGE))

    _text, record = provider.complete_text(PROMPT, purpose="response", fallback="fb")

    request_params = _row(record)["request_params"]
    assert request_params["json_mode"] is False
    assert "response_format" not in request_params
    assert request_params["temperature"] == provider.response_temperature


# --------------------------------------------------------------- retry/fallback
def test_the_http_status_retry_is_visible_in_the_exported_row(monkeypatch) -> None:
    """The response_format-dropping retry used to leave no trace at all (R11.1)."""
    provider = _provider(monkeypatch)
    calls: list[dict] = []

    def post(body: dict) -> dict:
        calls.append(dict(body))
        if len(calls) == 1:
            raise _status_error()
        return _response(usage=CLASSIC_USAGE)

    monkeypatch.setattr(provider, "_post", post)

    _payload, record = provider.complete_json(PROMPT, purpose="extraction")

    assert len(calls) == 2, "fixture guard: the retry must have fired"
    row = _row(record)
    assert row["response_metadata"]["attempts"] == 2
    assert row["response_metadata"]["retried_without_response_format"] is True
    assert row["response_metadata"]["retry_reason"] == "http_400"
    # The retry dropped both parameters, so neither may be reported as sent.
    assert "response_format" not in row["request_params"]
    assert "temperature" not in row["request_params"]
    assert row["request_params"]["json_mode"] is False


def test_a_successful_first_attempt_records_one_attempt_and_no_retry(
    monkeypatch,
) -> None:
    provider = _provider(monkeypatch)
    monkeypatch.setattr(provider, "_post", lambda _body: _response(usage=CLASSIC_USAGE))

    _payload, record = provider.complete_json(PROMPT, purpose="extraction")

    response_metadata = _row(record)["response_metadata"]
    assert response_metadata["attempts"] == 1
    assert response_metadata["retried_without_response_format"] is False
    assert "retry_reason" not in response_metadata


def test_a_phrasing_fallback_records_why_it_fell_back(monkeypatch) -> None:
    """``complete_text`` never raises; the artifact must still show it degraded."""
    provider = _provider(monkeypatch)

    def boom(_body: dict) -> dict:
        raise httpx.ConnectError("connection refused")

    monkeypatch.setattr(provider, "_post", boom)

    text, record = provider.complete_text(PROMPT, purpose="response", fallback="rule text")

    assert text == "rule text"
    response_metadata = _row(record)["response_metadata"]
    assert response_metadata["fell_back"] is True
    # The exception CLASS is recorded; its message could quote the request.
    assert response_metadata["error"] == "LLMError"
    assert response_metadata["parsed_ok"] is False


# ------------------------------------------------------- raw response retention
def _remote_record(raw: str = JSON_BODY) -> LLMCallRecord:
    return LLMCallRecord(
        call_id=content_id("call", "extraction", PROMPT), purpose="extraction",
        prompt=PROMPT, raw_response=raw, parsed_ok=True, latency_ms=812.5,
        provider="remote", model="gpt-4o-mini",
        metadata={"prompt_tokens": 120, "completion_tokens": 30, "total_tokens": 150},
    )


def test_save_raw_responses_on_writes_a_redacted_raw_response() -> None:
    """The default keeps the recorded output, which is what makes replay possible."""
    config = AppConfig()
    assert config.llm.save_raw_responses is True, "fixture guard: default is on"

    row = _row(_remote_record('{"note": "call me on +60 12-345 6789"}'), config)

    assert "12-345 6789" not in row["raw_response"], "PII is stripped on the way out"
    assert "note" in row["raw_response"], "structure survives redaction"
    assert "prompt" not in row, "prompts are never persisted"


def test_save_raw_responses_off_omits_the_field_entirely() -> None:
    """Off means ABSENT, so "not retained" cannot be read as "empty response"."""
    config = AppConfig()
    config.llm.save_raw_responses = False

    row = _row(_remote_record(), config)

    assert "raw_response" not in row
    # The accounting fields are unaffected by the retention switch.
    assert row["response_metadata"]["total_tokens"] == 150


def test_redact_candidate_text_wipes_the_bundled_raw_response() -> None:
    """The bundle honours the same switch the database write path honours."""
    config = AppConfig()
    config.logging.redact_candidate_text = True

    row = _row(_remote_record(), config)

    assert row["raw_response"] == REDACTED
    assert row["purpose"] == "extraction", "structural fields stay inspectable"


def test_a_secret_value_never_appears_in_a_written_row(tmp_path: Path) -> None:
    """R26.1: a credential echoed back by a model must not reach the artifact."""
    record = _remote_record(f'{{"echo": "Authorization: Bearer {FAKE_KEY}"}}')

    row = _row(record, AppConfig())
    line = json.dumps(row)

    assert FAKE_KEY not in line
    assert REDACTED_KEY in row["raw_response"]
    (tmp_path / "model_calls.jsonl").write_text(line)
    assert FAKE_KEY not in (tmp_path / "model_calls.jsonl").read_text()


def test_a_row_with_no_metadata_keeps_its_previous_shape() -> None:
    """A provider that records nothing extra still exports the original fields."""
    record = LLMCallRecord(
        call_id="call-1", purpose="response", prompt=PROMPT, raw_response="text",
        parsed_ok=True, latency_ms=1.5, provider="mock", model="mock-deterministic-v1")

    row = _row(record, AppConfig())

    assert row["call_id"] == "call-1"
    assert row["latency_ms"] == 1.5
    assert row["request_params"] == {
        "purpose": "response", "provider": "mock", "model": "mock-deterministic-v1"}
    assert row["response_metadata"] == {"parsed_ok": True, "latency_ms": 1.5}


# ---------------------------------------------------------------------- replay
def _write_records(tmp_path: Path, rows: list[dict]) -> Path:
    path = tmp_path / "model_calls.jsonl"
    with path.open("w") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")
    return path


def test_replay_serves_a_bundle_row_that_has_no_prompt(tmp_path: Path) -> None:
    """``call_id`` IS the lookup id, so a prompt-free bundle row replays (R18)."""
    row = _row(_remote_record(), AppConfig())
    assert "prompt" not in row, "fixture guard: the bundle persists no prompt"

    provider = ReplayProvider(_write_records(tmp_path, [row]))
    payload, record = provider.complete_json(PROMPT, purpose="extraction")

    assert payload == json.loads(JSON_BODY)
    assert record.raw_response == JSON_BODY
    assert provider.manifest()["records"] == 1


def test_replay_still_serves_records_that_carry_a_prompt(tmp_path: Path) -> None:
    """Older/external files keyed only by (purpose, prompt) keep working."""
    legacy = {"purpose": "extraction", "prompt": PROMPT, "raw_response": JSON_BODY}

    provider = ReplayProvider(_write_records(tmp_path, [legacy]))
    payload, _record = provider.complete_json(PROMPT, purpose="extraction")

    assert payload == json.loads(JSON_BODY)


def test_replay_indexes_a_record_under_both_keys(tmp_path: Path) -> None:
    """A record carrying both a prompt and a mismatched id answers to either."""
    rec = {"call_id": "mock-style-id", "purpose": "extraction", "prompt": PROMPT,
           "raw_response": JSON_BODY}

    provider = ReplayProvider(_write_records(tmp_path, [rec]))

    assert provider.manifest() == {
        "provider": "replay", "model": "replay", "mode": "replay",
        "records": 1, "keys": 2}
    assert provider.complete_json(PROMPT, purpose="extraction")[0]


def test_an_empty_replay_file_is_legitimate_and_text_falls_back(tmp_path: Path) -> None:
    """Deterministic bundles record no calls at all; that must degrade gracefully."""
    provider = ReplayProvider(_write_records(tmp_path, []))

    text, record = provider.complete_text(PROMPT, purpose="response", fallback="rule text")

    assert text == "rule text"
    assert record.provider == "replay"
    assert provider.manifest()["records"] == 0
    with pytest.raises(LLMError):
        provider.complete_json(PROMPT, purpose="extraction")


def test_a_row_written_without_raw_responses_is_reported_as_missing(
    tmp_path: Path,
) -> None:
    """With retention off there is nothing to replay, and that is said explicitly."""
    config = AppConfig()
    config.llm.save_raw_responses = False

    provider = ReplayProvider(_write_records(tmp_path, [_row(_remote_record(), config)]))

    with pytest.raises(LLMError) as excinfo:
        provider.complete_json(PROMPT, purpose="extraction")

    assert "no response body" in str(excinfo.value)


# ------------------------------------------------------------ end-to-end bundle
def _bundle_rows(out_dir: Path) -> list[dict]:
    lines = (out_dir / "model_calls.jsonl").read_text().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


@pytest.mark.parametrize("save_raw", [True, False], ids=["retained", "not_retained"])
def test_the_bundle_writer_threads_the_retention_config_to_disk(
    tmp_path: Path, save_raw: bool
) -> None:
    """``write_run_bundle`` honours ``config.llm.save_raw_responses`` on the real path."""
    config = load_config("configs/experiment_full.yaml", base_dir="configs")
    config.llm.save_raw_responses = save_raw
    service = AppService(config, CATALOG_PATH)
    candidate = service.create_candidate(
        {"candidate_id": "cand-r11", "skills": ["Python", "SQL"], "years_experience": 3})
    session_id = service.create_session(candidate.candidate_id, "full")
    result = service.process_turn(session_id, PROMPT, scenario_id="SC-R11")
    # A deterministic turn issues no model call, so attach the hybrid-shaped record
    # the remote provider would have produced.
    result.model_calls.append(_remote_record())

    out = write_run_bundle(result, tmp_path / "run", config)

    rows = _bundle_rows(out)
    assert len(rows) == 1
    assert rows[0]["response_metadata"]["total_tokens"] == 150
    assert "prompt" not in rows[0]
    assert ("raw_response" in rows[0]) is save_raw


# ---------------------------------------------------------------------------
# Malformed-but-successful responses must degrade ONE call, never abort a batch.
#
# Everything below used to escape the provider as a non-``LLMError``: a
# ``json.JSONDecodeError`` from ``resp.json()`` and a ``KeyError``/``IndexError``/
# ``TypeError`` from indexing ``data["choices"][0]["message"]["content"]``, both outside
# any handler. ``retry_call`` only catches ``LLMError`` and the orchestrator's fallback to
# the rule extractor only catches ``LLMError``, so one such reply would propagate out of
# the whole experiment loop -- discarding every run already completed in a multi-hour
# hybrid batch. Gateways return exactly these shapes with HTTP 200: HTML error pages,
# error envelopes, content-filter verdicts and empty choice lists.
# ---------------------------------------------------------------------------

#: HTTP-200 bodies a real gateway can return instead of a chat completion.
_MALFORMED_PAYLOADS = [
    pytest.param({}, id="empty-object"),
    pytest.param({"error": {"message": "quota exceeded", "code": "rate_limit"}},
                 id="error-envelope-with-200"),
    pytest.param({"choices": []}, id="empty-choices"),
    pytest.param({"choices": [{}]}, id="choice-without-message"),
    pytest.param({"choices": [{"message": {}}]}, id="message-without-content"),
    pytest.param({"choices": [{"message": {"content": None}}]}, id="null-content"),
    pytest.param({"choices": "not-a-list"}, id="choices-not-a-list"),
    pytest.param({"choices": ["not-a-dict"]}, id="choice-not-a-dict"),
    pytest.param([{"choices": []}], id="top-level-list"),
]


@pytest.mark.parametrize("payload", _MALFORMED_PAYLOADS)
def test_malformed_success_payload_raises_llm_error(monkeypatch, payload) -> None:
    """Every malformed HTTP-200 shape surfaces as ``LLMError``, not as a raw builtin.

    ``LLMError`` is the contract the bounded retry and the rule-extractor fallback are
    written against; anything else bypasses both.

    **Validates: Requirements 10.4, 10.5, 11.1**
    """
    provider = _provider(monkeypatch)
    monkeypatch.setattr(provider, "_post", lambda body: payload)

    with pytest.raises(LLMError):
        provider.complete_json(PROMPT, purpose="intent_extraction")


def test_non_json_body_on_a_200_raises_llm_error(monkeypatch) -> None:
    """An HTML error page returned with HTTP 200 is an ``LLMError``, not a ValueError.

    ``json.JSONDecodeError`` is a ``ValueError`` and not an httpx error, so it matched
    none of the provider's handlers.

    **Validates: Requirements 10.4, 10.5**
    """
    provider = _provider(monkeypatch)
    request = httpx.Request("POST", "https://example.invalid/v1/chat/completions")

    def _client_post(*_args, **_kwargs) -> httpx.Response:
        return httpx.Response(200, request=request, text="<html>gateway timeout</html>")

    monkeypatch.setattr(httpx.Client, "post", _client_post)

    with pytest.raises(LLMError):
        provider.complete_json(PROMPT, purpose="intent_extraction")


def test_malformed_payload_falls_back_to_the_rule_extractor(monkeypatch, tmp_path) -> None:
    """A run whose every attempt returns a malformed body still COMPLETES on rules.

    This is the property that matters for a long batch: the turn degrades, the run
    finishes, and the experiment keeps going.

    **Validates: Requirements 10.4, 10.5**
    """
    monkeypatch.setenv(API_KEY_ENV, FAKE_KEY)
    config = load_config("configs/hybrid_vectorengine.yaml", base_dir="configs")
    service = AppService(config, CATALOG_PATH)
    candidate = service.create_candidate(
        {"candidate_id": "c-malformed", "skills": ["Python"], "years_experience": 2})
    session_id = service.create_session(candidate.candidate_id, "full")
    orchestrator, _store = service._orchestrator_for(session_id, "full")
    # The real remote provider, with a transport that always returns an error envelope.
    monkeypatch.setattr(orchestrator.provider, "_post",
                        lambda body: {"error": {"message": "upstream unavailable"}})

    result = service.process_turn(
        session_id, "I want a data analyst role in Kuala Lumpur, at least RM4000.")

    assert result.run_record.success is True
    assert result.extracted_preferences.preferences
    assert all(p.metadata["extraction_method"] == "rule"
               for p in result.extracted_preferences.preferences)
    # Every spent attempt is recorded, so the degradation is auditable.
    assert result.model_calls, "the failed attempts left no record"
    assert all(c.metadata.get("failed") for c in result.model_calls)
