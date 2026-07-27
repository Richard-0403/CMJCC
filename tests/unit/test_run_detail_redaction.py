"""Unit tests for the run-detail levels and redaction of the SQL repository (R12.1-R12.3).

The read path is exercised against a real SQLAlchemy engine (in-memory SQLite, the
ORM is deliberately portable) with rows written directly, so ``get_run`` is tested
without a PostgreSQL server and without mocks.
"""

from __future__ import annotations

import pytest

from jobrec.config import AppConfig
from jobrec.llm.provider import LLMCallRecord
from jobrec.storage.repositories import SqlRepository
from jobrec.utils.redaction import (
    REDACTED,
    REDACTED_EMAIL,
    REDACTED_KEY,
    REDACTED_PHONE,
    redact,
    redact_payload,
)
from jobrec.utils.time import utcnow

RUN_ID = "run-r12"
CANDIDATE_ID = "cand-r12"
SESSION_ID = "sess-r12"
# The phone ends the sentence, the shape candidates actually use.
RAW_RESPONSE = '{"raw_text": "I am Aisyah, call me on +60 12-345 6789."}'


# --------------------------------------------------------------------- helpers
@pytest.fixture()
def session_factory():
    """A session factory on a fresh in-memory SQLite database with the schema applied."""
    from jobrec.storage.db import create_all, make_engine, make_session_factory

    engine = make_engine("sqlite://")
    create_all(engine)
    return make_session_factory(engine)


def _seed(session_factory) -> None:
    """Write the minimum row set for one run: run record, both state versions, one model call."""
    from jobrec.storage.models import (
        CandidateStateVersion,
        DialogueStateVersion,
        ModelCallRow,
        RunRecordRow,
    )

    with session_factory() as s:
        s.add(CandidateStateVersion(
            candidate_id=CANDIDATE_ID, version=1, updated_at=utcnow(),
            payload={"candidate_id": CANDIDATE_ID, "version": 1}))
        s.add(CandidateStateVersion(
            candidate_id=CANDIDATE_ID, version=2, updated_at=utcnow(),
            payload={"candidate_id": CANDIDATE_ID, "version": 2}))
        s.add(DialogueStateVersion(
            session_id=SESSION_ID, version=1, candidate_id=CANDIDATE_ID,
            payload={"session_id": SESSION_ID, "version": 1, "raw_text": "remote only"}))
        s.add(ModelCallRow(
            call_id="call-1", run_id=RUN_ID, purpose="extraction", provider="mock", model="m",
            payload={"purpose": "extraction", "latency_ms": 1.5, "raw_response": RAW_RESPONSE}))
        s.add(RunRecordRow(
            run_id=RUN_ID, scenario_id=None, session_id=SESSION_ID, candidate_id=CANDIDATE_ID,
            experiment_variant="full", success=True, failure_code=None, total_latency_ms=10.0,
            config_hash="c", catalog_hash="k", prompt_hash="p",
            payload={"run_id": RUN_ID, "state_object_ids": {
                "candidate_state": f"{CANDIDATE_ID}:v1",
                "dialogue_state": f"{SESSION_ID}:v1",
                "active_search_state": "search-1"}}))
        s.commit()


# ------------------------------------------------------------ redact() helper
def test_redact_strips_credentials_and_pii():
    text = "key sk-ABCDEFGH1234 api_key=supersecret mail me@example.com or +60 12-345 6789"
    out = redact(text)

    assert "sk-ABCDEFGH1234" not in out
    assert "supersecret" not in out
    assert "me@example.com" not in out
    assert "12-345 6789" not in out
    assert "mail" in out, "non-sensitive words survive redaction"


def test_redact_is_idempotent_and_preserves_short_numbers():
    text = "salary 4000 on 2026-01-01 from me@example.com"
    once = redact(text)

    assert redact(once) == once
    assert "4000" in once and "2026-01-01" in once


@pytest.mark.parametrize("closer", [".", "?", "!", ")", ".)", "...", ". "])
@pytest.mark.parametrize("phone", ["+60 12-345 6789", "012-345 6789", "+60123456789"])
def test_redact_catches_phone_numbers_at_the_end_of_a_sentence(phone, closer):
    """A number followed by sentence-final punctuation is still PII (R12.3)."""
    out = redact(f"call me on {phone}{closer}")

    assert phone not in out, "trailing punctuation must not shield a phone number"
    assert out == f"call me on {REDACTED_PHONE}{closer}"


@pytest.mark.parametrize("closer", [".", "?", "!", ")"])
def test_redact_catches_emails_and_keys_at_the_end_of_a_sentence(closer):
    """The sibling credential/e-mail patterns keep working against trailing punctuation."""
    assert redact(f"mail me@example.com{closer}") == f"mail {REDACTED_EMAIL}{closer}"
    assert redact(f"key sk-ABCDEFGH1234{closer}") == f"key {REDACTED_KEY}{closer}"


@pytest.mark.parametrize(
    "text",
    [
        "salary 4000.50 per month",
        "offer 4000.50.",
        "reference date 2026-01-01",
        "version 1.2.3 shipped",
        "logged at 2026-01-01T10:00:00.123456",
        "score 0.123456789 recorded",
    ],
)
def test_redact_preserves_decimals_dates_and_versions(text):
    """The digit-count guard and the ``.``-aware lookahead keep numeric data intact."""
    assert redact(text) == text


def test_redact_candidate_text_replaces_the_whole_string():
    assert redact("I want a remote data analyst job", redact_candidate_text=True) == REDACTED
    assert redact("", redact_candidate_text=True) == ""


def test_redact_payload_keeps_structure_and_only_drops_free_text():
    payload = {"call_id": "call-1", "latency_ms": 1.5, "raw_response": "hi, I am Aisyah"}

    out = redact_payload(payload, redact_candidate_text=True)

    assert out["call_id"] == "call-1"
    assert out["latency_ms"] == 1.5
    assert out["raw_response"] == REDACTED


# ------------------------------------------------------------------- get_run
def test_details_are_omitted_unless_requested(session_factory):
    _seed(session_factory)

    run = SqlRepository(session_factory).get_run(RUN_ID)

    assert run is not None
    assert run["run_record"]["run_id"] == RUN_ID
    assert "states" not in run
    assert "raw_model_outputs" not in run


def test_include_states_returns_the_pinned_state_versions(session_factory):
    _seed(session_factory)

    run = SqlRepository(session_factory).get_run(RUN_ID, include_states=True)

    assert run is not None
    # v1 is pinned by state_object_ids even though v2 exists.
    assert run["states"]["candidate_state"]["version"] == 1
    assert run["states"]["dialogue_state"]["session_id"] == SESSION_ID
    assert run["states"]["active_search_id"] == "search-1"


def test_include_raw_model_outputs_returns_redacted_outputs(session_factory):
    _seed(session_factory)

    run = SqlRepository(session_factory).get_run(RUN_ID, include_raw_model_outputs=True)

    assert run is not None
    call = run["raw_model_outputs"][0]
    assert call["call_id"] == "call-1"
    assert call["purpose"] == "extraction"
    assert "prompt" not in call, "prompts are never returned"
    assert "12-345 6789" not in call["raw_response"]


def test_redact_candidate_text_config_redacts_states_and_outputs(session_factory):
    _seed(session_factory)
    config = AppConfig()
    config.logging.redact_candidate_text = True

    run = SqlRepository(session_factory, config).get_run(
        RUN_ID, include_states=True, include_raw_model_outputs=True)

    assert run is not None
    assert run["raw_model_outputs"][0]["raw_response"] == REDACTED
    assert run["raw_model_outputs"][0]["purpose"] == "extraction"
    assert run["states"]["dialogue_state"]["raw_text"] == REDACTED


def test_unknown_run_is_none(session_factory):
    assert SqlRepository(session_factory).get_run("no-such-run", include_states=True) is None


# ------------------------------------------------------- stored model payload
def _call_record() -> LLMCallRecord:
    return LLMCallRecord(
        call_id="call-1", purpose="extraction", prompt="candidate said: hi",
        raw_response=RAW_RESPONSE, parsed_ok=True, latency_ms=1.5,
        provider="mock", model="m")


def test_stored_model_call_payload_never_holds_prompts_or_pii(session_factory):
    payload = SqlRepository(session_factory)._model_call_payload(_call_record())

    assert "prompt" not in payload
    assert payload["latency_ms"] == 1.5
    assert "12-345 6789" not in payload["raw_response"]


def test_save_raw_responses_off_drops_the_raw_response(session_factory):
    config = AppConfig()
    config.llm.save_raw_responses = False

    payload = SqlRepository(session_factory, config)._model_call_payload(_call_record())

    assert "raw_response" not in payload
