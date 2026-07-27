"""API-level tests for the run-detail parameters and redaction (R12.4).

These exercise the real HTTP surface (`GET /v1/runs/{run_id}`) with a FastAPI
``TestClient`` over a real :class:`~jobrec.app_service.AppService` and the shipped
catalog, no mocks. The repository-level behaviour (state-version pinning, SQL
loading) is covered by ``tests/unit/test_run_detail_redaction.py``; what is checked
here is that the endpoint really surfaces ``include_states`` and
``include_raw_model_outputs``, and that the redaction of R12.3 is visible in the
HTTP response body.

The run is produced by a genuine turn in deterministic mode, driven through the
same service the app serves. Deterministic mode calls no model, so the one test
that needs raw model outputs persists a real :class:`LLMCallRecord` alongside the
turn through the repository's own ``save_turn``; the record carries a prompt and
PII so the response can be inspected for leaks.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from fastapi.testclient import TestClient

from jobrec.api.app import create_app
from jobrec.app_service import AppService
from jobrec.config import load_config
from jobrec.llm.provider import LLMCallRecord
from jobrec.utils.redaction import REDACTED

CATALOG = "data/processed/jobs.jsonl"
CANDIDATE_ID = "cand-r12-api"
EMAIL = "aisyah.binti@example.com"
PHONE = "+60 12-345 6789"
SECRET = "supersecret123"
UTTERANCE = (
    "I want a data analyst role in Kuala Lumpur, hybrid is fine, at least RM4000. "
    # The phone ends the sentence: the real-world shape that used to slip past
    # the phone pattern's trailing lookahead.
    f"Mail me at {EMAIL}, or call {PHONE}. My api_key={SECRET} unlocks my portfolio."
)


def _model_call() -> LLMCallRecord:
    """A real model-call record whose prompt quotes the candidate and whose output has PII."""
    return LLMCallRecord(
        call_id="call-api-1", purpose="intent_extraction",
        prompt=f"Extract preferences.\nUtterance: {UTTERANCE}",
        raw_response=json.dumps({"raw_text": f"reach me at {EMAIL} or {PHONE}."}),
        parsed_ok=True, latency_ms=1.5, provider="mock", model="mock-deterministic-v1")


def _api(*, redact_candidate_text: bool = False,
         model_calls: Sequence[LLMCallRecord] = ()) -> tuple[TestClient, str]:
    """Run one real turn and return a client plus the resulting run id."""
    cfg = load_config("configs/experiment_full.yaml", base_dir="configs")
    cfg.logging.redact_candidate_text = redact_candidate_text
    svc = AppService(cfg, CATALOG)
    svc.create_candidate({
        "candidate_id": CANDIDATE_ID, "skills": ["Python", "SQL"], "years_experience": 1,
        "target_roles": ["Data Analyst"], "preferred_locations": ["Kuala Lumpur"],
        "salary_min": 4000, "salary_currency": "MYR", "work_modes": ["hybrid"]})
    session_id = svc.create_session(CANDIDATE_ID, "full")
    result = svc.process_turn(session_id, UTTERANCE)
    assert result.run_record.success, "the fixture run must be a successful turn"
    if model_calls:
        # Deterministic mode records no model call; persist the run again with one
        # so the raw-model-output path has something to return.
        svc.repo.save_turn(result, [], list(model_calls))
    return TestClient(create_app(svc)), result.run_record.run_id


def _get_run(client: TestClient, run_id: str, **params: bool) -> dict:
    response = client.get(f"/v1/runs/{run_id}", params=params)
    assert response.status_code == 200
    return response.json()


# ------------------------------------------------------------------ parameters
def test_detail_sections_are_absent_unless_requested():
    client, run_id = _api()

    run = _get_run(client, run_id)

    assert run["run_record"]["run_id"] == run_id
    assert "states" not in run
    assert "raw_model_outputs" not in run


def test_include_states_returns_the_state_objects():
    client, run_id = _api()

    run = _get_run(client, run_id, include_states=True)

    states = run["states"]
    assert states["candidate_state"]["candidate_id"] == CANDIDATE_ID
    assert states["candidate_state"]["version"] >= 1
    assert states["dialogue_state"]["turns"], "the candidate turn is part of the state"
    assert "raw_model_outputs" not in run, "one parameter must not enable the other"


def test_include_raw_model_outputs_returns_the_recorded_calls():
    client, run_id = _api(model_calls=[_model_call()])

    run = _get_run(client, run_id, include_raw_model_outputs=True)

    calls = run["raw_model_outputs"]
    assert [call["call_id"] for call in calls] == ["call-api-1"]
    assert calls[0]["purpose"] == "intent_extraction"
    assert calls[0]["latency_ms"] == 1.5
    assert "states" not in run


def test_both_parameters_can_be_requested_together():
    client, run_id = _api(model_calls=[_model_call()])

    run = _get_run(client, run_id, include_states=True, include_raw_model_outputs=True,
                   include_evidence=True, include_handoffs=True)

    assert set(run) >= {"run_record", "states", "raw_model_outputs", "evidence_log", "handoffs"}
    assert run["states"]["dialogue_state"]["session_id"]
    assert run["raw_model_outputs"]


# ------------------------------------------------------------------- redaction
def test_prompts_are_never_returned():
    client, run_id = _api(model_calls=[_model_call()])

    run = _get_run(client, run_id, include_raw_model_outputs=True)

    calls = run["raw_model_outputs"]
    assert all("prompt" not in call for call in calls)
    assert "Utterance:" not in json.dumps(calls), "no prompt text leaks through another field"


def test_credentials_and_pii_are_redacted_in_the_returned_details():
    client, run_id = _api(model_calls=[_model_call()])

    run = _get_run(client, run_id, include_states=True, include_raw_model_outputs=True)

    detail = json.dumps({"states": run["states"], "calls": run["raw_model_outputs"]})
    assert EMAIL not in detail
    assert "12-345 6789" not in detail
    assert SECRET not in detail
    assert "[REDACTED_EMAIL]" in detail and "[REDACTED_PHONE]" in detail
    # Redaction, not deletion: non-sensitive content still reaches the evaluator.
    assert "Kuala Lumpur" in detail


def test_redact_candidate_text_config_wipes_free_text_over_the_api():
    client, run_id = _api(redact_candidate_text=True, model_calls=[_model_call()])

    run = _get_run(client, run_id, include_states=True, include_raw_model_outputs=True)

    turn = run["states"]["dialogue_state"]["turns"][0]
    assert turn["text"] == REDACTED
    assert run["raw_model_outputs"][0]["raw_response"] == REDACTED
    # Structural fields stay inspectable for the evaluator.
    assert turn["speaker"] in {"candidate", "system"}
    assert run["states"]["dialogue_state"]["candidate_id"] == CANDIDATE_ID
    assert run["raw_model_outputs"][0]["purpose"] == "intent_extraction"


def test_unknown_run_returns_404():
    client, _ = _api()

    assert client.get("/v1/runs/no-such-run", params={"include_states": True}).status_code == 404
