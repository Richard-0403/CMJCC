"""Failure-path unit tests for evidence grounding, handoffs and recovery (R10).

Task 11.3. These are concrete, example-based negatives -- the enumerated failure cases
Requirement 10 demands coverage for -- exercised against the real validators and the real
orchestrator, never against reimplemented logic:

* **Grounding (R10.1, R10.2, R10.6).** An invalid (dangling) evidence id, a claim with no
  source at all, a claim pointing at the *wrong field*, a partially grounded claim, and
  unsupported salary / location / skill claims all go through
  :func:`jobrec.agents.explanation_agent.validate_claims` and
  :class:`~jobrec.agents.explanation_agent.ExplanationAgent`.
* **Handoffs (R10.3, R10.7).** Schema-invalid and missing-required-field payloads are fed
  to the real :class:`~jobrec.domain.handoff.AgentHandoff` model, and a run whose handoff
  failed is checked never to be scored as a success.
* **Recovery (R10.4).** An agent exception, a timeout retried through
  :func:`jobrec.llm.retry.retry_call`, and a partial failure that recovers (either on the
  retry or via the rule fallback) are driven with
  :class:`tests.support.fault_injection.FaultInjectingProvider`.

**How "the event and the final status" (R10.5) are asserted.** Per the design, each of
these events is recorded on the run's decision log rather than only on the Python logging
channel, so the assertions follow the channel the code actually uses:

* a rejected claim is recorded as a claim carrying ``support_status="unsupported"`` and is
  absent from the delivered response (its final status);
* a failed handoff is recorded as an :class:`AgentHandoff` with ``status="failed"``,
  ``validation_passed=False`` and an ``error_code``, alongside ``RunRecord.success=False``;
* an invalid handoff *payload* is reported by the validation error, which names the
  offending field;
* the retry / fallback rungs additionally log on the ``jobrec.orchestration.orchestrator``
  logger, asserted with ``caplog`` exactly as ``tests/unit/test_field_validation.py`` does.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pydantic import ValidationError

from jobrec.agents.explanation_agent import ExplanationAgent, validate_claims
from jobrec.app_service import AppService
from jobrec.config import load_config
from jobrec.domain.dialogue import ClarificationAction
from jobrec.domain.enums import (
    ConfirmationStatus,
    ErrorCode,
    EvidenceSource,
    PersistenceScope,
    ResponseType,
    RunMode,
    WorkflowState,
)
from jobrec.domain.handoff import AgentHandoff
from jobrec.domain.recommendation import RecommendationDecision, Response, ResponseClaim
from jobrec.domain.run_record import RunRecord
from jobrec.evidence_store import EvidenceStore
from jobrec.llm.provider import LLMTimeout
from jobrec.llm.retry import retry_call
from jobrec.orchestration.orchestrator import TurnResult
from jobrec.prompts import render_intent_extraction
from jobrec.utils.time import utcnow
from jobrec_eval.loaders import RunBundle
from jobrec_eval.metrics import MetricsComputer
from jobrec_eval.scenarios import Scenario
from tests.conftest import CATALOG_PATH, make_active
from tests.support.fault_injection import (
    DANGLING_EVIDENCE_ID,
    REQUIRED_HANDOFF_FIELDS,
    FaultInjectingProvider,
    make_claim,
    make_dangling_claim,
    make_unsupported_claim,
    missing_field_handoff_payload,
    schema_invalid_handoff_payload,
    valid_handoff_payload,
)

ORCHESTRATOR_LOGGER = "jobrec.orchestration.orchestrator"

#: An utterance the pipeline resolves to a recommendation in the deterministic catalog.
UTTERANCE = "I want a data analyst role in Kuala Lumpur, at least RM4000."

#: Response types a *successful* run may end with (an error type means the run failed).
NON_ERROR_RESPONSES = {
    ResponseType.RECOMMENDATION.value,
    ResponseType.CLARIFICATION.value,
    ResponseType.NO_MATCH.value,
}


# --------------------------------------------------------------------------- #
# evidence helpers
# --------------------------------------------------------------------------- #
def _register(
    store: EvidenceStore,
    field_name: str,
    value: Any,
    *,
    object_id: str = "cand-failure",
    source: EvidenceSource = EvidenceSource.DIALOGUE,
) -> str:
    """Register one real EvidenceItem in ``store`` and return its id."""
    item = store.register_field(
        source,
        object_id,
        field_name,
        value,
        confidence=1.0,
        confirmation=ConfirmationStatus.CONFIRMED,
        scope=PersistenceScope.SESSION,
    )
    return item.evidence_id


def _evidence_id_for(
    field_name: str,
    value: Any,
    *,
    object_id: str = "cand-failure",
    source: EvidenceSource = EvidenceSource.DIALOGUE,
) -> str:
    """The id the registry *would* mint for this (source, object, field, value).

    Minted in a throwaway store, so the id is well-formed and computed by the real
    id function, yet is not registered anywhere the validator can see it. Because
    evidence ids are content-addressed over the field name, this is also how a
    "wrong field" reference is produced: the same value under a different field
    name yields a different, unresolvable id.
    """
    return _register(EvidenceStore(), field_name, value, object_id=object_id, source=source)


def _decision(active, *, no_match: bool = False, reason_codes: tuple[str, ...] = ()):
    """A minimal, schema-valid decision for driving ExplanationAgent claim generation."""
    return RecommendationDecision(
        decision_id="dec-failure-paths",
        session_id=active.session_id,
        active_search_id=active.active_search_id,
        context_id=None,
        experiment_variant="full",
        retrieved_job_ids=[],
        eligibility_results=[],
        ranked_jobs=[],
        selected_job_ids=[],
        no_match=no_match,
        no_match_reason_codes=list(reason_codes),
        created_at=utcnow(),
        scorer_version="test",
        config_hash="cfg-hash",
    )


# --------------------------------------------------------------------------- #
# R10.1 / R10.6 -- invalid, missing and wrong-field evidence
# --------------------------------------------------------------------------- #
def test_dangling_evidence_id_claim_is_flagged_unsupported_and_never_presented() -> None:
    """A claim whose evidence id resolves to nothing is rejected, not shown (R10.1/10.6)."""
    store = EvidenceStore()
    # candidate_preference: the proposition is about what the candidate asked for, and the
    # evidence is their own stated work mode. The type has to match the evidence now that
    # semantic_status checks whether that KIND of evidence can carry the claim.
    grounded = make_claim(
        claim_type="candidate_preference",
        text="You asked for a hybrid role.",
        evidence_ids=[_register(store, "work_modes", ["hybrid"])],
        predicate="candidate_preference", field_name="work_modes",
        expected_value=["hybrid"],
    )
    dangling = make_dangling_claim(predicate="candidate_preference",
                                    field_name="work_modes", expected_value=["hybrid"])
    assert not store.exists(DANGLING_EVIDENCE_ID), "the dangling id must not resolve"

    supported, dropped = validate_claims([grounded, dangling], store)

    # Event: the rejected claim is recorded, flagged unsupported.
    assert [c.claim_id for c in dropped] == [dangling.claim_id]
    assert [c.support_status for c in dropped] == ["unsupported"]
    # Final status: only the grounded claim survives for presentation.
    assert [c.claim_id for c in supported] == [grounded.claim_id]
    assert [c.support_status for c in supported] == ["supported"]
    # The validator never mutates its inputs. The incoming default is now "unknown", not
    # "supported": a claim no validator has examined must not arrive already believed.
    assert dangling.support_status == "unknown"


def test_claim_with_a_missing_source_is_flagged_unsupported() -> None:
    """A claim carrying no evidence ids at all has no source, so it is rejected (R10.2/10.6)."""
    store = EvidenceStore()
    claim = make_unsupported_claim()
    assert claim.evidence_ids == []

    supported, dropped = validate_claims([claim], store)

    assert supported == []
    assert [c.support_status for c in dropped] == ["unsupported"]


def test_claim_referencing_the_wrong_field_does_not_resolve_to_evidence() -> None:
    """Evidence ids are field-scoped: pointing at the wrong field is rejected (R10.1/10.6)."""
    store = EvidenceStore()
    value = ["Kuala Lumpur"]
    right_field_id = _register(store, "preferred_locations", value)
    wrong_field_id = _evidence_id_for("salary_min", value)

    # Same source object and same value, different field -> a different, unregistered id.
    assert wrong_field_id != right_field_id
    assert not store.exists(wrong_field_id)

    claim = make_claim(
        claim_type="candidate_preference",
        text="You said your minimum salary is Kuala Lumpur.",
        evidence_ids=[wrong_field_id],
    )
    supported, dropped = validate_claims([claim], store)

    assert supported == []
    assert [c.support_status for c in dropped] == ["unsupported"]


def test_partially_grounded_claim_is_rejected_as_a_whole() -> None:
    """One unresolvable id is enough to reject a claim: grounding is all-or-nothing (R10.6)."""
    store = EvidenceStore()
    good_id = _register(store, "target_roles", ["data analyst"])
    claim = make_claim(
        text="You want a data analyst role at a famously supportive company.",
        evidence_ids=[good_id, DANGLING_EVIDENCE_ID],
    )

    supported, dropped = validate_claims([claim], store)

    assert supported == []
    assert [c.support_status for c in dropped] == ["unsupported"]


# --------------------------------------------------------------------------- #
# R10.2 -- unsupported salary / location / skill claims
# --------------------------------------------------------------------------- #
#: (claim_type, job field, value, claim text) for the three claim families R10.2 names.
UNSUPPORTED_CLAIM_CASES = [
    (
        "job_attribute",
        "salary_min_monthly_myr",
        8000.0,
        "This role pays at least RM8000 per month.",
    ),
    ("job_attribute", "city", "Penang", "This role is based in Penang."),
    (
        "skill_gap",
        "required_skills",
        ["kubernetes"],
        "The role requires kubernetes, which is not in your listed skills.",
    ),
]


@pytest.mark.parametrize(("claim_type", "field_name", "value", "text"), UNSUPPORTED_CLAIM_CASES)
def test_unsupported_salary_location_and_skill_claims_are_flagged(
    claim_type: str, field_name: str, value: Any, text: str
) -> None:
    """Salary, location and skill claims are all rejected while their evidence is absent.

    Registering exactly the same evidence afterwards flips the identical claim to
    supported, so the rejection is genuinely about grounding (R10.2, R10.6).
    """
    store = EvidenceStore()
    evidence_id = _evidence_id_for(
        field_name, value, object_id="job-1", source=EvidenceSource.JOB_POSTING
    )
    # The proposition each case asserts. Without it the verdict would be ``unknown`` -- "no
    # checker can rule on this" -- rather than ``unsupported``, which is what these cases are
    # about: the evidence is absent, then present, and the SAME claim flips.
    proposition: dict = (
        {"predicate": "skill_not_recorded", "job_id": "job-1",
         "claim_args": {"skill": "kubernetes"}}
        if claim_type == "skill_gap"
        else {"predicate": "job_attribute", "job_id": "job-1",
              "field_name": field_name, "observed_value": value}
    )
    claim = make_claim(claim_type=claim_type, text=text, evidence_ids=[evidence_id],
                       **proposition)

    supported, dropped = validate_claims([claim], store)
    assert supported == []
    assert [c.support_status for c in dropped] == ["unsupported"]

    _register(store, field_name, value, object_id="job-1", source=EvidenceSource.JOB_POSTING)
    if claim_type == "skill_gap":
        # A skill gap compares two things, so job-side evidence alone is not enough to flip
        # it: the claim also has to cite what the candidate's record says. Supplying only
        # the job's requirement is exactly the state in which 1883 of these were
        # adjudicated unsupported.
        supported, dropped = validate_claims([claim], store)
        assert supported == []
        assert [c.semantic_status for c in dropped] == ["unsupported"]
        claim = claim.model_copy(update={"evidence_ids": [
            *claim.evidence_ids,
            _register(store, "skills_have", ["python"], object_id="cand-1",
                      source=EvidenceSource.PROFILE),
        ]})
    supported, dropped = validate_claims([claim], store)
    assert dropped == []
    assert [c.support_status for c in supported] == ["supported"]


def test_explanation_agent_excludes_ungrounded_claims_from_the_response(config) -> None:
    """The real agent presents only grounded claims and returns the rejected ones (R10.6)."""
    store = EvidenceStore()
    roles_id = _register(store, "target_roles", ["data analyst"])
    wrong_field_id = _evidence_id_for("salary_min", ["Kuala Lumpur"])
    active = make_active(
        field_evidence_map={
            "target_roles": [roles_id],
            "preferred_locations": [wrong_field_id],
            "salary_min": [DANGLING_EVIDENCE_ID],
        }
    )

    response, dropped = ExplanationAgent(store, config).explain(_decision(active), active, {})

    # Final status: every delivered claim resolves to registered evidence.
    assert len(response.claims) == 1
    assert all(store.exists(e) for c in response.claims for e in c.evidence_ids)
    assert "target roles" in response.claims[0].text
    assert response.claims[0].support_status == "supported"

    # Event: both ungrounded claims are recorded, flagged unsupported.
    assert len(dropped) == 2
    assert {c.support_status for c in dropped} == {"unsupported"}
    dropped_text = " | ".join(c.text for c in dropped)
    assert "preferred locations" in dropped_text
    assert "salary min" in dropped_text


def test_no_match_response_excludes_ungrounded_no_match_reasons(config) -> None:
    """No-match reasons are grounded too: an ungrounded reason is dropped (R10.6)."""
    store = EvidenceStore()
    salary_id = _register(store, "salary_min", 4000.0)
    active = make_active(
        hard_constraint_fields=["salary_min", "preferred_locations"],
        field_evidence_map={
            "salary_min": [salary_id],
            "preferred_locations": [DANGLING_EVIDENCE_ID],
        },
    )
    decision = _decision(active, no_match=True, reason_codes=("salary_min",))

    response, dropped = ExplanationAgent(store, config).explain(decision, active, {})

    assert response.response_type == ResponseType.NO_MATCH.value
    assert len(response.claims) == 1
    assert "salary min" in response.claims[0].text
    assert len(dropped) == 1
    assert dropped[0].support_status == "unsupported"
    assert "preferred locations" in dropped[0].text


# --------------------------------------------------------------------------- #
# R10.3 -- schema-invalid and missing-field handoffs
# --------------------------------------------------------------------------- #
def test_valid_handoff_payload_is_accepted_as_the_baseline() -> None:
    """The mutation base really is valid, so the negatives below isolate one defect each."""
    handoff = AgentHandoff(**valid_handoff_payload())

    assert handoff.status == "attempted"
    assert handoff.validation_passed is False
    assert handoff.completed_at is None


def test_schema_invalid_handoff_is_rejected_and_names_the_offending_field() -> None:
    """An out-of-vocabulary ``status`` fails validation, reporting that field (R10.3)."""
    payload = schema_invalid_handoff_payload()
    assert set(REQUIRED_HANDOFF_FIELDS) <= set(payload), "not a missing-field case"

    with pytest.raises(ValidationError) as excinfo:
        AgentHandoff(**payload)

    errors = excinfo.value.errors()
    assert [e["loc"] for e in errors] == [("status",)]
    assert errors[0]["type"] == "literal_error"


def test_handoff_with_an_unknown_extra_field_is_rejected() -> None:
    """The contract forbids unknown fields, so a widened payload is invalid (R10.3)."""
    with pytest.raises(ValidationError) as excinfo:
        AgentHandoff(**valid_handoff_payload(unexpected_field="surprise"))

    errors = excinfo.value.errors()
    assert [e["loc"] for e in errors] == [("unexpected_field",)]
    assert errors[0]["type"] == "extra_forbidden"


@pytest.mark.parametrize("omitted", REQUIRED_HANDOFF_FIELDS)
def test_handoff_missing_a_required_field_is_rejected(omitted: str) -> None:
    """Every required field is genuinely required, and the loss is reported (R10.3)."""
    payload = missing_field_handoff_payload(omit=omitted)
    assert omitted not in payload

    with pytest.raises(ValidationError) as excinfo:
        AgentHandoff(**payload)

    errors = excinfo.value.errors()
    assert [e["loc"] for e in errors] == [(omitted,)]
    assert errors[0]["type"] == "missing"


def test_handoff_missing_several_required_fields_reports_every_loss() -> None:
    """Multiple omissions are all reported, not just the first (R10.3)."""
    omitted = ["contract_name", "status"]

    with pytest.raises(ValidationError) as excinfo:
        AgentHandoff(**missing_field_handoff_payload(omit=omitted))

    losses = sorted(e["loc"][0] for e in excinfo.value.errors())
    assert losses == sorted(omitted)
    assert {e["type"] for e in excinfo.value.errors()} == {"missing"}


# --------------------------------------------------------------------------- #
# R10.4 / R10.5 / R10.7 -- agent exception, retry, recovery, and run status
# --------------------------------------------------------------------------- #
class ExplodingRankingAgent:
    """Stand-in ranking agent that raises inside the orchestrator's guarded region."""

    def __init__(self, name: str, error: Exception) -> None:
        self.name = name
        self._error = error
        self.calls = 0

    def rank(self, *args: Any, **kwargs: Any) -> Any:
        self.calls += 1
        raise self._error


def _service() -> AppService:
    """An AppService on the real catalog with the in-memory repository."""
    config = load_config("configs/experiment_full.yaml", base_dir="configs")
    return AppService(config, CATALOG_PATH)


def _session(service: AppService, candidate_id: str) -> str:
    service.create_candidate(
        {
            "candidate_id": candidate_id,
            "skills": ["Python", "SQL"],
            "years_experience": 1,
            "preferred_locations": ["Kuala Lumpur"],
            "work_modes": ["hybrid"],
        }
    )
    return service.create_session(candidate_id, "full")


def _orchestrator_logs(caplog: Any) -> list[tuple[int, str]]:
    """The orchestrator's ``(level, message)`` log entries, in emission order."""
    return [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.name == ORCHESTRATOR_LOGGER
    ]


@pytest.fixture(scope="module")
def exploded_turn() -> TurnResult:
    """One real run whose ranking agent raises part-way through the workflow."""
    service = _service()
    session_id = _session(service, "cand-agent-exception")
    orchestrator, _store = service._orchestrator_for(session_id, "full")
    boom = ExplodingRankingAgent(orchestrator.ranking.name, RuntimeError("ranking exploded"))
    orchestrator.ranking = boom

    result = service.process_turn(session_id, UTTERANCE)
    assert boom.calls == 1, "the injected fault never fired"
    return result


def test_agent_exception_produces_a_failed_run_with_an_error_response(exploded_turn) -> None:
    """An agent exception is converted into an explicit failed run, never a silent success.

    Event: the workflow records the FAILED state and a failed handoff carrying
    ``INTERNAL_ERROR``. Final status: ``success=False`` with that failure code and an
    error response holding no claims (R10.4, R10.5).
    """
    record = exploded_turn.run_record

    assert record.success is False
    assert record.failure_code == ErrorCode.INTERNAL_ERROR.value
    assert record.workflow_states[-1] == WorkflowState.FAILED.value
    assert exploded_turn.decision is None

    assert exploded_turn.response.response_type == ResponseType.ERROR.value
    assert "RuntimeError" in exploded_turn.response.message
    assert exploded_turn.response.claims == []

    # The decision log still shows how far the run got before the fault.
    assert exploded_turn.evidence_log, "no decision-log entries recorded"
    completed = [h for h in exploded_turn.handoffs if h.status == "completed"]
    assert [h.contract_name for h in completed] == [
        "ExtractedPreferenceSet",
        "CMJCCOutput",
        "RetrievalOutcome",
        "EligibilityResults",
    ]


def test_failed_handoff_is_recorded_and_the_run_is_not_scored_as_a_success(
    exploded_turn,
) -> None:
    """An invalid handoff is logged with its error code and the run cannot be a success (R10.7)."""
    failed = [h for h in exploded_turn.handoffs if h.status == "failed"]

    assert len(failed) == 1
    assert failed[0].validation_passed is False
    assert failed[0].error_code == ErrorCode.INTERNAL_ERROR.value
    assert failed[0].completed_at is None
    assert failed[0].output_schema_version is None

    # R10.7: a run containing a non-validated handoff is never scored a success.
    assert not all(h.validation_passed for h in exploded_turn.handoffs)
    assert exploded_turn.run_record.success is False
    # Every handoff attempt, failed included, is referenced by the run record.
    assert exploded_turn.run_record.handoff_ids == [h.handoff_id for h in exploded_turn.handoffs]


def test_timeout_is_retried_and_the_following_attempt_succeeds() -> None:
    """A single injected timeout is absorbed by the bounded retry helper (R10.4)."""
    provider = FaultInjectingProvider(fail_times=1)
    prompt = render_intent_extraction(UTTERANCE)

    payload, call = retry_call(
        lambda: provider.complete_json(prompt, purpose="intent_extraction"), 2
    )

    assert provider.failures == 1
    assert provider.attempts == 2, "the call was not retried exactly once"
    assert "preferences" in payload
    assert call.parsed_ok is True


def test_timeout_beyond_the_retry_budget_raises_the_last_error() -> None:
    """Retries are bounded: once the budget is spent the timeout surfaces (R10.4)."""
    provider = FaultInjectingProvider(fail_times=5)
    prompt = render_intent_extraction(UTTERANCE)

    with pytest.raises(LLMTimeout) as excinfo:
        retry_call(lambda: provider.complete_json(prompt, purpose="intent_extraction"), 2)

    assert provider.attempts == 3, "attempts must be max_retries + 1"
    assert "injected LLMTimeout" in str(excinfo.value)

    provider.reset()
    assert (provider.attempts, provider.failures) == (0, 0)


def test_partial_failure_recovers_on_the_retry_and_the_run_still_succeeds(caplog) -> None:
    """A timeout on the first model call recovers on the retry; the run completes (R10.4/10.5)."""
    caplog.set_level(logging.WARNING, logger=ORCHESTRATOR_LOGGER)
    service = _service()
    session_id = _session(service, "cand-partial-recovery")
    orchestrator, _store = service._orchestrator_for(session_id, "full")
    provider = FaultInjectingProvider(fail_times=1)
    orchestrator.provider = provider
    orchestrator.config.llm.mode = RunMode.HYBRID
    orchestrator.config.llm.max_retries = 2

    result = service.process_turn(session_id, UTTERANCE)

    assert provider.failures == 1
    assert provider.attempts == 2, "the extraction call was not retried exactly once"
    # Final status: the run completed successfully after the recovery.
    assert result.run_record.success is True
    assert result.run_record.failure_code is None
    assert result.run_record.workflow_states[-1] == WorkflowState.COMPLETED.value
    assert all(h.validation_passed and h.status == "completed" for h in result.handoffs)
    assert result.response.response_type in NON_ERROR_RESPONSES
    # BOTH attempts are recorded -- the timed-out one and the recovered one -- and the
    # model value was used (no rule fallback). This assertion used to demand exactly
    # one record, which is what hid the failed attempt from every exported artifact:
    # a bundle showing one successful call was indistinguishable from a call that
    # succeeded first time, so retry cost and reliability were unmeasurable.
    assert len(result.model_calls) == 2
    failed, recovered = result.model_calls
    assert failed.metadata["failed"] is True and failed.parsed_ok is False
    assert failed.metadata["error"] == "LLMTimeout", "only the exception CLASS is kept"
    assert failed.raw_response == "" and failed.call_id.endswith("#failed1")
    assert recovered.parsed_ok is True and "failed" not in recovered.metadata
    assert recovered.call_id != failed.call_id, "a failed attempt must not shadow the recording"
    assert result.extracted_preferences.preferences
    assert all(
        p.metadata["extraction_method"] == "llm" for p in result.extracted_preferences.preferences
    )
    assert not [m for _level, m in _orchestrator_logs(caplog) if "rule extractor" in m]


def test_exhausted_timeouts_recover_via_the_rule_fallback_and_are_logged(caplog) -> None:
    """When the retry budget is spent the run recovers on rules, with a logged warning.

    Event: the orchestrator logs the model failure and the fallback. Final status: the
    run still completes successfully and every field is attributed to the rule
    extractor rather than fabricated (R10.4, R10.5).
    """
    caplog.set_level(logging.WARNING, logger=ORCHESTRATOR_LOGGER)
    service = _service()
    session_id = _session(service, "cand-rule-recovery")
    orchestrator, _store = service._orchestrator_for(session_id, "full")
    provider = FaultInjectingProvider(fail_times=9)
    orchestrator.provider = provider
    orchestrator.config.llm.mode = RunMode.HYBRID
    orchestrator.config.llm.max_retries = 1

    result = service.process_turn(session_id, UTTERANCE)

    assert provider.attempts == 2, "attempts must be max_retries + 1"
    # No SUCCESSFUL call is recorded, but every spent attempt is: the old expectation
    # of an empty list meant an exhausted-retry run and a run that never called the
    # model produced byte-identical evidence.
    assert len(result.model_calls) == 2, "every spent attempt must be recorded"
    assert all(c.metadata["failed"] is True for c in result.model_calls)
    assert not any(c.parsed_ok for c in result.model_calls)
    assert any(
        "model call failed; falling back to rule extractor" in message
        and level >= logging.WARNING
        for level, message in _orchestrator_logs(caplog)
    ), "the recovery was not logged"

    assert result.run_record.success is True
    assert result.run_record.failure_code is None
    assert result.response.response_type in NON_ERROR_RESPONSES
    assert result.extracted_preferences.preferences
    assert all(
        p.metadata["extraction_method"] == "rule" for p in result.extracted_preferences.preferences
    )


# --------------------------------------------------------------------------- #
# Property-based test (Property 18)
# --------------------------------------------------------------------------- #

#: (field, value) pairs a generated claim may cite. Values are distinct per field, and
#: evidence ids are content-addressed over the field name, so the id minted for one
#: field's value under a *different* field name never collides with a registered id.
_CLAIM_EVIDENCE_FIELDS: list[tuple[str, Any]] = [
    ("target_roles", ["data analyst"]),
    ("preferred_locations", ["Kuala Lumpur"]),
    ("work_modes", ["hybrid"]),
    ("salary_min", 4000.0),
    ("skills_have", ["python", "sql"]),
]

#: Claim types that may carry evidence in a delivered response.
_CLAIM_TYPES = ["candidate_preference", "job_attribute", "ranking_reason", "skill_gap"]

#: How a generated claim is grounded. Only ``grounded`` can legitimately end up
#: supported; the other four are the R10 failure shapes -- a dangling id, no source at
#: all, a wrong-field reference, and a partially grounded claim.
_GROUNDING_KINDS = ["grounded", "dangling", "no_source", "wrong_field", "partial"]

#: Mixed claim sets: each element is (grounding kind, evidence field index, claim type).
_CLAIM_SPECS = st.lists(
    st.tuples(
        st.sampled_from(_GROUNDING_KINDS),
        st.integers(min_value=0, max_value=len(_CLAIM_EVIDENCE_FIELDS) - 1),
        st.sampled_from(_CLAIM_TYPES),
    ),
    min_size=1,
    max_size=6,
)


def _claims_for(specs: list[tuple[str, int, str]], store: EvidenceStore) -> list[ResponseClaim]:
    """Build the mixed claim set described by ``specs``, registering the real evidence.

    Only the grounded portion of each claim is registered in ``store``, so every other
    id stays unresolvable there. The claim index is woven into the text, so the
    content-addressed claim ids stay distinct.
    """
    field_count = len(_CLAIM_EVIDENCE_FIELDS)
    claims: list[ResponseClaim] = []
    for index, (kind, field_index, claim_type) in enumerate(specs):
        field_name, value = _CLAIM_EVIDENCE_FIELDS[field_index]
        text = f"claim {index}: {kind} reference to {field_name}."
        if kind == "no_source":
            claims.append(make_unsupported_claim(claim_type=claim_type, text=text))
            continue
        if kind == "dangling":
            claims.append(make_dangling_claim(claim_type=claim_type, text=text))
            continue
        if kind == "wrong_field":
            other_field = _CLAIM_EVIDENCE_FIELDS[(field_index + 1) % field_count][0]
            evidence_ids = [_evidence_id_for(other_field, value)]
        elif kind == "partial":
            evidence_ids = [_register(store, field_name, value), DANGLING_EVIDENCE_ID]
        else:
            evidence_ids = [_register(store, field_name, value)]
        claims.append(make_claim(claim_type=claim_type, text=text, evidence_ids=evidence_ids))
    return claims


# Feature: cmjcc-experiment-readiness, Property 18: Every supported response claim resolves
# to a registered evidence id
@settings(max_examples=100)
@given(specs=_CLAIM_SPECS)
def test_property_every_supported_claim_resolves_to_registered_evidence(
    specs: list[tuple[str, int, str]],
) -> None:
    """Grounding partitions any claim mix: supported <=> every cited id resolves.

    Checked in both directions against the store's own resolution, for arbitrary mixes
    of grounded, dangling, source-less, wrong-field and partially grounded claims:
    nothing presented as supported has an id the store cannot resolve, and no claim
    with an absent or unresolvable id ever reaches the supported set.

    **Validates: Requirements 10.6, 31.4**
    """
    store = EvidenceStore()
    claims = _claims_for(specs, store)
    before = [claim.model_copy(deep=True) for claim in claims]

    supported, dropped = validate_claims(claims, store)

    # Direction 1: every delivered claim cites evidence, and all of it resolves.
    for claim in supported:
        assert claim.support_status == "supported"
        assert claim.evidence_ids, "a supported claim must cite at least one evidence id"
        assert all(store.exists(evidence_id) for evidence_id in claim.evidence_ids)

    # Direction 2: an unresolvable or absent id keeps a claim out of the supported set.
    supported_ids = {claim.claim_id for claim in supported}
    for claim in claims:
        resolves = bool(claim.evidence_ids) and all(
            store.exists(evidence_id) for evidence_id in claim.evidence_ids
        )
        if not resolves:
            assert claim.claim_id not in supported_ids

    # Every rejected claim is flagged, and one of the two dimensions says why. A dangling
    # reference is no longer the only reason to reject: evidence can resolve and still fail
    # to establish the proposition, which is the case the old "dropped implies dangling"
    # assertion could not express -- and the reason the validator passed all 11197 claims
    # of the official pair while human raters rejected 2349.
    for claim in dropped:
        assert claim.support_status in ("unsupported", "unknown")
        dangling = not claim.evidence_ids or not all(
            store.exists(evidence_id) for evidence_id in claim.evidence_ids
        )
        assert dangling or claim.semantic_status != "supported", claim

    # The partition is exhaustive: no claim is silently lost or duplicated.
    assert len(supported) + len(dropped) == len(claims)
    assert sorted(claim.claim_id for claim in supported + dropped) == sorted(
        claim.claim_id for claim in claims
    )
    # The validator never mutates its inputs.
    assert claims == before


# --------------------------------------------------------------------------- #
# Property-based test (Property 19)
# --------------------------------------------------------------------------- #
# Where the rule under test lives. ``RunRecord.success`` is only the flag the orchestrator
# *records* for a run (see ``_finish``: it is passed in, not derived). The decision of
# whether a run is *scored* as a success is made in the evaluation layer, by
# ``jobrec_eval.metrics.MetricsComputer._task_success``, surfaced as the ``task_success``
# column of ``run_metrics`` alongside ``success_run`` (the recorded flag). So the property
# drives the real MetricsComputer over real run bundles instead of restating the rule.

#: Scenario expectations whose task-success verdict follows from the recorded response
#: alone. A recommendation-expected scenario additionally needs the authoritative
#: reference context so HCSR can be recomputed, which the evaluation tests cover.
_SCORED_EXPECTATIONS = ("no_match", "clarification")

#: The slot a generated clarification asks about; also the scenario's acceptable slot.
_CLARIFICATION_SLOT = "preferred_locations"

#: How one generated handoff step ended. ``completed`` and ``recovered_then_completed``
#: are fully validated -- a ``recovered`` attempt is one whose fault the retry absorbed
#: (R10.4), recorded exactly as the task 11.4 failure set records it, so recovery is not
#: mistaken for an invalid handoff. ``failed`` and ``attempted`` carry
#: ``validation_passed=False``: one rejected outright, one left hanging without completing.
_HANDOFF_STEPS: dict[str, tuple[dict[str, Any], ...]] = {
    "completed": ({"validation_passed": True, "status": "completed"},),
    "recovered_then_completed": (
        {
            "validation_passed": True,
            "status": "recovered",
            "error_code": ErrorCode.MODEL_TIMEOUT.value,
        },
        {"validation_passed": True, "status": "completed"},
    ),
    "failed": (
        {
            "validation_passed": False,
            "status": "failed",
            "error_code": ErrorCode.HANDOFF_VALIDATION_FAILED.value,
        },
    ),
    "attempted": ({"validation_passed": False, "status": "attempted"},),
}

#: The kinds that leave every handoff of the run validated.
_VALID_HANDOFF_KINDS = ("completed", "recovered_then_completed")


def _prop19_scenario(expectation: str) -> Scenario:
    """A real ``Scenario`` expecting ``expectation``, as the scenario loader would build it."""
    return Scenario(
        scenario_id=f"sc-prop-19-{expectation}",
        scenario_type="failure_path",
        difficulty="medium",
        memory_dependency="none",
        context_dependency="low",
        no_match_expected=expectation == "no_match",
        clarification_expected=expectation == "clarification",
        acceptable_slots=[_CLARIFICATION_SLOT],
        expected_response=expectation,
    )


@pytest.fixture(scope="module")
def scorer() -> MetricsComputer:
    """The real evaluation-layer scorer, over the two scenarios the property generates.

    Catalog, reference and label inputs are empty on purpose: these scenarios return no
    jobs, so no ranking or HCSR recomputation is reached and the verdict comes from the
    real ``_task_success`` branch for the scenario's expectation.
    """
    config = load_config("configs/experiment_full.yaml", base_dir="configs")
    labels = pd.DataFrame(columns=["scenario_id", "job_id", "relevance_grade"])
    scenarios = {s.scenario_id: s for s in (_prop19_scenario(e) for e in _SCORED_EXPECTATIONS)}
    return MetricsComputer(config, [], {}, labels, scenarios)


def _generated_handoffs(kinds: list[str], run_id: str) -> list[AgentHandoff]:
    """Real ``AgentHandoff`` records for the generated mix, in attempt order."""
    handoffs: list[AgentHandoff] = []
    for kind in kinds:
        for step in _HANDOFF_STEPS[kind]:
            overrides = dict(step)
            if overrides["status"] == "completed":
                overrides["completed_at"] = utcnow()
            handoffs.append(
                AgentHandoff(
                    **valid_handoff_payload(
                        handoff_id=f"ho-prop-19-{len(handoffs)}", run_id=run_id, **overrides
                    )
                )
            )
    return handoffs


def _bundle(
    expectation: str,
    handoffs: list[AgentHandoff],
    run_record: RunRecord,
    response: Response,
    *,
    decision: RecommendationDecision | None = None,
    clarification: ClarificationAction | None = None,
) -> RunBundle:
    """A real ``RunBundle`` holding the JSON shapes ``load_bundles`` reads from disk."""
    return RunBundle(
        variant="full",
        scenario_id=f"sc-prop-19-{expectation}",
        run_index=0,
        path=Path("."),
        run_record=run_record.model_dump(mode="json"),
        decision=decision.model_dump(mode="json") if decision is not None else None,
        response=response.model_dump(mode="json"),
        claims=[],
        handoffs=[h.model_dump(mode="json") for h in handoffs],
        evidence_log=[],
        latency={},
        active_search=None,
        job_context=None,
        clarification=(
            clarification.model_dump(mode="json") if clarification is not None else None
        ),
    )


def _validated_run_bundle(expectation: str, handoffs: list[AgentHandoff]) -> RunBundle:
    """A completed run whose every handoff validated, in the shape its expectation wants."""
    active = make_active()
    run_id = "run-prop-19-validated"
    decision = clarification = None
    if expectation == "no_match":
        decision = _decision(active, no_match=True, reason_codes=("salary_min",))
        response = Response(
            response_id="resp-prop-19-no-match",
            session_id=active.session_id,
            response_type=ResponseType.NO_MATCH.value,
            message="No active posting clears your stated salary floor.",
            created_at=utcnow(),
        )
    else:
        clarification = ClarificationAction(
            clarification_id="clar-prop-19",
            target_fields=[_CLARIFICATION_SLOT],
            reason_code="missing_hard_constraint",
            question_text="Which locations would you consider?",
            created_at=utcnow(),
        )
        response = Response(
            response_id="resp-prop-19-clarification",
            session_id=active.session_id,
            response_type=ResponseType.CLARIFICATION.value,
            message=clarification.question_text,
            created_at=utcnow(),
        )
    run_record = RunRecord(
        run_id=run_id,
        scenario_id=f"sc-prop-19-{expectation}",
        session_id=active.session_id,
        candidate_id=active.candidate_id,
        experiment_variant="full",
        workflow_states=[WorkflowState.COMPLETED.value],
        handoff_ids=[h.handoff_id for h in handoffs],
        final_response_id=response.response_id,
        started_at=utcnow(),
        completed_at=utcnow(),
        success=True,
        failure_code=None,
        config_hash="cfg-hash",
        catalog_hash="cat-hash",
        prompt_hash="prompt-hash",
        code_version="test",
    )
    return _bundle(
        expectation, handoffs, run_record, response,
        decision=decision, clarification=clarification,
    )


def _failed_run_bundle(
    exploded_turn: TurnResult, expectation: str, handoffs: list[AgentHandoff]
) -> RunBundle:
    """A run with a non-validated handoff, using the real orchestrator's failure artifacts.

    The run record and response come verbatim from the exploded run above -- the failed
    handoff, ``success=False``, the failure code and the error response are what the real
    orchestrator wrote -- with only the handoff list swapped for the generated mix.
    """
    return _bundle(
        expectation,
        handoffs,
        exploded_turn.run_record.model_copy(
            update={"handoff_ids": [h.handoff_id for h in handoffs]}
        ),
        exploded_turn.response,
    )


# Feature: cmjcc-experiment-readiness, Property 19: Invalid handoffs prevent a run from
# being scored as success
@settings(max_examples=100)
@given(
    kinds=st.lists(st.sampled_from(tuple(_HANDOFF_STEPS)), min_size=1, max_size=5),
    expectation=st.sampled_from(_SCORED_EXPECTATIONS),
)
def test_property_invalid_handoffs_are_never_scored_as_a_success(
    scorer: MetricsComputer,
    exploded_turn: TurnResult,
    kinds: list[str],
    expectation: str,
) -> None:
    """A run holding any non-validated handoff is never scored a task success.

    Handoff mixes are generated freely over the four shapes a run can record: a completed
    attempt, an attempt whose fault a retry absorbed (``recovered`` plus its completed
    retry, so a recovery is not mistaken for an invalid handoff), a rejected attempt, and
    an attempt left hanging without ever completing. The verdict is produced by the real
    evaluation-layer scorer, ``MetricsComputer.run_metrics``, which is where a run is
    scored as a success; ``success_run`` is the ``RunRecord.success`` flag it reports
    beside it.

    The invalid-handoff runs are scored over the artifacts the real orchestrator writes
    for exactly that situation (the exploded run's own record and error response), so
    pairing an invalid handoff with a failed run is the system's behaviour rather than
    this test's assumption. Both directions are asserted: such a run scores 0 whichever
    outcome its scenario expected, while a run whose handoffs all validated scores 1 --
    so the zeros are a verdict and not a constant.

    **Validates: Requirements 10.7**
    """
    # The failure shape used below is the orchestrator's own: a run whose handoff did not
    # validate is recorded as a failed run carrying an error response.
    assert not all(h.validation_passed for h in exploded_turn.handoffs)
    assert exploded_turn.run_record.success is False
    assert exploded_turn.response.response_type == ResponseType.ERROR.value

    handoffs = _generated_handoffs(kinds, "run-prop-19")
    invalid = [h for h in handoffs if not h.validation_passed]
    # The generated labels agree with the real models' recorded validation state.
    assert bool(invalid) == any(kind not in _VALID_HANDOFF_KINDS for kind in kinds)

    bundle = (
        _failed_run_bundle(exploded_turn, expectation, handoffs)
        if invalid
        else _validated_run_bundle(expectation, handoffs)
    )
    row = scorer.run_metrics([bundle]).iloc[0]

    if invalid:
        # R10.7: the affected run is not counted as a success, at either layer.
        assert row["task_success"] == 0
        assert not row["success_run"]
        assert row["response_type"] == ResponseType.ERROR.value
    else:
        # Converse: the same scorer does return a success once every handoff validated.
        assert row["task_success"] == 1
        assert row["success_run"]
