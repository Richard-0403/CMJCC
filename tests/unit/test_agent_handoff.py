"""Agent-handoff test suite: the contract chain between components (R23.1, R23.2).

Task 21.6. Where ``tests/unit/test_failure_paths.py`` covers the handoff *contract* in
isolation (a valid payload, a schema-invalid ``status``, an unknown extra field, every
required field omitted one at a time, and Property 19 -- an invalid handoff is never
scored a success), this suite covers the part R23.1 leaves untested: the **chain a real
run emits**.

Every test here drives a real turn through :class:`~jobrec.app_service.AppService` in
deterministic mode over the real catalog, then inspects the handoffs the orchestrator
actually recorded:

* the *valid* handoff case (R23.1) is the successful chain -- each agent boundary emits one
  handoff naming the right ``from_component``/``to_component``/``contract_name``, in
  workflow order, each ``validation_passed=True`` and ``status="completed"`` with a
  ``completed_at``, with both schema versions recorded, and with every handoff id
  referenced by the run record;
* the chain is checked on all three terminal paths a successful run can take --
  recommendation, no-match, and the clarification short-circuit, where the chain
  legitimately stops at the CMJCC boundary;
* the *schema-invalid* and *missing-field* cases (R23.1) are re-checked against the
  payloads the orchestrator really emits rather than fabricated ones: a real handoff
  round-trips through :class:`~jobrec.domain.handoff.AgentHandoff`, while the same payload
  with a mistyped or dropped chain-identity field is rejected and the offending field is
  named.
"""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from jobrec.app_service import AppService
from jobrec.config import load_config
from jobrec.domain.enums import ResponseType, WorkflowState
from jobrec.domain.handoff import AgentHandoff
from jobrec.orchestration.cmjcc import CMJCC
from jobrec.orchestration.orchestrator import TurnResult
from tests.conftest import CATALOG_PATH
from tests.support.fault_injection import REQUIRED_HANDOFF_FIELDS

#: The handoff chain a run that reaches a recommendation (or a no-match) must emit, in
#: order: ``(from_component, to_component, contract_name)``. Component names are spelled
#: out rather than read back off the orchestrator so a silently renamed or re-wired
#: boundary fails; ``test_expected_chain_names_match_the_real_components`` keeps the
#: literals honest by checking them against the real component objects.
FULL_CHAIN: tuple[tuple[str, str, str], ...] = (
    ("candidate_understanding_agent", "memory_agent", "ExtractedPreferenceSet"),
    ("memory_agent", "cmjcc", "CMJCCOutput"),
    ("hybrid_retriever", "job_context_agent", "RetrievalOutcome"),
    ("job_context_agent", "ranking_agent", "EligibilityResults"),
    ("ranking_agent", "explanation_agent", "RecommendationDecision"),
)

#: A clarification short-circuits after the CMJCC merge, so the chain stops there: the
#: retrieval, eligibility and explanation boundaries are never crossed.
CLARIFICATION_CHAIN = FULL_CHAIN[:2]

#: The schema version the orchestrator stamps on both ends of a completed handoff.
SCHEMA_VERSION = "1.0.0"

#: Utterances that steer the deterministic pipeline onto each terminal path.
RECOMMENDATION_UTTERANCE = "I want a data analyst role in Kuala Lumpur, at least RM4000."
NO_MATCH_UTTERANCE = "Only a data analyst in Kuala Lumpur paying at least RM50000 per month."
CLARIFICATION_UTTERANCE = "I am looking for something in Kuala Lumpur, hybrid, around RM5000."


def _chain(handoffs: list[AgentHandoff]) -> list[tuple[str, str, str]]:
    """The (from, to, contract) triples of ``handoffs``, in emission order."""
    return [(h.from_component, h.to_component, h.contract_name) for h in handoffs]


def _service() -> AppService:
    """An AppService on the real catalog, deterministic mode, in-memory repository."""
    config = load_config("configs/experiment_full.yaml", base_dir="configs")
    return AppService(config, CATALOG_PATH)


def _run(candidate_id: str, utterance: str, profile: dict[str, Any] | None = None) -> TurnResult:
    """One real turn: create the candidate and session, then process ``utterance``."""
    service = _service()
    service.create_candidate(
        profile
        or {
            "candidate_id": candidate_id,
            "skills": ["Python", "SQL"],
            "years_experience": 1,
            "preferred_locations": ["Kuala Lumpur"],
            "work_modes": ["hybrid"],
        }
    )
    session_id = service.create_session(candidate_id, "full")
    return service.process_turn(session_id, utterance)


@pytest.fixture(scope="module")
def recommendation_turn() -> TurnResult:
    """A successful run that ends in a recommendation: the complete handoff chain."""
    result = _run("cand-handoff-recommendation", RECOMMENDATION_UTTERANCE)
    assert result.response.response_type == ResponseType.RECOMMENDATION.value
    return result


@pytest.fixture(scope="module")
def no_match_turn() -> TurnResult:
    """A successful run that ends in a no-match: the same chain, a different outcome."""
    result = _run("cand-handoff-no-match", NO_MATCH_UTTERANCE)
    assert result.response.response_type == ResponseType.NO_MATCH.value
    return result


@pytest.fixture(scope="module")
def clarification_turn() -> TurnResult:
    """A run that short-circuits into a clarification, so the chain stops early."""
    result = _run(
        "cand-handoff-clarification",
        CLARIFICATION_UTTERANCE,
        profile={
            "candidate_id": "cand-handoff-clarification",
            "skills": ["Python"],
            "years_experience": 2,
        },
    )
    assert result.response.response_type == ResponseType.CLARIFICATION.value
    return result


# --------------------------------------------------------------------------- #
# R23.1 -- valid handoffs: the chain a successful run emits
# --------------------------------------------------------------------------- #
def test_expected_chain_names_match_the_real_components() -> None:
    """The component names asserted below are the real agents' own names.

    Keeps ``FULL_CHAIN`` from going stale: a renamed component makes the literals wrong,
    and this test says so directly instead of the chain assertions failing obscurely.
    """
    service = _service()
    session_id = service.create_session("cand-handoff-names", "full")
    orchestrator, _store = service._orchestrator_for(session_id, "full")

    real_names = {
        orchestrator.rule_extractor.name,
        orchestrator.memory.name,
        CMJCC.name,
        orchestrator.retriever.name,
        orchestrator.job_context.name,
        orchestrator.ranking.name,
        orchestrator.explainer.name,
    }
    chain_names = {name for step in FULL_CHAIN for name in step[:2]}
    assert chain_names == real_names


def test_recommendation_run_emits_every_boundary_in_workflow_order(
    recommendation_turn,
) -> None:
    """Each agent boundary emits exactly one handoff, correctly named and ordered (R23.1)."""
    assert _chain(recommendation_turn.handoffs) == list(FULL_CHAIN)
    # One handoff per boundary: no boundary is crossed twice or silently skipped.
    assert len(recommendation_turn.handoffs) == len(FULL_CHAIN)
    assert recommendation_turn.run_record.workflow_states[-1] == WorkflowState.COMPLETED.value


@pytest.mark.parametrize("turn_name", ["recommendation_turn", "no_match_turn"])
def test_every_handoff_of_a_successful_run_is_completed_and_validated(
    turn_name: str, request: pytest.FixtureRequest
) -> None:
    """A valid handoff is recorded as completed, validated, timestamped, error-free (R23.1)."""
    turn: TurnResult = request.getfixturevalue(turn_name)
    assert turn.run_record.success is True
    assert turn.run_record.failure_code is None
    assert turn.handoffs, "a successful run must record its handoffs"

    for handoff in turn.handoffs:
        assert handoff.status == "completed"
        assert handoff.validation_passed is True
        assert handoff.error_code is None
        assert handoff.completed_at is not None
        assert handoff.completed_at >= handoff.attempted_at

    # The chain is recorded in the order it was attempted.
    attempts = [h.attempted_at for h in turn.handoffs]
    assert attempts == sorted(attempts)


def test_input_and_output_schema_versions_are_recorded_on_every_handoff(
    recommendation_turn,
) -> None:
    """Both ends of each completed contract carry a schema version (R23.1)."""
    for handoff in recommendation_turn.handoffs:
        assert handoff.input_schema_version == SCHEMA_VERSION
        # A completed handoff validated its output, so the output version is recorded too.
        assert handoff.output_schema_version == SCHEMA_VERSION


def test_run_record_references_every_handoff_exactly_once(recommendation_turn) -> None:
    """The run record indexes the whole chain, in order, with distinct ids (R23.1)."""
    record = recommendation_turn.run_record
    handoff_ids = [h.handoff_id for h in recommendation_turn.handoffs]

    assert record.handoff_ids == handoff_ids
    assert len(set(handoff_ids)) == len(handoff_ids), "handoff ids must be distinct"
    assert all(h.run_id == record.run_id for h in recommendation_turn.handoffs)


def test_no_match_run_emits_the_same_chain_as_a_recommendation(no_match_turn) -> None:
    """A no-match still crosses every boundary: the outcome differs, not the chain (R23.1)."""
    assert _chain(no_match_turn.handoffs) == list(FULL_CHAIN)
    assert no_match_turn.decision is not None and no_match_turn.decision.no_match
    assert no_match_turn.run_record.workflow_states[-1] == WorkflowState.COMPLETED.value


def test_clarification_run_stops_the_chain_at_the_cmjcc_boundary(clarification_turn) -> None:
    """The short-circuit emits only the boundaries it crossed, all of them valid (R23.1)."""
    assert _chain(clarification_turn.handoffs) == list(CLARIFICATION_CHAIN)
    # The unreached boundaries are absent rather than recorded as attempted-but-unfinished.
    crossed = {h.contract_name for h in clarification_turn.handoffs}
    assert crossed.isdisjoint({"RetrievalOutcome", "EligibilityResults", "RecommendationDecision"})

    assert clarification_turn.clarification is not None
    assert clarification_turn.run_record.success is True
    assert all(
        h.status == "completed" and h.validation_passed for h in clarification_turn.handoffs
    )
    assert clarification_turn.run_record.handoff_ids == [
        h.handoff_id for h in clarification_turn.handoffs
    ]


# --------------------------------------------------------------------------- #
# R23.1 -- the same contract, checked against the payloads a real run emits
# --------------------------------------------------------------------------- #
def test_emitted_handoffs_round_trip_through_the_contract(recommendation_turn) -> None:
    """Every handoff the orchestrator emits re-validates unchanged (R23.1)."""
    for handoff in recommendation_turn.handoffs:
        assert AgentHandoff(**handoff.model_dump()) == handoff


#: (field, invalid value) pairs: a real payload mistyped on one field at a time.
_MISTYPED_FIELDS = [
    ("attempted_at", "yesterday"),
    ("completed_at", "shortly after"),
    ("validation_passed", "maybe"),
    ("contract_name", 17),
]


@pytest.mark.parametrize(("field_name", "bad_value"), _MISTYPED_FIELDS)
def test_emitted_handoff_with_a_mistyped_field_is_rejected(
    recommendation_turn, field_name: str, bad_value: Any
) -> None:
    """A schema-invalid value on an otherwise real payload is rejected by field (R23.1)."""
    payload = recommendation_turn.handoffs[0].model_dump()
    payload[field_name] = bad_value

    with pytest.raises(ValidationError) as excinfo:
        AgentHandoff(**payload)

    assert [e["loc"] for e in excinfo.value.errors()] == [(field_name,)]


@pytest.mark.parametrize("omitted", ["from_component", "to_component", "contract_name"])
def test_emitted_handoff_missing_a_chain_identity_field_is_rejected(
    recommendation_turn, omitted: str
) -> None:
    """The fields that identify a boundary are required, so a chain cannot be anonymous."""
    assert omitted in REQUIRED_HANDOFF_FIELDS
    payload = recommendation_turn.handoffs[0].model_dump()
    del payload[omitted]

    with pytest.raises(ValidationError) as excinfo:
        AgentHandoff(**payload)

    errors = excinfo.value.errors()
    assert [e["loc"] for e in errors] == [(omitted,)]
    assert errors[0]["type"] == "missing"
