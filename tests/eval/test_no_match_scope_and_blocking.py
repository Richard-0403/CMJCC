"""P0-5: a no-match explanation must describe the set that was actually searched.

The claim that was wrong
------------------------
The opening line said no job satisfies all the hard requirements at once, which asserts
JOINT INFEASIBILITY over the catalogue. For SC-E-02 and SC-E-04 that is false and the
project's own data-quality check says so: 4 and 1 catalogue jobs respectively DO clear those
scenarios' hard constraints. They are simply outside the requested role families, so the
true statement is about the search scope -- target roles, then the retrieved set -- and not
about the catalogue.

Three further defects in the same response
------------------------------------------
* Every hard field was named as a blocking reason, including fields that rejected nothing.
  A constraint that filtered no job is not a reason the result was empty.
* Per-field counts were phrased as removals ("removed 7 of 18"). Those sets OVERLAP, since a
  job usually fails several conditions at once, so the numbers are not independent removals
  and do not sum to the total. The phrasing invited exactly that arithmetic.
* The closing line always offered to relax a soft or unconfirmed preference, even when the
  candidate had stated none -- advice that cannot be acted on.

And the retrieval record
------------------------
On an empty lexical recall the pipeline substitutes the whole catalogue as the pool, but
``retrieved_job_ids`` still reported what recall had returned: nothing. A no-match built on
that reads as "nothing was retrieved" when the truth is "everything was retrieved and
everything failed" -- opposite diagnoses from the same bundle.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobrec.app_service import AppService
from jobrec.config import load_config

CATALOG = "data/processed/jobs.jsonl"
CONFIG = "configs/experiment_full.yaml"
SCENARIOS = "evaluation/data/scenarios.jsonl"

#: The two authoritative scenarios whose no-match rests on ROLE SCOPE, not on the hard
#: constraints being jointly unsatisfiable.
ROLE_SCOPE_NO_MATCH = ("SC-E-02", "SC-E-04")


@pytest.fixture(scope="module")
def config():
    return load_config(CONFIG, base_dir="configs")


@pytest.fixture(scope="module")
def scenarios() -> dict:
    rows = [json.loads(line) for line
            in Path(SCENARIOS).read_text(encoding="utf-8").splitlines() if line.strip()]
    return {s["scenario_id"]: s for s in rows}


def _run(config, scenario: dict):
    service = AppService(config, CATALOG)
    candidate = service.create_candidate(scenario["profile"])
    session = service.create_session(candidate.candidate_id, "full")
    result = None
    for text in scenario["turns"]:
        result = service.process_turn(session, text,
                                      scenario_id=scenario["scenario_id"])
    return result


# ------------------------------------------------------- SC-E-02 / SC-E-04 wording
@pytest.mark.parametrize("scenario_id", ROLE_SCOPE_NO_MATCH)
def test_the_no_match_message_is_scoped_not_global(config, scenarios, scenario_id):
    """The response must not claim the hard constraints are jointly unsatisfiable."""
    result = _run(config, scenarios[scenario_id])
    assert result.decision is not None and result.decision.no_match, (
        f"{scenario_id} is declared no-match and must still produce one")

    message = result.response.message
    assert "search scope" in message, message
    # The exact wording that asserted more than the evidence supports.
    assert "No jobs currently satisfy all of your hard requirements at once." not in message


@pytest.mark.parametrize("scenario_id", ROLE_SCOPE_NO_MATCH)
def test_no_delivered_claim_asserts_global_infeasibility(config, scenarios, scenario_id):
    """Every surviving claim is about the searched set, and each one is grounded."""
    result = _run(config, scenarios[scenario_id])
    for claim in result.response.claims:
        assert claim.support_status == "supported", claim
        assert claim.predicate, f"a delivered claim states no proposition: {claim}"


# -------------------------------------------------- only real blocking fields
def test_a_hard_field_that_blocked_nothing_is_not_named_as_a_reason(config, scenarios):
    """Reasons come from the diagnosis, not from the list of hard constraints."""
    for scenario_id in ROLE_SCOPE_NO_MATCH:
        result = _run(config, scenarios[scenario_id])
        diagnosis = result.decision.no_match_diagnosis or {}
        blocking = {b["field"] for b in diagnosis.get("blocking_constraints", [])}
        claimed = {c.field_name for c in result.response.claims
                   if c.claim_type in ("no_match_reason", "no_match_cause")}
        assert claimed <= blocking, (
            f"{scenario_id} named non-blocking fields: {claimed - blocking}")


def test_a_non_blocking_hard_field_is_excluded(config):
    """The filter itself, on a constructed case.

    Neither SC-E-02 nor SC-E-04 happens to have a hard field that blocked nothing, so the
    exclusion is exercised directly rather than left to chance.
    """
    from jobrec.agents.explanation_agent import ExplanationAgent
    from jobrec.domain.recommendation import RecommendationDecision
    from jobrec.evidence_store import EvidenceStore

    store = EvidenceStore()
    agent = ExplanationAgent(store, config)

    class _Active:
        candidate_id = "cand-1"
        hard_constraint_fields = ["salary_min", "work_modes"]
        salary_min = 4000
        work_modes = ["onsite"]
        field_evidence_map = {"salary_min": ["ev-a"], "work_modes": ["ev-b"]}

    from jobrec.utils.time import utcnow

    decision = RecommendationDecision(
        decision_id="dec-1", session_id="s", active_search_id="a",
        experiment_variant="full", no_match=True, no_match_reason_codes=[],
        created_at=utcnow(), scorer_version="test", config_hash="cfg", context_id=None,
        # Only salary blocked anything; work_modes rejected nothing.
        no_match_diagnosis={"evaluated_jobs": 3, "eligible_jobs": 0,
                            "blocking_constraints": [
                                {"field": "salary_min", "filtered_jobs": 3,
                                 "blocked_job_ids": ["j1", "j2", "j3"]}],
                            "relaxation_candidates": [], "stage_trace": []},
    )
    response, dropped = agent._no_match(decision, _Active())

    # The reasons are the CLAIMS; the message carries the scope statement.
    named = {c.field_name for c in [*response.claims, *dropped]}
    assert "salary_min" in named, named
    assert "work_modes" not in named, (
        "a hard field that blocked nothing was still offered as a reason")
    assert "search scope" in response.message


# ------------------------------------------------ per-stage and per-field job ids
def test_the_diagnosis_records_which_jobs_each_stage_and_field_kept(config, scenarios):
    result = _run(config, scenarios["SC-E-02"])
    diagnosis = result.decision.no_match_diagnosis or {}

    stages = {s["stage"]: s for s in diagnosis.get("stage_trace", [])}
    for stage in ("retrieved", "eligible_after_hard_constraints", "ranked"):
        assert "job_ids" in stages[stage], f"{stage} recorded no ids"
        assert len(stages[stage]["job_ids"]) == stages[stage]["jobs"], stage

    assert diagnosis["evaluated_job_ids"]
    assert len(diagnosis["evaluated_job_ids"]) == diagnosis["evaluated_jobs"]
    for record in diagnosis["blocking_constraints"]:
        assert len(record["blocked_job_ids"]) == record["filtered_jobs"], record
        # Every blocked job was one that was evaluated.
        assert set(record["blocked_job_ids"]) <= set(diagnosis["evaluated_job_ids"])


def test_the_per_field_counts_overlap_and_are_not_presented_as_removals(config, scenarios):
    """The sets intersect, which is precisely why "removed N" was the wrong phrasing."""
    result = _run(config, scenarios["SC-E-02"])
    diagnosis = result.decision.no_match_diagnosis or {}
    records = diagnosis["blocking_constraints"]
    total = sum(r["filtered_jobs"] for r in records)
    union = set().union(*[set(r["blocked_job_ids"]) for r in records]) if records else set()

    if total > len(union):
        # Overlapping, so the counts cannot be read as independent removals.
        for claim in result.response.claims:
            if claim.claim_type == "no_match_cause":
                assert "did not meet" in claim.text, claim.text
                assert "removed" not in claim.text, claim.text


# ------------------------------------------------------- relaxation advice
def test_relaxation_is_only_offered_when_something_is_actually_relaxable(config,
                                                                        scenarios):
    for scenario_id in ROLE_SCOPE_NO_MATCH:
        result = _run(config, scenarios[scenario_id])
        diagnosis = result.decision.no_match_diagnosis or {}
        offered = "You could relax" in result.response.message
        assert offered == bool(diagnosis.get("relaxation_candidates")), (
            f"{scenario_id}: relaxation advice does not match the diagnosis")


@pytest.mark.parametrize("scenario_id", ROLE_SCOPE_NO_MATCH)
def test_the_response_does_not_guess_about_jobs_it_never_evaluated(config, scenarios,
                                                                  scenario_id):
    """It is TRUE that jobs outside the target roles clear these hard constraints -- 4 for
    SC-E-02 and 1 for SC-E-04 -- and the response still must not say so.

    The diagnosis describes the RETRIEVED pool, which is already role-scoped, so those jobs
    were never evaluated and ``eligible_jobs`` is 0 here. A response asserting their
    existence would be right by luck, from a layer that cannot see them. Pinned as a test
    because the tempting fix is to add the sentence anyway.
    """
    result = _run(config, scenarios[scenario_id])
    diagnosis = result.decision.no_match_diagnosis or {}

    assert diagnosis.get("eligible_jobs") == 0, (
        "eligible jobs appeared inside the scope; revisit whether the response may now "
        "describe them")
    assert "outside your target roles" not in result.response.message
    # What it may say: nothing in the searched set qualified.
    assert "search scope" in result.response.message


# ------------------------------------------------------- the fallback retrieval record
def test_an_empty_recall_fallback_records_what_was_really_evaluated(config):
    """``retrieved_job_ids`` must be the evaluated set, not the empty recall result."""
    service = AppService(config, CATALOG)
    candidate = service.create_candidate(
        {"candidate_id": "fallback-cand", "skills": [], "years_experience": 2})
    session = service.create_session(candidate.candidate_id, "full")
    # Deliberately unmatchable text, so lexical recall returns nothing and the pipeline
    # substitutes the whole catalogue.
    result = service.process_turn(
        session, "zzqqxx unmatchable gibberish token", scenario_id="fallback")

    outcome = result.retrieval_outcome
    if not getattr(outcome, "expanded", False):
        pytest.skip("lexical recall was not empty, so the fallback did not fire")
    assert result.decision is not None
    assert result.decision.retrieved_job_ids, (
        "the fallback evaluated the whole catalogue but recorded no retrieved jobs")
    assert len(result.decision.retrieved_job_ids) == len(
        result.decision.eligibility_results)
