"""A no-match explanation must be recomputable from a recorded stage trace.

The diagnosis used to record only which fields blocked jobs INSIDE the retrieval pool.
That cannot distinguish "these requirements are jointly unsatisfiable" from "nothing in
the requested role family was retrieved at all" -- and the response asserted the first.
SC-E-02 and SC-E-04 are exactly the second case: catalogue jobs do clear their hard
constraints, they are simply outside the requested roles, which is what the data-quality
warning on those two scenarios says.

So the causal claim is now separate from the descriptive one, and is only permitted to cite
a record of what a filter actually removed.
"""

from __future__ import annotations

from jobrec.agents.explanation_agent import (
    CAUSAL_EFFECT_KEY,
    _records_a_filtering_effect,
    semantic_status,
)
from jobrec.agents.job_context_agent import diagnose_no_match
from jobrec.domain.constraints import ConstraintDefinition, EligibilityResult
from jobrec.domain.enums import (
    ConfirmationStatus,
    ConstraintStrength,
    EvidenceSource,
    PersistenceScope,
    UnknownPolicy,
)
from jobrec.domain.recommendation import ResponseClaim
from jobrec.evidence_store import EvidenceStore


def _blocked(job_id: str, field: str) -> EligibilityResult:
    return EligibilityResult(
        eligibility_result_id=f"elig-{job_id}", job_id=job_id, eligible=False, checks=[],
        hard_violation_count=1, unknown_hard_constraint_count=0,
        filtered_reason_codes=[f"{field}:below_min"])


def _context() -> object:
    from jobrec.domain.constraints import JobContextState
    from jobrec.utils.time import utcnow

    return JobContextState(
        context_id="ctx", active_search_id="as", catalog_snapshot_id="cat-test",
        constraints=[ConstraintDefinition(
            constraint_id="con-1", field_name="salary_min", operator="gte",
            expected_value=4000.0, strength=ConstraintStrength.HARD, weight=0.0,
            evidence_ids=[], unknown_policy=UnknownPolicy.FAIL, rule_id="rule.salary_min")],
        normalized_at=utcnow())


def test_the_diagnosis_records_every_stage() -> None:
    diag = diagnose_no_match(
        [_blocked("job-1", "salary_min"), _blocked("job-2", "salary_min")],
        _context(), catalog_size=200, pool_size=2, ranked_size=0)

    stages = {s["stage"]: s["jobs"] for s in diag["stage_trace"]}
    assert stages == {"catalog": 200, "retrieved": 2,
                      "eligible_after_hard_constraints": 0, "ranked": 0}
    assert diag["evaluated_jobs"] == 2
    assert diag["eligible_jobs"] == 0
    # And the per-field count survives, so a causal claim has a number to cite.
    assert diag["blocking_constraints"] == [
        # The blocked ids as well as the count: the count alone could not be checked, and the
        # per-field sets overlap, so they must not be read as independent removals.
        {"field": "salary_min", "filtered_jobs": 2,
         "blocked_job_ids": ["job-1", "job-2"]}]


def test_the_stage_trace_separates_an_empty_pool_from_an_infeasible_one() -> None:
    """The distinction the old diagnosis could not express.

    Both runs end with nothing ranked. Only one of them evaluated any job at all, and
    without the retrieved count an explanation cannot tell the reader which happened.
    """
    infeasible = diagnose_no_match([_blocked("job-1", "salary_min")], _context(),
                                   catalog_size=200, pool_size=1, ranked_size=0)
    nothing_retrieved = diagnose_no_match([], _context(),
                                          catalog_size=200, pool_size=0, ranked_size=0)

    assert infeasible["evaluated_jobs"] == 1
    assert nothing_retrieved["evaluated_jobs"] == 0
    assert nothing_retrieved["blocking_constraints"] == [], (
        "no constraint can be blamed when no job was evaluated")


def test_a_causal_claim_needs_a_filtering_record() -> None:
    """Constraint-existence evidence cannot carry "this is why", a stage record can."""
    store = EvidenceStore()
    stated = store.register_field(
        EvidenceSource.DIALOGUE, "cand-1", "salary_min", 4000.0,
        confidence=1.0, confirmation=ConfirmationStatus.CONFIRMED,
        scope=PersistenceScope.SESSION)
    effect = store.register_field(
        EvidenceSource.SYSTEM_RULE, "dec-1", "filtered_by:salary_min",
        {"field": "salary_min", CAUSAL_EFFECT_KEY: 2},
        confidence=1.0, confirmation=ConfirmationStatus.CONFIRMED,
        scope=PersistenceScope.SESSION)

    assert not _records_a_filtering_effect(stated)
    assert _records_a_filtering_effect(effect)

    def claim(evidence_ids):
        return ResponseClaim(claim_id="claim-1", claim_type="no_match_cause",
                             text="Applying your requirement removed 2 of the 2 evaluated.",
                             evidence_ids=evidence_ids,
                             predicate="no_match_cause", field_name="salary_min",
                             claim_args={"removed": 2, "evaluated_jobs": 2})

    assert semantic_status(claim([stated.evidence_id]), store) == "unsupported"
    assert semantic_status(
        claim([stated.evidence_id, effect.evidence_id]), store) == "supported"


def test_the_descriptive_claim_still_stands_on_the_statement_alone() -> None:
    """The non-causal form is deliberately still available.

    Requiring a stage record for it too would have left a no-match response with no reasons
    at all whenever the trace was absent -- trading an unsupported explanation for none.
    """
    store = EvidenceStore()
    stated = store.register_field(
        EvidenceSource.DIALOGUE, "cand-1", "work_modes", ["onsite"],
        confidence=1.0, confirmation=ConfirmationStatus.CONFIRMED,
        scope=PersistenceScope.SESSION)
    descriptive = ResponseClaim(
        claim_id="claim-2", claim_type="no_match_reason",
        text="Your stated requirement on work modes was applied as a hard filter.",
        evidence_ids=[stated.evidence_id],
        predicate="constraint_applied", field_name="work_modes")

    assert semantic_status(descriptive, store) == "supported"
