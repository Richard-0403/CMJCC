"""Explanation-grounding test suite (R22): the supported side of claim grounding.

Task 21.5. Requirement 22 asks for a suite covering *supported*, *unsupported* and
*dropped* claims. The unsupported/dropped half is already exercised in depth by
``tests/unit/test_failure_paths.py`` (dangling evidence ids, missing sources, wrong-field
references, partially grounded claims, unsupported salary/location/skill claims, the
agent excluding ungrounded claims, ungrounded no-match reasons, plus Property 18 on the
validator's partition). This module deliberately does not repeat those negatives.

What it adds is the positive half and the claim lifecycle, driven through the real
pipeline (``JobContextAgent`` -> ``RankingAgent`` -> ``ExplanationAgent``) rather than
hand-built claim objects, so the evidence a claim cites is the evidence the system
actually registered:

* a properly grounded claim **is** produced and delivered for every claim family the
  agent generates -- ``candidate_preference``, ``ranking_reason``, ``skill_gap`` and
  ``no_match_reason`` -- and nothing is dropped when everything is grounded;
* each delivered claim is **traceable**: its text names the value the cited evidence
  holds, and a ranking reason cites exactly the feature that justified it;
* the **supported/dropped partition is reported to the caller** by ``explain``: one
  ungrounded claim is removed from the response and handed back, while the grounded
  claims for the same job are still delivered;
* the difference between a claim being **flagged** unsupported and **dropped** from the
  response: grounding is re-decided from the store on every pass, the status the caller
  supplied is overwritten, and the caller's own claim objects are never mutated.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import pytest

from jobrec.agents.explanation_agent import ExplanationAgent, validate_claims
from jobrec.agents.job_context_agent import JobContextAgent
from jobrec.domain.constraints import EligibilityResult
from jobrec.domain.enums import (
    ConfirmationStatus,
    EvidenceSource,
    PersistenceScope,
    ResponseType,
)
from jobrec.domain.job import ActiveSearchState, JobPosting
from jobrec.domain.recommendation import RankedJob, RecommendationDecision, Response, ResponseClaim
from jobrec.evidence_store import EvidenceStore
from jobrec.ranking.scoring import RankingAgent
from jobrec.utils.time import utcnow
from tests.conftest import make_active
from tests.support.fault_injection import DANGLING_EVIDENCE_ID, make_claim

#: Candidate-side fields whose evidence the grounded fixture registers, with the values
#: ``make_active`` states for them. The agent's summary claims and the ranking features
#: both read their candidate evidence from this map.
CANDIDATE_EVIDENCE: dict[str, Any] = {
    "target_roles": ["data analyst"],
    "preferred_locations": ["Kuala Lumpur"],
    "salary_min": 4000.0,
    "salary_currency": "MYR",
    "work_modes": ["hybrid"],
    "skills_have": ["python", "sql"],
}

#: The claim families the agent generates for a recommendation, in generation order.
RECOMMENDATION_CLAIM_FAMILIES = ("candidate_preference", "ranking_reason", "skill_gap")

#: A required skill the candidate does not have, so exactly one skill gap is reported.
MISSING_SKILL = "kubernetes"


# --------------------------------------------------------------------------- #
# fixtures / helpers
# --------------------------------------------------------------------------- #
def _register(
    store: EvidenceStore,
    field_name: str,
    value: Any,
    *,
    object_id: str = "cand-grounding",
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


def _job() -> JobPosting:
    """A posting that satisfies the active search on role, location, mode and salary.

    ``required_skills`` deliberately holds one skill the candidate has and one it does
    not: the coverage feature therefore stays below the agent's presentation threshold
    (so it does not crowd the four strong features out of the response) while producing
    exactly one skill gap to explain.
    """
    return JobPosting(
        job_id="job-grounded",
        title="Data Analyst",
        company="ACME Analytics",
        description="Data analyst role",
        normalized_title="data analyst",
        role_family="data analyst",
        required_skills=["python", MISSING_SKILL],
        preferred_skills=[],
        city="Kuala Lumpur",
        country="Malaysia",
        work_mode="hybrid",
        salary_min_monthly_myr=5000.0,
        salary_max_monthly_myr=7000.0,
        salary_currency="MYR",
        min_years_experience=0.0,
        max_years_experience=3.0,
        experience_level="junior",
        source_snapshot_id="snap",
        ingested_at=datetime(2025, 1, 1, tzinfo=UTC),
        raw_payload_hash="hash-job-grounded",
    )


def _evidence_map(
    store: EvidenceStore, *, dangling_field: str | None = None
) -> dict[str, list[str]]:
    """Register the candidate-side evidence, optionally leaving one field ungrounded."""
    evidence: dict[str, list[str]] = {}
    for field_name, value in CANDIDATE_EVIDENCE.items():
        if field_name == dangling_field:
            evidence[field_name] = [DANGLING_EVIDENCE_ID]
        else:
            evidence[field_name] = [_register(store, field_name, value)]
    return evidence


@dataclass(frozen=True)
class ExplainedTurn:
    """Everything one explanation pass produced, for assertions across the artifacts."""

    store: EvidenceStore
    active: ActiveSearchState
    job: JobPosting
    ranked: list[RankedJob]
    eligibility: list[EligibilityResult]
    response: Response
    dropped: list[ResponseClaim]

    @property
    def ranked_job(self) -> RankedJob:
        return self.ranked[0]

    def claims_of(self, claim_type: str) -> list[ResponseClaim]:
        return [c for c in self.response.claims if c.claim_type == claim_type]


def _explain(config, *, dangling_field: str | None = None) -> ExplainedTurn:
    """Run the real constraint / ranking / explanation path over one compatible job.

    The ranking agent registers the job-side evidence into the same store the
    explanation agent validates against, exactly as the orchestrator wires them, so the
    grounding decision is made over real registered evidence.
    """
    store = EvidenceStore()
    active = make_active(field_evidence_map=_evidence_map(store, dangling_field=dangling_field))
    job = _job()
    jobs_by_id = {job.job_id: job}

    context_agent = JobContextAgent(config)
    context = context_agent.build_context(active, "snap")
    eligibility = [context_agent.evaluate(job, context)]
    assert eligibility[0].eligible, "the fixture job must survive the hard constraints"

    ranked = RankingAgent(store, config).rank(active, jobs_by_id, eligibility)
    assert ranked, "the eligible job must be ranked"

    decision = RecommendationDecision(
        decision_id="dec-explanation-grounding",
        session_id=active.session_id,
        active_search_id=active.active_search_id,
        context_id=context.context_id,
        experiment_variant="full",
        retrieved_job_ids=[job.job_id],
        eligibility_results=eligibility,
        ranked_jobs=ranked,
        selected_job_ids=[rj.job_id for rj in ranked],
        no_match=False,
        no_match_reason_codes=[],
        created_at=utcnow(),
        scorer_version="test",
        config_hash="cfg-hash",
    )
    response, dropped = ExplanationAgent(store, config).explain(decision, active, jobs_by_id)
    return ExplainedTurn(
        store=store,
        active=active,
        job=job,
        ranked=ranked,
        eligibility=eligibility,
        response=response,
        dropped=dropped,
    )


def _no_match_decision(active: ActiveSearchState, reason_codes: tuple[str, ...]):
    return RecommendationDecision(
        decision_id="dec-explanation-grounding-no-match",
        session_id=active.session_id,
        active_search_id=active.active_search_id,
        context_id=None,
        experiment_variant="full",
        retrieved_job_ids=[],
        eligibility_results=[],
        ranked_jobs=[],
        selected_job_ids=[],
        no_match=True,
        no_match_reason_codes=list(reason_codes),
        created_at=utcnow(),
        scorer_version="test",
        config_hash="cfg-hash",
    )


@pytest.fixture()
def grounded(config) -> ExplainedTurn:
    """One explanation pass in which every claim's evidence is registered."""
    return _explain(config)


# --------------------------------------------------------------------------- #
# R22.1 -- supported claims are produced and delivered
# --------------------------------------------------------------------------- #
def test_fully_grounded_recommendation_delivers_every_claim_and_drops_none(
    grounded: ExplainedTurn,
) -> None:
    """With all evidence registered, every generated claim is delivered as supported.

    The converse of the failure-path suite: nothing is dropped, so the drops observed
    there are a grounding verdict rather than the agent's default behaviour.
    """
    assert grounded.dropped == []
    assert grounded.response.response_type == ResponseType.RECOMMENDATION.value
    assert grounded.response.claims, "a recommendation must carry grounded claims"

    for claim in grounded.response.claims:
        assert claim.support_status == "supported"
        assert claim.evidence_ids, "a delivered claim must cite evidence"
        assert all(grounded.store.exists(e) for e in claim.evidence_ids)

    # Claims are delivered in generation order: the need summary, then this job's
    # ranking reasons, then its skill gaps.
    families = [c.claim_type for c in grounded.response.claims]
    assert sorted(set(families)) == sorted(RECOMMENDATION_CLAIM_FAMILIES)
    assert families == sorted(families, key=RECOMMENDATION_CLAIM_FAMILIES.index)
    assert Counter(families)["skill_gap"] == 1


@pytest.mark.parametrize("claim_type", RECOMMENDATION_CLAIM_FAMILIES)
def test_every_recommendation_claim_family_has_a_supported_claim(
    grounded: ExplainedTurn, claim_type: str
) -> None:
    """Each family the agent can generate really is represented in the response (R22.1)."""
    claims = grounded.claims_of(claim_type)

    assert claims, f"no supported {claim_type} claim was delivered"
    assert {c.support_status for c in claims} == {"supported"}


def test_candidate_preference_claims_trace_to_the_value_their_evidence_holds(
    grounded: ExplainedTurn,
) -> None:
    """A preference claim names the field and the value its cited evidence recorded."""
    claims = grounded.claims_of("candidate_preference")
    assert len(claims) == 3, "the summary covers roles, locations and salary floor"

    for claim in claims:
        items = [grounded.store.get(e) for e in claim.evidence_ids]
        assert all(item is not None for item in items)
        for item in items:
            assert item.source == EvidenceSource.DIALOGUE
            # The claim text repeats the field and the value the evidence holds, so the
            # sentence can be checked against its source without re-deriving anything.
            assert item.field_name.replace("_", " ") in claim.text
            assert str(item.normalized_value) in claim.text
            assert item.normalized_value == getattr(grounded.active, item.field_name)


def test_ranking_reason_claims_trace_to_the_feature_that_justified_them(
    grounded: ExplainedTurn,
) -> None:
    """Every ranking reason cites exactly one strong feature's evidence set."""
    claims = grounded.claims_of("ranking_reason")
    assert claims, "a scored job must be explained by its features"

    strong = {
        f.explanation_code: f
        for f in grounded.ranked_job.features
        if f.normalized_score >= 0.6 and f.weight > 0
    }
    # The fixture is built so role, location, work mode and salary are all strong.
    assert {"role_exact", "location_match", "work_mode_match", "salary_meets_min"} <= set(strong)

    matched_codes = []
    for claim in claims:
        codes = [
            code
            for code, feature in strong.items()
            if feature.evidence_ids and claim.evidence_ids == feature.evidence_ids
        ]
        assert len(codes) == 1, f"claim {claim.text!r} does not trace to one strong feature"
        matched_codes.append(codes[0])
        # Both sides of the reason are cited: what the candidate said and what the
        # posting states.
        sources = {grounded.store.get(e).source for e in claim.evidence_ids}
        assert EvidenceSource.JOB_POSTING in sources
        assert EvidenceSource.DIALOGUE in sources
        assert all(
            grounded.store.get(e).source_object_id in {grounded.job.job_id, "cand-grounding"}
            for e in claim.evidence_ids
        )

    assert len(set(matched_codes)) == len(matched_codes), "two claims reused one feature"


def test_skill_gap_claim_traces_to_the_required_skills_evidence(grounded: ExplainedTurn) -> None:
    """The gap sentence names a skill the cited job evidence actually requires."""
    (claim,) = grounded.claims_of("skill_gap")

    assert MISSING_SKILL in claim.text
    items = [grounded.store.get(e) for e in claim.evidence_ids]
    assert items and all(item is not None for item in items)
    for item in items:
        assert item.source == EvidenceSource.JOB_POSTING
        assert item.source_object_id == grounded.job.job_id
        assert item.field_name == "required_skills"
        assert MISSING_SKILL in item.normalized_value
    assert MISSING_SKILL in grounded.ranked_job.skill_gaps


def test_delivered_prose_is_backed_by_the_delivered_claims(grounded: ExplainedTurn) -> None:
    """Each job-level sentence in the message is one of the supported claims (R22.1)."""
    job_claims = grounded.claims_of("ranking_reason") + grounded.claims_of("skill_gap")

    for claim in job_claims:
        assert claim.text in grounded.response.message

    # Every bulleted sentence about the job is a claim, so no factual line is unbacked.
    bullets = [
        line.strip()[2:]
        for line in grounded.response.message.splitlines()
        if line.strip().startswith("- ")
    ]
    assert bullets
    assert set(bullets) == {c.text for c in job_claims}


def test_fully_grounded_no_match_reasons_are_all_delivered(config) -> None:
    """A no-match response delivers a grounded reason for every hard constraint (R22.1)."""
    store = EvidenceStore()
    active = make_active(field_evidence_map=_evidence_map(store))
    decision = _no_match_decision(active, ("salary_min:salary_below_min",))

    response, dropped = ExplanationAgent(store, config).explain(decision, active, {})

    assert dropped == []
    assert response.response_type == ResponseType.NO_MATCH.value
    assert [c.claim_type for c in response.claims] == ["no_match_reason"] * len(
        active.hard_constraint_fields
    )
    for field_name, claim in zip(active.hard_constraint_fields, response.claims, strict=True):
        assert field_name.replace("_", " ") in claim.text
        assert claim.support_status == "supported"
        assert claim.evidence_ids == active.field_evidence_map[field_name]
        assert all(store.exists(e) for e in claim.evidence_ids)
    assert "salary_min:salary_below_min" in response.message


# --------------------------------------------------------------------------- #
# R22.1 -- the supported / dropped partition reported to the caller
# --------------------------------------------------------------------------- #
def test_one_ungrounded_claim_is_handed_back_and_the_rest_are_still_delivered(
    config, grounded: ExplainedTurn
) -> None:
    """Grounding is per claim: the work-mode reason is dropped, its siblings survive.

    The caller therefore receives a partition of the generated claims -- the delivered
    ones on the response, the rejected one returned alongside it -- and a single
    ungrounded reference never suppresses the rest of the explanation.
    """
    partial = _explain(config, dangling_field="work_modes")

    (dropped_claim,) = partial.dropped
    assert dropped_claim.claim_type == "ranking_reason"
    assert "Work mode" in dropped_claim.text
    # Flagged, recorded and handed back -- but not delivered.
    assert dropped_claim.support_status == "unsupported"
    assert DANGLING_EVIDENCE_ID in dropped_claim.evidence_ids
    assert dropped_claim.claim_id not in {c.claim_id for c in partial.response.claims}
    assert dropped_claim.text not in {c.text for c in partial.response.claims}

    # The partition is exhaustive against the fully grounded run: exactly that one claim
    # moved from the response to the dropped list.
    delivered = {c.claim_id for c in partial.response.claims}
    all_grounded = {c.claim_id for c in grounded.response.claims}
    assert delivered | {dropped_claim.claim_id} == all_grounded
    assert len(partial.response.claims) + len(partial.dropped) == len(grounded.response.claims)
    assert {c.support_status for c in partial.response.claims} == {"supported"}
    assert all(partial.store.exists(e) for c in partial.response.claims for e in c.evidence_ids)


def test_grounding_is_re_decided_and_the_callers_claims_are_left_untouched(store) -> None:
    """The status the caller supplied is never trusted: it is recomputed from the store.

    A grounded claim arriving as ``unknown`` or ``unsupported`` is delivered as
    supported, so the flag on a delivered claim always reflects this pass, and the
    caller's own objects keep the status they came in with.
    """
    evidence_id = _register(store, "work_modes", ["hybrid"])
    stale = [
        make_claim(text="You asked for a hybrid role.", evidence_ids=[evidence_id]).model_copy(
            update={"support_status": status}
        )
        for status in ("unknown", "unsupported")
    ]

    supported, dropped = validate_claims(stale, store)

    assert dropped == []
    assert [c.support_status for c in supported] == ["supported", "supported"]
    assert [c.claim_id for c in supported] == [c.claim_id for c in stale]
    # The validator returns copies; the caller's claims are unchanged.
    assert [c.support_status for c in stale] == ["unknown", "unsupported"]
