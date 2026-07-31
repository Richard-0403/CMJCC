"""P0-4: a claim is checked against its PROPOSITION, not against its evidence's type.

Where this started
------------------
The original validator asked only "do the cited evidence ids resolve". For several claim
types the builder registers the evidence it is about to cite moments earlier, so resolution
was true by construction: it marked all 11197 claims of the official pair supported while
human adjudication found 2349 unsupported -- a detection rate of exactly zero.

The first repair added evidence-CLASS rules: a claim about the candidate needs candidate-side
evidence, a claim about a job needs job-side evidence, a causal claim needs a filtering
record. Better, and still not entailment. These three counterexamples all passed it:

* a ``candidate_preference`` claim about SALARY citing the candidate's LOCATION evidence,
* a ``skill_gap`` claim that Excel is not recorded, citing evidence that records Excel,
* a ``no_match_cause`` claim about WORK MODE citing the SALARY stage's filtering record.

Each cites the right KIND of evidence for its type and says something the evidence does not
establish. The gap is that nothing compared the claim's field, value or arguments with the
evidence's.

What is asserted here
---------------------
Claims now carry a structured proposition -- ``predicate`` plus its arguments -- and the
validator checks that proposition against the resolved evidence. Verification never parses
the English text: the text is a rendering of the proposition, not its definition, so a
checker that read the text would be validating the renderer.

An unknown or missing predicate returns ``unknown`` and the claim is dropped. That is
deliberately not the same as passing: a claim no checker understands must not arrive already
believed, and a builder that forgets to state its proposition must not be rewarded with a
default pass.
"""

from __future__ import annotations

import pytest

from jobrec.agents.explanation_agent import semantic_status
from jobrec.domain.enums import ConfirmationStatus, EvidenceSource, PersistenceScope
from jobrec.domain.recommendation import ResponseClaim
from jobrec.evidence_store import EvidenceStore

CAND = "cand-1"
JOB = "job-1"
OTHER_JOB = "job-2"


@pytest.fixture
def store() -> EvidenceStore:
    return EvidenceStore()


def _candidate(store: EvidenceStore, field: str, value, *, source=EvidenceSource.DIALOGUE,
               confirmation=ConfirmationStatus.CONFIRMED) -> str:
    return store.register_field(
        source, CAND, field, value, confidence=0.9, confirmation=confirmation,
        scope=PersistenceScope.SESSION, raw_text=f"{field}={value}",
    ).evidence_id


def _job(store: EvidenceStore, field: str, value, job_id: str = JOB) -> str:
    return store.register_field(
        EvidenceSource.JOB_POSTING, job_id, field, value, confidence=1.0,
        confirmation=ConfirmationStatus.CONFIRMED, scope=PersistenceScope.SESSION,
    ).evidence_id


def _stage(store: EvidenceStore, field: str, removed: int, blocked=None) -> str:
    value = {"field": field, "filtered_count": removed}
    if blocked is not None:
        value["blocked_job_ids"] = list(blocked)
    return store.register_field(
        EvidenceSource.SYSTEM_RULE, "decision-1", f"filtered_by:{field}", value,
        confidence=1.0, confirmation=ConfirmationStatus.CONFIRMED,
        scope=PersistenceScope.SESSION,
    ).evidence_id


def _claim(**kwargs) -> ResponseClaim:
    base = {"claim_id": "c1", "text": "rendered text is never parsed by the validator"}
    return ResponseClaim(**{**base, **kwargs})


# ============================================================ the counterexamples
def test_location_evidence_cannot_support_a_salary_preference(store):
    """Counterexample 1. Candidate-side evidence, wrong field."""
    claim = _claim(claim_type="candidate_preference", predicate="candidate_preference",
                   field_name="salary_min", expected_value=4000,
                   evidence_ids=[_candidate(store, "preferred_locations", "Kuala Lumpur")])
    assert semantic_status(claim, store) == "unsupported"


def test_a_recorded_skill_cannot_support_a_not_recorded_claim(store):
    """Counterexample 2. Both evidence classes present, and the value CONTRADICTS."""
    claim = _claim(claim_type="skill_gap", predicate="skill_not_recorded", job_id=JOB,
                   claim_args={"skill": "excel"},
                   evidence_ids=[_job(store, "required_skills", ["excel", "sql"]),
                                 _candidate(store, "skills_have", ["excel", "python"])])
    assert semantic_status(claim, store) == "unsupported"


def test_salary_filtering_evidence_cannot_support_a_work_mode_cause(store):
    """Counterexample 3. A real filtering record, for a DIFFERENT field."""
    claim = _claim(claim_type="no_match_cause", predicate="no_match_cause",
                   field_name="work_modes",
                   claim_args={"removed": 7, "evaluated_jobs": 18},
                   evidence_ids=[_candidate(store, "work_modes", "remote"),
                                 _stage(store, "salary_min", 7)])
    assert semantic_status(claim, store) == "unsupported"


# ================================================== 1. candidate_preference(field, value)
def test_candidate_preference_holds_when_field_and_value_agree(store):
    claim = _claim(claim_type="candidate_preference", predicate="candidate_preference",
                   field_name="salary_min", expected_value=4000,
                   evidence_ids=[_candidate(store, "salary_min", 4000)])
    assert semantic_status(claim, store) == "supported"


def test_candidate_preference_rejects_a_different_value_for_the_right_field(store):
    """The superseded-value case: citing an earlier statement of the same field.

    Checked by requiring the cited evidence to carry the value the claim asserts, so an
    overwritten or conflicting older value fails without needing a "superseded" flag.
    """
    claim = _claim(claim_type="candidate_preference", predicate="candidate_preference",
                   field_name="salary_min", expected_value=4000,
                   evidence_ids=[_candidate(store, "salary_min", 3000)])
    assert semantic_status(claim, store) == "unsupported"


def test_candidate_preference_rejects_job_side_evidence(store):
    claim = _claim(claim_type="candidate_preference", predicate="candidate_preference",
                   field_name="salary_min", expected_value=4000,
                   evidence_ids=[_job(store, "salary_min", 4000)])
    assert semantic_status(claim, store) == "unsupported"


def test_candidate_preference_rejects_unconfirmed_evidence(store):
    claim = _claim(claim_type="candidate_preference", predicate="candidate_preference",
                   field_name="salary_min", expected_value=4000,
                   evidence_ids=[_candidate(store, "salary_min", 4000,
                                            confirmation=ConfirmationStatus.UNCONFIRMED)])
    assert semantic_status(claim, store) == "unsupported"


def test_candidate_preference_accepts_one_value_of_a_stated_list(store):
    """One evidence item states one value; the claim may render the whole list."""
    claim = _claim(claim_type="candidate_preference", predicate="candidate_preference",
                   field_name="target_roles", expected_value=["data analyst", "bi analyst"],
                   evidence_ids=[_candidate(store, "target_roles", "data analyst")])
    assert semantic_status(claim, store) == "supported"


# ================================================ 2. salary_meets_min(job_id, threshold)
def test_salary_meets_min_holds_when_the_job_minimum_clears_the_threshold(store):
    claim = _claim(claim_type="ranking_reason", predicate="salary_meets_min", job_id=JOB,
                   field_name="salary_min", expected_value=4000, observed_value=4500,
                   evidence_ids=[_candidate(store, "salary_min", 4000),
                                 _job(store, "salary_min_monthly_myr", 4500)])
    assert semantic_status(claim, store) == "supported"


def test_salary_meets_min_rejects_a_job_below_the_threshold(store):
    """The guaranteed-minimum rule, enforced at the CLAIM as well as at eligibility."""
    claim = _claim(claim_type="ranking_reason", predicate="salary_meets_min", job_id=JOB,
                   field_name="salary_min", expected_value=4000, observed_value=3000,
                   evidence_ids=[_candidate(store, "salary_min", 4000),
                                 _job(store, "salary_min_monthly_myr", 3000)])
    assert semantic_status(claim, store) == "unsupported"


def test_salary_meets_min_rejects_another_jobs_salary_evidence(store):
    claim = _claim(claim_type="ranking_reason", predicate="salary_meets_min", job_id=JOB,
                   field_name="salary_min", expected_value=4000, observed_value=4500,
                   evidence_ids=[_candidate(store, "salary_min", 4000),
                                 _job(store, "salary_min_monthly_myr", 4500,
                                      job_id=OTHER_JOB)])
    assert semantic_status(claim, store) == "unsupported"


def test_salary_meets_min_needs_both_sides(store):
    """The job's salary alone cannot establish that it meets the CANDIDATE's minimum."""
    claim = _claim(claim_type="ranking_reason", predicate="salary_meets_min", job_id=JOB,
                   field_name="salary_min", expected_value=4000, observed_value=4500,
                   evidence_ids=[_job(store, "salary_min_monthly_myr", 4500)])
    assert semantic_status(claim, store) == "unsupported"


# ================================================ 3. skill_not_recorded(job_id, skill)
def test_skill_not_recorded_holds_when_the_job_requires_it_and_the_profile_lacks_it(store):
    claim = _claim(claim_type="skill_gap", predicate="skill_not_recorded", job_id=JOB,
                   claim_args={"skill": "excel"},
                   evidence_ids=[_job(store, "required_skills", ["excel", "sql"]),
                                 _candidate(store, "skills_have", ["python"])])
    assert semantic_status(claim, store) == "supported"


def test_skill_not_recorded_rejects_a_skill_the_job_does_not_require(store):
    claim = _claim(claim_type="skill_gap", predicate="skill_not_recorded", job_id=JOB,
                   claim_args={"skill": "excel"},
                   evidence_ids=[_job(store, "required_skills", ["sql"]),
                                 _candidate(store, "skills_have", ["python"])])
    assert semantic_status(claim, store) == "unsupported"


def test_skill_not_recorded_needs_the_candidate_side_too(store):
    """The original 1883-claim failure: job evidence only cannot show an absence."""
    claim = _claim(claim_type="skill_gap", predicate="skill_not_recorded", job_id=JOB,
                   claim_args={"skill": "excel"},
                   evidence_ids=[_job(store, "required_skills", ["excel"])])
    assert semantic_status(claim, store) == "unsupported"


def test_skill_not_recorded_accepts_an_empty_recorded_skill_list(store):
    """An empty profile is evidence of absence; a MISSING profile entry is not."""
    claim = _claim(claim_type="skill_gap", predicate="skill_not_recorded", job_id=JOB,
                   claim_args={"skill": "excel"},
                   evidence_ids=[_job(store, "required_skills", ["excel"]),
                                 _candidate(store, "skills_have", [],
                                            source=EvidenceSource.PROFILE)])
    assert semantic_status(claim, store) == "supported"


# ============================== 4. ranking_match(job_id, field, candidate_value, job_value)
def test_ranking_match_holds_when_both_sides_agree_with_the_claim(store):
    claim = _claim(claim_type="ranking_reason", predicate="ranking_match", job_id=JOB,
                   field_name="preferred_locations", expected_value="Kuala Lumpur",
                   observed_value="Kuala Lumpur", claim_args={"feature": "location_match"},
                   evidence_ids=[_candidate(store, "preferred_locations", "Kuala Lumpur"),
                                 _job(store, "city", "Kuala Lumpur")])
    assert semantic_status(claim, store) == "supported"


def test_ranking_match_rejects_a_job_value_that_differs(store):
    claim = _claim(claim_type="ranking_reason", predicate="ranking_match", job_id=JOB,
                   field_name="preferred_locations", expected_value="Kuala Lumpur",
                   observed_value="Kuala Lumpur", claim_args={"feature": "location_match"},
                   evidence_ids=[_candidate(store, "preferred_locations", "Kuala Lumpur"),
                                 _job(store, "city", "Penang")])
    assert semantic_status(claim, store) == "unsupported"


def test_ranking_match_rejects_evidence_about_another_job(store):
    claim = _claim(claim_type="ranking_reason", predicate="ranking_match", job_id=JOB,
                   field_name="preferred_locations", expected_value="Kuala Lumpur",
                   observed_value="Kuala Lumpur", claim_args={"feature": "location_match"},
                   evidence_ids=[_candidate(store, "preferred_locations", "Kuala Lumpur"),
                                 _job(store, "city", "Kuala Lumpur", job_id=OTHER_JOB)])
    assert semantic_status(claim, store) == "unsupported"


def test_ranking_match_requires_a_named_feature(store):
    """The claim has to say which ranking feature it is explaining."""
    claim = _claim(claim_type="ranking_reason", predicate="ranking_match", job_id=JOB,
                   field_name="preferred_locations", expected_value="Kuala Lumpur",
                   observed_value="Kuala Lumpur",
                   evidence_ids=[_candidate(store, "preferred_locations", "Kuala Lumpur"),
                                 _job(store, "city", "Kuala Lumpur")])
    assert semantic_status(claim, store) == "unsupported"


# ================ 5. no_match_cause(field, blocked_job_ids, evaluated_job_ids)
def test_no_match_cause_holds_when_the_stage_record_matches_the_field_and_count(store):
    claim = _claim(claim_type="no_match_cause", predicate="no_match_cause",
                   field_name="salary_min",
                   claim_args={"removed": 7, "evaluated_jobs": 18},
                   evidence_ids=[_candidate(store, "salary_min", 4000),
                                 _stage(store, "salary_min", 7)])
    assert semantic_status(claim, store) == "supported"


def test_no_match_cause_rejects_a_count_that_disagrees_with_the_record(store):
    claim = _claim(claim_type="no_match_cause", predicate="no_match_cause",
                   field_name="salary_min",
                   claim_args={"removed": 12, "evaluated_jobs": 18},
                   evidence_ids=[_candidate(store, "salary_min", 4000),
                                 _stage(store, "salary_min", 7)])
    assert semantic_status(claim, store) == "unsupported"


def test_no_match_cause_needs_a_filtering_record_not_just_the_constraint(store):
    """"the requirement exists" is correlational; the claim is causal."""
    claim = _claim(claim_type="no_match_cause", predicate="no_match_cause",
                   field_name="salary_min", claim_args={"removed": 7, "evaluated_jobs": 18},
                   evidence_ids=[_candidate(store, "salary_min", 4000)])
    assert semantic_status(claim, store) == "unsupported"


def test_no_match_cause_checks_blocked_ids_against_the_count_when_present(store):
    """P0-5 will record the actual ids; the check tightens automatically when it does."""
    ok = _claim(claim_type="no_match_cause", predicate="no_match_cause",
                field_name="salary_min",
                claim_args={"removed": 2, "evaluated_jobs": 18,
                            "blocked_job_ids": ["job-7", "job-9"]},
                evidence_ids=[_candidate(store, "salary_min", 4000),
                              _stage(store, "salary_min", 2, blocked=["job-7", "job-9"])])
    assert semantic_status(ok, store) == "supported"

    mismatched = _claim(claim_type="no_match_cause", predicate="no_match_cause",
                        field_name="salary_min",
                        claim_args={"removed": 2, "evaluated_jobs": 18,
                                    "blocked_job_ids": ["job-7", "job-8"]},
                        evidence_ids=[_candidate(store, "salary_min", 4000),
                                      _stage(store, "salary_min", 2,
                                             blocked=["job-7", "job-9"])])
    assert semantic_status(mismatched, store) == "unsupported"


# ============================================= constraint_applied, the non-causal form
def test_constraint_applied_holds_on_the_candidates_own_statement(store):
    claim = _claim(claim_type="no_match_reason", predicate="constraint_applied",
                   field_name="work_modes",
                   evidence_ids=[_candidate(store, "work_modes", "onsite")])
    assert semantic_status(claim, store) == "supported"


def test_constraint_applied_rejects_a_different_fields_statement(store):
    claim = _claim(claim_type="no_match_reason", predicate="constraint_applied",
                   field_name="work_modes",
                   evidence_ids=[_candidate(store, "salary_min", 4000)])
    assert semantic_status(claim, store) == "unsupported"


# ==================================================================== 6. unknown
def test_an_unknown_predicate_is_unknown_rather_than_supported(store):
    claim = _claim(claim_type="candidate_preference", predicate="vibes_align",
                   field_name="salary_min", expected_value=4000,
                   evidence_ids=[_candidate(store, "salary_min", 4000)])
    assert semantic_status(claim, store) == "unknown"


def test_a_claim_with_no_predicate_is_unknown_rather_than_supported(store):
    """A builder that forgets to state its proposition must not be rewarded with a pass.

    This is also what keeps a legacy bundle's claims from being silently re-blessed if they
    are ever fed back through the validator: no predicate, no verdict.
    """
    claim = _claim(claim_type="candidate_preference", field_name="salary_min",
                   expected_value=4000,
                   evidence_ids=[_candidate(store, "salary_min", 4000)])
    assert semantic_status(claim, store) == "unknown"


def test_an_unknown_verdict_drops_the_claim_from_the_response(store):
    from jobrec.agents.explanation_agent import validate_claims

    claim = _claim(claim_type="candidate_preference", predicate="vibes_align",
                   field_name="salary_min", expected_value=4000,
                   evidence_ids=[_candidate(store, "salary_min", 4000)])
    delivered, dropped = validate_claims([claim], store)

    assert delivered == []
    assert len(dropped) == 1
    assert dropped[0].semantic_status == "unknown"
    # The trace dimension still passes -- the ids DO resolve. Keeping the two verdicts
    # separate is what lets a reader tell a dangling reference from an unverifiable
    # proposition.
    assert dropped[0].trace_status == "supported"
    assert dropped[0].support_status == "unknown"


def test_evidence_that_does_not_resolve_is_still_unsupported(store):
    """The original structural check has not been dropped, only joined."""
    claim = _claim(claim_type="candidate_preference", predicate="candidate_preference",
                   field_name="salary_min", expected_value=4000,
                   evidence_ids=["ev-does-not-exist"])
    assert semantic_status(claim, store) == "unsupported"


def test_no_evidence_at_all_is_unsupported(store):
    claim = _claim(claim_type="candidate_preference", predicate="candidate_preference",
                   field_name="salary_min", expected_value=4000, evidence_ids=[])
    assert semantic_status(claim, store) == "unsupported"
