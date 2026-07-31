"""Salary threshold policy: guaranteed minimum, one rule, every layer.

"at least RM4000" is a statement about what the candidate will be GUARANTEED, so a posting
satisfies it only when its own floor clears the threshold. Eligibility used to accept a
merely overlapping range -- 3000-4500 passed a 4000 minimum through an outcome named
``salary_range_crosses_min`` -- on the strength of a ceiling the candidate has no claim to.

Two things made that hard to see. It was systematic rather than occasional, so no single
case looked wrong, and the automatic oracle grades through the same comparison, so the
over-generous reading was ground truth too and no metric could contradict it. Human
adjudication is what surfaced it.

The audit plan proposed giving the oracle its own independent implementation. These tests
deliberately do NOT do that. A second implementation catches CODING slips, and the defect
here was a SPECIFICATION error: written twice, the wrong rule would have been written
twice. What it does instead is pin the rule at its boundaries and assert that every layer
which speaks about salary agrees on the same pair -- which is the failure that actually
costs something, and the one a shared implementation cannot rule out on its own once
someone changes it.
"""

from __future__ import annotations

import pytest

from jobrec.agents.job_context_agent import JobContextAgent
from jobrec.config import load_config
from jobrec.domain.enums import ConstraintOutcome
from jobrec.ranking.features import salary_preference

THRESHOLD = 4000.0


@pytest.fixture(scope="module")
def config():
    return load_config("configs/experiment_full.yaml", base_dir="configs")


@pytest.fixture(scope="module")
def agent(config):
    return JobContextAgent(config)


def _job(minimum, maximum=None, **extra):
    """A catalogue-shaped posting carrying only what the salary rule reads."""
    from jobrec.domain.job import JobPosting
    from jobrec.utils.time import utcnow

    return JobPosting(
        job_id="job-test", title="Data Analyst", company="X", description="d",
        normalized_title="data analyst", role_family="data analyst",
        required_skills=[], preferred_skills=[],
        salary_min_monthly_myr=minimum, salary_max_monthly_myr=maximum,
        country="MY", city="Kuala Lumpur", work_mode="hybrid",
        is_active=True,
        source_snapshot_id="snap-test", ingested_at=utcnow(),
        raw_payload_hash="hash-test", **extra)


# ------------------------------------------------------------------ boundary table
@pytest.mark.parametrize("minimum,maximum,expected,code", [
    # the posting's floor decides, and only the floor
    (THRESHOLD - 1, None, ConstraintOutcome.FAIL, "salary_below_min"),
    (THRESHOLD, None, ConstraintOutcome.PASS, "salary_meets_min"),
    (THRESHOLD + 1, None, ConstraintOutcome.PASS, "salary_meets_min"),
    # a range that straddles the threshold is NOT satisfied: the guaranteed part is below it
    (THRESHOLD - 500, THRESHOLD + 1000, ConstraintOutcome.FAIL, "salary_below_min"),
    (THRESHOLD - 1, THRESHOLD * 10, ConstraintOutcome.FAIL, "salary_below_min"),
    # a range entirely above it is
    (THRESHOLD, THRESHOLD + 2000, ConstraintOutcome.PASS, "salary_meets_min"),
    # no stated floor is UNKNOWN, and a ceiling does not stand in for one
    (None, THRESHOLD + 5000, ConstraintOutcome.UNKNOWN, "salary_minimum_unknown"),
    (None, None, ConstraintOutcome.UNKNOWN, "salary_minimum_unknown"),
])
def test_the_posting_floor_decides(agent, minimum, maximum, expected, code) -> None:
    outcome, observed, explanation = agent._check_salary(
        _job(minimum, maximum), THRESHOLD)
    assert outcome == expected, (minimum, maximum, outcome)
    assert explanation == code
    assert observed["compared_field"] == "salary_min_monthly_myr"


def test_the_overlapping_range_pass_is_gone(agent) -> None:
    """No input may produce the old ``salary_range_crosses_min`` PASS."""
    for minimum in (None, 0.0, THRESHOLD - 1000, THRESHOLD, THRESHOLD + 1000):
        for maximum in (None, THRESHOLD - 1, THRESHOLD, THRESHOLD + 5000):
            _outcome, _observed, code = agent._check_salary(_job(minimum, maximum), THRESHOLD)
            assert code != "salary_range_crosses_min", (minimum, maximum)


# ------------------------------------------------------- cross-layer agreement
@pytest.mark.parametrize("minimum,maximum", [
    (THRESHOLD - 500, THRESHOLD + 1000),
    (THRESHOLD - 1, None),
    (THRESHOLD, None),
    (THRESHOLD + 1, THRESHOLD + 2000),
])
def test_eligibility_and_ranking_never_contradict_each_other(
    agent, minimum, maximum
) -> None:
    """The ranking feature may only claim the minimum is MET when eligibility agrees.

    Ranking is allowed to award partial credit for a near miss -- it is a graded soft
    feature. What it may not do is emit ``salary_meets_min``, which becomes the
    user-visible claim "Salary meets your stated minimum", for a posting eligibility
    rejects on that very constraint.
    """
    from jobrec.domain.job import ActiveSearchState
    from jobrec.utils.time import utcnow

    job = _job(minimum, maximum)
    outcome, _observed, _code = agent._check_salary(job, THRESHOLD)
    active = ActiveSearchState(
        active_search_id="as-test", session_id="s", candidate_id="c",
        candidate_state_version=1, dialogue_state_version=1,
        salary_min=THRESHOLD, generated_at=utcnow())
    feature = salary_preference(active, job, salary_scale=1000.0, penalize_unknown=False)

    if feature.code == "salary_meets_min":
        assert outcome == ConstraintOutcome.PASS, (
            f"ranking claims the minimum is met while eligibility returned {outcome}"
        )
    if outcome == ConstraintOutcome.PASS:
        assert feature.code == "salary_meets_min", (
            f"eligibility passed the salary constraint but ranking reported "
            f"{feature.code}"
        )


def test_the_claim_cites_the_field_that_was_compared(agent) -> None:
    """Evidence for a salary claim must name the normalised projection.

    The raw ``salary_min`` is in the posting's own currency and period. Citing it showed a
    reader ``job_posting:salary_min=1350`` as support for meeting a stated 4000, while the
    comparison had correctly used the normalised 4725 MYR -- a true conclusion that its own
    evidence contradicted, and the reason 275 of these claims were adjudicated unsupported.
    """
    from jobrec.domain.job import ActiveSearchState
    from jobrec.utils.time import utcnow

    active = ActiveSearchState(
        active_search_id="as-test", session_id="s", candidate_id="c",
        candidate_state_version=1, dialogue_state_version=1,
        salary_min=THRESHOLD, generated_at=utcnow())
    for minimum, maximum in ((THRESHOLD + 100, None), (THRESHOLD - 100, THRESHOLD + 100)):
        feature = salary_preference(active, _job(minimum, maximum),
                                    salary_scale=1000.0, penalize_unknown=False)
        assert feature.job_fields, feature.code
        assert all(f.endswith("_monthly_myr") for f in feature.job_fields), (
            f"{feature.code} cites un-normalised field(s): {feature.job_fields}"
        )
