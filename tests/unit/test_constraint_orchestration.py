"""Constraint-orchestration suite: hard filtering, unknown policy, no-match diagnosis.

Requirement 20 asks for a dedicated suite that verifies *hard-filter-before-rank*
behaviour end to end at the agent level: which jobs the constraint bundle removes,
how missing job fields are treated per :class:`UnknownPolicy`, and what the system
reports when nothing survives. ``tests/unit/test_constraints.py`` covers individual
field operators; this module covers the orchestration around them — the filter ->
rank -> select sequence and the diagnosis produced when that sequence yields nothing.

**Validates: Requirements 20.1, 20.2**
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from jobrec.agents.job_context_agent import JobContextAgent, diagnose_no_match
from jobrec.domain.constraints import JobContextState
from jobrec.domain.enums import (
    ConstraintOutcome,
    ConstraintStrength,
    ExperimentVariant,
    UnknownPolicy,
)
from jobrec.domain.job import ActiveSearchState, JobPosting
from jobrec.domain.recommendation import RecommendationDecision
from jobrec.evidence_store import EvidenceStore
from jobrec.orchestration.feature_flags import FeatureFlags
from jobrec.ranking.scoring import SCORER_VERSION, RankingAgent
from tests.conftest import make_active

# Shared Hypothesis strategies for "an active search plus a job set that always
# contains one compatible job". They live in test_memory_and_ranking because that
# module needed them first; reused here rather than duplicated so both properties
# explore the same generated shape.
from tests.unit.test_memory_and_ranking import _active_and_jobs


def _job(job_id: str = "j-fit", **overrides) -> JobPosting:
    """A job that satisfies the default :func:`make_active` search."""
    base = dict(
        job_id=job_id, title="Junior Data Analyst", company="Acme",
        description="d", normalized_title="junior data analyst",
        role_family="data analyst", required_skills=["python", "sql"],
        preferred_skills=[], employment_type="full_time",
        salary_min=5000, salary_max=7000, salary_currency="MYR", salary_period="month",
        salary_min_monthly_myr=5000.0, salary_max_monthly_myr=7000.0,
        country="Malaysia", city="Kuala Lumpur", work_mode="hybrid",
        min_years_experience=1, max_years_experience=3, experience_level="junior",
        required_work_authorization=[], application_deadline=date(2026, 6, 1),
        is_active=True, source_snapshot_id="snap", ingested_at=datetime(2026, 1, 1, tzinfo=UTC),
        raw_payload_hash="h",
    )
    base.update(overrides)
    return JobPosting(**base)


def _decide(
    cfg, active: ActiveSearchState, jobs: list[JobPosting]
) -> tuple[JobContextState, list, RecommendationDecision]:
    """Run the real filter-then-rank path and assemble the decision it produces.

    Mirrors ``ConversationOrchestrator.process_turn`` steps 5-6: evaluate every job
    against the constraint bundle, rank only what survived, then select the first
    ``top_k`` ranked entries. Nothing here re-implements a constraint or a score.
    """
    agent = JobContextAgent(cfg)
    context = agent.build_context(active, "snap")
    eligibility = [agent.evaluate(job, context) for job in jobs]
    ranked = RankingAgent(EvidenceStore(), cfg).rank(
        active, {j.job_id: j for j in jobs}, eligibility
    )
    decision = RecommendationDecision(
        decision_id="dec-test",
        session_id=active.session_id,
        active_search_id=active.active_search_id,
        context_id=context.context_id,
        experiment_variant=cfg.experiment.variant.value,
        retrieved_job_ids=[j.job_id for j in jobs],
        eligibility_results=eligibility,
        ranked_jobs=ranked,
        selected_job_ids=[rj.job_id for rj in ranked[: cfg.experiment.top_k]],
        no_match=len(ranked) == 0,
        created_at=datetime.now(UTC),
        scorer_version=SCORER_VERSION,
        config_hash=cfg.config_hash(),
    )
    return context, eligibility, decision


def _with_unknown_default(config, policy: UnknownPolicy):
    cfg = config.model_copy(deep=True)
    cfg.context.hard_constraint_unknown_default = policy
    return cfg


def _no_salary(job_id: str = "j-fit", **overrides) -> JobPosting:
    """The compatible job with every salary field missing (unknown-policy input)."""
    return _job(
        job_id, salary_min=None, salary_max=None, salary_currency=None,
        salary_period="unknown", salary_min_monthly_myr=None, salary_max_monthly_myr=None,
        **overrides,
    )


def _outcomes(result) -> dict[str, ConstraintOutcome]:
    return {c.field_name: c.outcome for c in result.checks}


# ---------------------------------------------------------------------------
# Hard-constraint filtering
# ---------------------------------------------------------------------------


def test_hard_constraint_violation_filters_job_out(config):
    active = make_active(
        preferred_locations=["Penang"], hard_constraint_fields=["preferred_locations"]
    )
    agent = JobContextAgent(config)
    result = agent.evaluate(_job(city="Kuala Lumpur"), agent.build_context(active, "snap"))

    assert not result.eligible
    assert result.hard_violation_count == 1
    assert "preferred_locations:location_mismatch" in result.filtered_reason_codes


def test_soft_preference_mismatch_scores_but_never_filters(config):
    """A failing soft preference stays in the audit trail without removing the job."""
    active = make_active(
        work_modes=["remote"],
        hard_constraint_fields=["preferred_locations"],
        soft_preference_fields=["work_modes"],
    )
    agent = JobContextAgent(config)
    context = agent.build_context(active, "snap")
    result = agent.evaluate(_job(work_mode="onsite"), context)

    strengths = {c.field_name: c.strength for c in context.constraints}
    assert strengths["work_modes"] == ConstraintStrength.SOFT
    assert _outcomes(result)["work_modes"] == ConstraintOutcome.FAIL
    assert result.eligible
    assert result.hard_violation_count == 0
    assert result.filtered_reason_codes == []


def test_hard_filtering_happens_before_ranking(config):
    """Only eligible jobs are scored, and only scored jobs can be selected."""
    active = make_active(salary_min=6000.0, hard_constraint_fields=["salary_min"])
    good = _job("j-good", salary_min_monthly_myr=7000.0, salary_max_monthly_myr=9000.0)
    violating = _job("j-poor", salary_min_monthly_myr=2000.0, salary_max_monthly_myr=3000.0)

    _, eligibility, decision = _decide(config, active, [good, violating])

    # The filtered job keeps its audit record but never reaches the ranker.
    assert {e.job_id for e in eligibility} == {"j-good", "j-poor"}
    assert {e.job_id for e in eligibility if not e.eligible} == {"j-poor"}
    assert [rj.job_id for rj in decision.ranked_jobs] == ["j-good"]
    assert decision.selected_job_ids == ["j-good"]
    assert not decision.no_match


def test_full_variant_enables_explicit_constraint_orchestration(config):
    """The full variant is the one that runs the hard filter (not a forked path)."""
    assert config.experiment.variant == ExperimentVariant.FULL
    assert FeatureFlags.from_config(config).explicit_constraint_orchestration


# ---------------------------------------------------------------------------
# Unknown-constraint policy
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("policy", "eligible"),
    [
        (UnknownPolicy.FAIL, False),
        (UnknownPolicy.PASS, True),
        (UnknownPolicy.PENALIZE, True),
        (UnknownPolicy.CLARIFY, True),
    ],
)
def test_unknown_hard_field_follows_configured_policy(config, policy, eligible):
    """A missing job field is never a silent pass: it is failed or counted."""
    cfg = _with_unknown_default(config, policy)
    active = make_active(salary_min=4000.0, hard_constraint_fields=["salary_min"])
    agent = JobContextAgent(cfg)
    result = agent.evaluate(_no_salary(), agent.build_context(active, "snap"))

    assert _outcomes(result)["salary_min"] == ConstraintOutcome.UNKNOWN
    assert result.eligible is eligible
    if eligible:
        assert result.hard_violation_count == 0
        assert result.unknown_hard_constraint_count == 1
        assert f"salary_min:unknown_{policy.value}" in result.filtered_reason_codes
    else:
        assert result.hard_violation_count == 1
        assert result.unknown_hard_constraint_count == 0
        assert "salary_min:unknown_fail" in result.filtered_reason_codes


def test_unknown_work_mode_fails_regardless_of_default_policy(config):
    """work_modes carries a field-specific FAIL policy that the default cannot loosen."""
    cfg = _with_unknown_default(config, UnknownPolicy.PASS)
    active = make_active(work_modes=["hybrid"], hard_constraint_fields=["work_modes"])
    agent = JobContextAgent(cfg)
    context = agent.build_context(active, "snap")
    policies = {c.field_name: c.unknown_policy for c in context.constraints}
    result = agent.evaluate(_job(work_mode="unspecified"), context)

    assert policies["work_modes"] == UnknownPolicy.FAIL
    assert not result.eligible
    assert "work_modes:unknown_fail" in result.filtered_reason_codes


def test_unknown_soft_field_is_not_a_hard_unknown(config):
    """Unknown values on soft fields are scored, not filtered or counted as hard."""
    active = make_active(
        work_modes=["hybrid"],
        hard_constraint_fields=["preferred_locations"],
        soft_preference_fields=["work_modes"],
    )
    agent = JobContextAgent(config)
    context = agent.build_context(active, "snap")
    policies = {c.field_name: c.unknown_policy for c in context.constraints}
    result = agent.evaluate(_job(work_mode="unspecified"), context)

    assert policies["work_modes"] == UnknownPolicy.PASS
    assert _outcomes(result)["work_modes"] == ConstraintOutcome.UNKNOWN
    assert result.eligible
    assert result.unknown_hard_constraint_count == 0


# ---------------------------------------------------------------------------
# No-match diagnosis
# ---------------------------------------------------------------------------


def test_no_match_diagnosis_ranks_blocking_constraints(config):
    active = make_active(
        preferred_locations=["Penang"], salary_min=9000.0,
        hard_constraint_fields=["preferred_locations", "salary_min"],
    )
    both = _job("j-both", city="Kuala Lumpur",
                salary_min_monthly_myr=5000.0, salary_max_monthly_myr=7000.0)
    location_only = _job("j-loc", city="Kuala Lumpur",
                         salary_min_monthly_myr=12000.0, salary_max_monthly_myr=15000.0)

    context, eligibility, decision = _decide(config, active, [both, location_only])
    assert decision.no_match
    assert decision.selected_job_ids == []

    diagnosis = diagnose_no_match(eligibility, context)
    counts = {b["field"]: b["filtered_jobs"] for b in diagnosis["blocking_constraints"]}
    assert diagnosis["no_match"] is True
    assert counts == {"preferred_locations": 2, "salary_min": 1}
    # Most-blocking constraint is reported first so the response can name it.
    assert [b["field"] for b in diagnosis["blocking_constraints"]][0] == "preferred_locations"


def test_no_match_diagnosis_never_offers_a_hard_field_for_relaxation(config):
    """Stated hard constraints are reported, never proposed for silent relaxation."""
    active = make_active(
        preferred_locations=["Penang"], hard_constraint_fields=["preferred_locations"]
    )
    context, eligibility, decision = _decide(config, active, [_job(city="Kuala Lumpur")])
    assert decision.no_match

    diagnosis = diagnose_no_match(eligibility, context)
    hard_fields = {c.field_name for c in context.constraints
                   if c.strength == ConstraintStrength.HARD}
    assert "preferred_locations" in {b["field"] for b in diagnosis["blocking_constraints"]}
    for candidate in diagnosis["relaxation_candidates"]:
        assert candidate["field"] not in hard_fields
        assert candidate["requires_confirmation"] is True


def test_no_match_diagnosis_reports_unknown_policy_blocks(config):
    """A job filtered by the unknown policy is still explained, not silently dropped."""
    active = make_active(salary_min=4000.0, hard_constraint_fields=["salary_min"])
    context, eligibility, decision = _decide(config, active, [_no_salary("j-unknown")])
    assert decision.no_match

    diagnosis = diagnose_no_match(eligibility, context)
    assert {b["field"] for b in diagnosis["blocking_constraints"]} == {"salary_min"}


# ---------------------------------------------------------------------------
# Property-based test (Property 16)
# ---------------------------------------------------------------------------

# Feature: cmjcc-experiment-readiness, Property 16: No hard-violating job is ever selected
# under the full variant
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    case=_active_and_jobs(),
    unknown_default=st.sampled_from(list(UnknownPolicy)),
)
def test_property_no_hard_violating_job_is_selected_under_full_variant(
    config, case, unknown_default
) -> None:
    """Every id in ``selected_job_ids`` passes all applicable hard constraints.

    The generated catalog is driven through the real filter-then-rank path under the
    full variant, so a hard-violating job could only be selected if hard filtering
    stopped preceding ranking.

    **Validates: Requirements 31.2**
    """
    active, jobs = case
    cfg = config.model_copy(deep=True)
    cfg.experiment.variant = ExperimentVariant.FULL
    cfg.context.hard_constraint_unknown_default = unknown_default
    assert FeatureFlags.from_config(cfg).explicit_constraint_orchestration

    context, eligibility, decision = _decide(cfg, active, jobs)
    by_id = {e.job_id: e for e in eligibility}
    hard_ids = {c.constraint_id for c in context.constraints
                if c.strength == ConstraintStrength.HARD}
    selected = set(decision.selected_job_ids)

    # Non-vacuity: the seeded compatible job always survives, so there is something
    # for the filter to have wrongly let through.
    assert selected, "at least the seeded compatible job must be selected"
    assert len(selected) <= cfg.experiment.top_k

    for job_id in decision.selected_job_ids:
        result = by_id[job_id]
        assert result.eligible
        assert result.hard_violation_count == 0
        failed_hard = [c.field_name for c in result.checks
                       if c.constraint_id in hard_ids and c.outcome == ConstraintOutcome.FAIL]
        assert not failed_hard, f"{job_id} selected despite hard failures {failed_hard}"

    # Conversely, no filtered job is selected or even scored.
    filtered = {e.job_id for e in eligibility if not e.eligible}
    assert not (filtered & selected)
    assert not (filtered & {rj.job_id for rj in decision.ranked_jobs})
