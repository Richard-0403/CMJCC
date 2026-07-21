"""Unit tests for the constraint / eligibility engine."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from jobrec.agents.job_context_agent import JobContextAgent
from jobrec.domain.enums import ConstraintOutcome
from jobrec.domain.job import JobPosting
from tests.conftest import make_active


def _job(**overrides) -> JobPosting:
    base = dict(
        job_id="j1", title="Junior Data Analyst", company="Acme",
        description="d", normalized_title="junior data analyst",
        role_family="data analyst", required_skills=["python", "sql"],
        preferred_skills=[], salary_min=5000, salary_max=7000, salary_currency="MYR",
        salary_period="month", salary_min_monthly_myr=5000.0, salary_max_monthly_myr=7000.0,
        country="Malaysia", city="Kuala Lumpur", work_mode="hybrid",
        min_years_experience=1, max_years_experience=3, experience_level="junior",
        required_work_authorization=[], application_deadline=date(2026, 6, 1),
        is_active=True, source_snapshot_id="snap", ingested_at=datetime(2026, 1, 1),
        raw_payload_hash="h",
    )
    base.update(overrides)
    return JobPosting(**base)


@pytest.fixture()
def agent(config):
    return JobContextAgent(config)


def test_salary_pass_when_range_meets_min(agent):
    ctx = agent.build_context(make_active(salary_min=4000.0), "snap")
    res = agent.evaluate(_job(salary_min_monthly_myr=5000.0, salary_max_monthly_myr=7000.0), ctx)
    assert res.eligible


def test_salary_fail_when_max_below_min(agent):
    ctx = agent.build_context(make_active(salary_min=8000.0), "snap")
    res = agent.evaluate(_job(salary_min_monthly_myr=5000.0, salary_max_monthly_myr=7000.0), ctx)
    assert not res.eligible
    assert "salary_min:salary_below_min" in res.filtered_reason_codes


def test_unknown_salary_hard_fails_by_policy(agent):
    ctx = agent.build_context(make_active(salary_min=4000.0), "snap")
    job = _job(salary_min=None, salary_max=None, salary_min_monthly_myr=None, salary_max_monthly_myr=None)
    res = agent.evaluate(job, ctx)
    # hard_constraint_unknown_default is 'fail' in the full config
    assert not res.eligible


def test_location_mismatch_fails(agent):
    ctx = agent.build_context(make_active(preferred_locations=["Penang"]), "snap")
    res = agent.evaluate(_job(city="Kuala Lumpur"), ctx)
    assert not res.eligible


def test_expired_job_fails(agent):
    ctx = agent.build_context(make_active(), "snap")
    res = agent.evaluate(_job(application_deadline=date(2025, 1, 1), is_active=False), ctx)
    assert not res.eligible
    assert any("not_expired" in c for c in res.filtered_reason_codes) or res.hard_violation_count > 0


def test_active_no_deadline_passes(agent):
    ctx = agent.build_context(make_active(salary_min=None, preferred_locations=[]), "snap")
    res = agent.evaluate(_job(application_deadline=None, is_active=True), ctx)
    checks = {c.field_name: c.outcome for c in res.checks}
    assert checks["not_expired"] == ConstraintOutcome.PASS


def test_required_skill_hard_fails_when_missing(agent):
    active = make_active(skills_have=["python"], hard_constraint_fields=["required_skills"])
    ctx = agent.build_context(active, "snap")
    res = agent.evaluate(_job(required_skills=["python", "sql", "aws"]), ctx)
    assert not res.eligible
