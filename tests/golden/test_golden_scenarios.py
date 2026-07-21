"""Golden scenario tests (landing-plan section 22.4).

Runs the fixed scenario set through the deterministic pipeline and asserts the
expected behaviour, plus the ablation-difference and claim-validator scenarios.
"""

from __future__ import annotations

import pytest

from jobrec.app_service import AppService
from jobrec.config import load_config
from jobrec.evaluation.experiment_runner import load_scenarios

SCENARIOS = {s["scenario_id"]: s for s in load_scenarios("data/scenarios/scenarios.jsonl")}


def _service(variant: str) -> AppService:
    cfg = load_config("configs/experiment_full.yaml", base_dir="configs")
    from jobrec.domain.enums import ExperimentVariant

    cfg.experiment.variant = ExperimentVariant(variant)
    return AppService(cfg, "data/processed/jobs.jsonl")


def _run(scenario, variant="full"):
    svc = _service(variant)
    profile = dict(scenario["profile"])
    profile.setdefault("candidate_id", scenario["scenario_id"] + "-c")
    svc.create_candidate(profile)
    sid = svc.create_session(profile["candidate_id"], variant)
    result = None
    for text in scenario["turns"]:
        result = svc.process_turn(sid, text, scenario_id=scenario["scenario_id"])
    return result


@pytest.mark.parametrize("sid", list(SCENARIOS.keys()))
def test_scenarios_run_successfully(sid):
    result = _run(SCENARIOS[sid])
    assert result.run_record.success
    assert len(result.dropped_claims) == 0  # no unsupported factual claims


def test_sc01_recommendation():
    assert _run(SCENARIOS["sc01_complete_request"]).response.response_type == "recommendation"


def test_sc02_clarification_missing_role():
    result = _run(SCENARIOS["sc02_missing_role"])
    assert result.response.response_type == "clarification"
    assert result.clarification.target_fields == ["target_roles"]


def test_sc03_location_override():
    result = _run(SCENARIOS["sc03_profile_location_conflict"])
    assert result.active_search_state.preferred_locations == ["Kuala Lumpur"]


def test_sc04_latest_salary_controls():
    result = _run(SCENARIOS["sc04_salary_change"])
    assert result.active_search_state.salary_min == 4000.0


def test_sc07_no_match_blocks_on_salary():
    result = _run(SCENARIOS["sc07_all_salary_fail"])
    assert result.response.response_type == "no_match"
    assert "salary_min" in result.decision.no_match_reason_codes


def test_sc08_no_expired_in_results():
    result = _run(SCENARIOS["sc08_expired_excluded"])
    svc_jobs = {j.job_id: j for j in _service("full").jobs}
    for jid in result.decision.selected_job_ids:
        job = svc_jobs[jid]
        assert job.is_active
        assert job.application_deadline is None or str(job.application_deadline) >= "2026-01-01"


def test_sc09_unknown_salary_not_eligible_when_hard():
    result = _run(SCENARIOS["sc09_salary_unknown"])
    jobs = {j.job_id: j for j in _service("full").jobs}
    eligible_ids = [e.job_id for e in result.decision.eligibility_results if e.eligible]
    for jid in eligible_ids:
        assert jobs[jid].salary_min_monthly_myr is not None


def test_sc10_memory_remembers_role():
    result = _run(SCENARIOS["sc10_multiturn_memory"], variant="full")
    assert "data analyst" in result.active_search_state.target_roles


def test_no_memory_loses_prior_turn_role():
    # In no_memory, a role stated in turn 1 is NOT available in turn 2.
    result = _run(SCENARIOS["sc10_multiturn_memory"], variant="no_memory")
    assert "data analyst" not in result.active_search_state.target_roles
