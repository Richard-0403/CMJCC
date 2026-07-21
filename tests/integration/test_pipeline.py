"""Integration tests across the full deterministic pipeline via AppService."""

from __future__ import annotations

import pytest

from jobrec.app_service import AppService
from jobrec.config import load_config


@pytest.fixture()
def svc():
    cfg = load_config("configs/experiment_full.yaml", base_dir="configs")
    return AppService(cfg, "data/processed/jobs.jsonl")


def _candidate(svc, **overrides):
    profile = {"candidate_id": "c1", "skills": ["Python", "SQL"], "years_experience": 1,
               "preferred_locations": ["Kuala Lumpur"], "work_modes": ["hybrid"]}
    profile.update(overrides)
    return svc.create_candidate(profile)


def test_create_session_and_recommend(svc):
    _candidate(svc)
    sid = svc.create_session("c1", "full")
    res = svc.process_turn(sid, "I want a data analyst role in Kuala Lumpur, at least RM4000.")
    assert res.response.response_type == "recommendation"
    assert res.decision is not None and len(res.decision.selected_job_ids) > 0
    assert len(res.dropped_claims) == 0
    assert all(h.validation_passed for h in res.handoffs)


def test_no_match_reports_blocking(svc):
    _candidate(svc, candidate_id="c2")
    sid = svc.create_session("c2", "full")
    res = svc.process_turn(sid, "Only a data analyst in Kuala Lumpur paying at least RM50000 per month.")
    assert res.response.response_type == "no_match"
    assert res.decision.no_match
    assert "salary_min" in res.decision.no_match_reason_codes


def test_llm_failure_falls_back_to_rules(svc, monkeypatch):
    """A failing provider must not corrupt state; rule extraction takes over."""
    from jobrec.domain.enums import RunMode

    _candidate(svc, candidate_id="c3")
    sid = svc.create_session("c3", "full")
    orch, _ = svc._orchestrator_for(sid, "full")

    class BoomProvider:
        name = "boom"
        model = "boom"

        def complete_json(self, prompt, *, purpose):
            from jobrec.llm.provider import LLMInvalidJSON
            raise LLMInvalidJSON("boom")
        def complete_text(self, prompt, *, purpose, fallback=""):
            return fallback, None
        def manifest(self):
            return {"provider": "boom"}

    orch.provider = BoomProvider()
    orch.config.llm.mode = RunMode.HYBRID
    res = svc.process_turn(sid, "data analyst in Kuala Lumpur at least RM4000")
    # even with a broken model, deterministic fallback yields a valid response
    assert res.response.response_type in {"recommendation", "clarification", "no_match"}
    assert res.run_record.success


def test_multiturn_memory_remembers_role(svc):
    svc.create_candidate({"candidate_id": "c4", "skills": ["Python", "SQL"],
                          "years_experience": 1, "preferred_locations": ["Kuala Lumpur"]})
    sid = svc.create_session("c4", "full")
    svc.process_turn(sid, "I am interested in data analyst roles.")
    res2 = svc.process_turn(sid, "Something hybrid with at least RM4000 would be great.")
    assert "data analyst" in res2.active_search_state.target_roles
