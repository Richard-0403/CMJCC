"""Integration tests for the HYBRID-mode pipeline via AppService.

This is the project's first hybrid-mode pipeline test: every other suite runs
``deterministic`` mode, which bypasses the model-extraction path entirely
(``orchestrator._extract`` short-circuits to the rule extractor). Hybrid mode with the
``mock`` provider is the deterministic-but-model-shaped path where
``parse_extraction_lenient`` and ``validate_extraction`` actually run, so it is the only
configuration that exercises R8 field validation end to end - and where a whole turn used
to fail with ``INTERNAL_ERROR`` (``TypeError: float() argument must be a string or a real
number, not 'dict'``) because field validation replaces a stated ``salary_min`` with the
canonical ``{min_salary, max_salary, currency, period}`` structure (R8.2) while
``MemoryAgent`` compared it against long-term memory with ``float()``.

Regression coverage: a hybrid turn completes AND the stated salary constraint survives
(R8.9 - never silently drop a stated constraint), including the long-term write-back that
feeds later turns.
"""

from __future__ import annotations

import pytest

from jobrec.app_service import AppService
from jobrec.config import load_config
from jobrec.domain.enums import RunMode

CATALOG_PATH = "data/processed/jobs.jsonl"

#: A profile that already states a long-term salary floor. This matters: conflict
#: detection only compares a stated salary against an EXISTING long-term value, so a
#: profile without ``salary_min`` never reaches the crash site.
PROFILE = {
    "candidate_id": "hybrid-c1",
    "skills": ["Python", "SQL"],
    "years_experience": 1,
    "preferred_locations": ["Kuala Lumpur"],
    "work_modes": ["hybrid"],
    "salary_min": 3500,
    "salary_currency": "MYR",
}

UTTERANCE = "I want a data analyst role in Kuala Lumpur, hybrid is fine, at least RM4000."


@pytest.fixture()
def hybrid_svc() -> AppService:
    cfg = load_config("configs/hybrid.yaml", base_dir="configs")
    assert cfg.llm.mode == RunMode.HYBRID, "configs/hybrid.yaml no longer selects hybrid mode"
    assert cfg.llm.provider == "mock", "this test relies on the offline mock provider"
    return AppService(cfg, CATALOG_PATH)


def test_hybrid_turn_succeeds_with_the_stated_salary_constraint_applied(hybrid_svc):
    """A hybrid-mode turn completes and the stated RM4000 floor still filters."""
    hybrid_svc.create_candidate(PROFILE)
    sid = hybrid_svc.create_session(PROFILE["candidate_id"], "full")

    res = hybrid_svc.process_turn(sid, UTTERANCE)

    assert res.run_record.success, res.response.message
    assert res.run_record.failure_code is None
    assert res.response.response_type == "recommendation"

    # R8.9: the stated constraint reached the active search as a hard constraint.
    active = res.active_search_state
    assert active.salary_min == 4000.0
    assert active.salary_currency == "MYR"
    assert "salary_min" in active.hard_constraint_fields
    assert res.decision is not None and res.decision.selected_job_ids

    # The model-shaped path really ran: the validated preference carries the canonical
    # salary structure, and the value the domain sees is projected from it.
    salary_prefs = [
        p for p in res.extracted_preferences.preferences if p.field_name == "salary_min"
    ]
    assert salary_prefs, "hybrid extraction produced no salary_min preference"
    assert salary_prefs[-1].normalized_value["min_salary"] == 4000.0
    assert res.run_record.model_manifest["provider"] == "mock"


def test_hybrid_long_term_write_back_stores_a_scalar_salary(hybrid_svc):
    """A durable hybrid statement writes a scalar salary that later turns can compare."""
    hybrid_svc.create_candidate({"candidate_id": "hybrid-c2", "skills": ["Python"],
                                 "years_experience": 2})
    sid = hybrid_svc.create_session("hybrid-c2", "full")

    first = hybrid_svc.process_turn(
        sid, "From now on I only want data analyst roles paying at least RM6000 per month."
    )
    assert first.run_record.success, first.response.message
    written = first.candidate_state.salary_min
    assert written is not None, "durable salary statement was not written to long-term memory"
    # Long-term memory keeps ONE salary shape (the scalar that create_candidate_state
    # writes from a profile), not the extraction-time structure.
    assert isinstance(written.value, float) and written.value == 6000.0

    # The next turn compares against that long-term value without failing.
    second = hybrid_svc.process_turn(sid, "Kuala Lumpur is fine, at least RM4000 this time.")
    assert second.run_record.success, second.response.message
    assert second.active_search_state.salary_min == 4000.0
