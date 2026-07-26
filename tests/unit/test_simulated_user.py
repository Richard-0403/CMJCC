"""Unit tests for the SimulatedUser clarification answerer (R7.1)."""

from __future__ import annotations

from datetime import UTC, datetime

from jobrec.agents.candidate_understanding import CandidateUnderstandingAgent
from jobrec.domain.dialogue import ClarificationAction
from jobrec_eval.scenarios import Scenario
from jobrec_eval.simulated_user import SimulatedUser


def _clar(target_fields, reason_code="missing_role_target") -> ClarificationAction:
    return ClarificationAction(
        clarification_id="clar-1",
        target_fields=list(target_fields),
        reason_code=reason_code,
        priority_score=0.9,
        question_text="What role are you looking for?",
        options=[],
        related_conflict_ids=[],
        created_at=datetime.now(UTC),
    )


def test_answers_target_roles_from_default_when_acceptable() -> None:
    scenario = {
        "scenario_id": "SC-B-01",
        "acceptable_slots": ["target_roles"],
        "profile": {"skills": ["Python"], "years_experience": 2},
        "expects": {"response_type": "clarification"},
    }
    user = SimulatedUser(scenario)
    result = user.answer(_clar(["target_roles"]))
    assert result is not None
    utterance, slot = result
    assert slot == "target_roles"
    # The utterance must re-extract into target_roles via the rule extractor.
    extracted = CandidateUnderstandingAgent().extract(utterance)
    assert any(p.field_name == "target_roles" for p in extracted.preferences)


def test_prefers_profile_value_over_default() -> None:
    scenario = {
        "scenario_id": "SC-X",
        "acceptable_slots": ["preferred_locations"],
        "profile": {"preferred_locations": ["Penang"]},
    }
    user = SimulatedUser(scenario)
    result = user.answer(_clar(["preferred_locations"], reason_code="clarification_required_field"))
    assert result is not None
    utterance, slot = result
    assert slot == "preferred_locations"
    assert "Penang" in utterance


def test_returns_none_when_slot_not_answerable() -> None:
    scenario = {
        "scenario_id": "SC-Y",
        "acceptable_slots": [],  # nothing declared answerable
        "profile": {"skills": ["Python"]},
    }
    user = SimulatedUser(scenario)
    # salary_currency is neither an acceptable slot nor present in the profile.
    assert user.answer(_clar(["salary_currency"], "ambiguous_salary_currency")) is None


def test_returns_none_for_empty_target_fields() -> None:
    user = SimulatedUser({"acceptable_slots": ["target_roles"], "profile": {}})
    assert user.answer(_clar([])) is None


def test_prefers_unasked_answerable_slot() -> None:
    scenario = {
        "acceptable_slots": ["target_roles", "preferred_locations"],
        "profile": {},
    }
    user = SimulatedUser(scenario)
    result = user.answer(
        _clar(["target_roles", "preferred_locations"]), asked_slots={"target_roles"}
    )
    assert result is not None
    _, slot = result
    assert slot == "preferred_locations"


def test_accepts_scenario_dataclass() -> None:
    scenario = Scenario(
        scenario_id="SC-B-01",
        scenario_type="clarification",
        difficulty="medium",
        memory_dependency="none",
        context_dependency="low",
        no_match_expected=False,
        clarification_expected=True,
        acceptable_slots=["target_roles"],
    )
    user = SimulatedUser(scenario)
    result = user.answer(_clar(["target_roles"]))
    assert result is not None
    assert result[1] == "target_roles"
