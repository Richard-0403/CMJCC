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


# ---------------------------------------------------------------------------
# A scenario may DECLARE the answer to its own clarification.
#
# Without this, an ambiguous-role scenario is answered from a single global default:
# SC-G-02 asks for "an analyst position of some sort in Penang" and was answered
# "data analyst" from ``_DEFAULTS``. The relevance oracle then graded that scenario
# against a constant living in the evaluation harness rather than against anything the
# scenario states -- and the harness's answer and the oracle's reference had no mechanism
# forcing them to agree. Declaring the answer makes it one reviewable input that both read.
# ---------------------------------------------------------------------------

def test_declared_clarification_answer_wins_over_the_global_default() -> None:
    """The declaration is used verbatim, not the domain default.

    **Validates: Requirements 33.1**
    """
    scenario = {
        "scenario_id": "SC-G-02",
        "acceptable_slots": ["target_roles"],
        "profile": {"skills": ["Python"], "years_experience": 3},
        "reference": {"clarification_answer": {"target_roles": "business analyst"}},
    }
    utterance, slot = SimulatedUser(scenario).answer(_clar(["target_roles"]))

    assert slot == "target_roles"
    assert "business analyst" in utterance.lower()
    assert "data analyst" not in utterance.lower()
    # It still has to be re-extractable, or the answer never reaches the active search.
    extracted = CandidateUnderstandingAgent().extract(utterance)
    assert any(p.field_name == "target_roles" for p in extracted.preferences)


def test_declared_answer_wins_over_the_profile_too() -> None:
    """Declaration beats profile: the profile is what the candidate already had on file,
    the declaration is what this scenario says they answer when asked.

    **Validates: Requirements 33.1**
    """
    scenario = {
        "scenario_id": "SC-X",
        "acceptable_slots": ["preferred_locations"],
        "profile": {"preferred_locations": ["Penang"]},
        "reference": {"clarification_answer": {"preferred_locations": "Kuala Lumpur"}},
    }
    utterance, _slot = SimulatedUser(scenario).answer(
        _clar(["preferred_locations"], reason_code="clarification_required_field"))

    assert "Kuala Lumpur" in utterance
    assert "Penang" not in utterance


def test_a_declared_answer_makes_an_otherwise_unanswerable_slot_answerable() -> None:
    """Declaring an answer is sufficient on its own.

    An ambiguous-role scenario has no profile role by design -- that is the ambiguity --
    so without the declaration the slot depends on a global default existing for it.

    **Validates: Requirements 33.1**
    """
    scenario = {
        "scenario_id": "SC-Z",
        "acceptable_slots": [],
        "profile": {},
        "reference": {"clarification_answer": {"employment_types": "contract"}},
    }
    result = SimulatedUser(scenario).answer(
        _clar(["employment_types"], reason_code="clarification_required_field"))
    assert result is not None
    assert "contract" in result[0].lower()


def test_no_declaration_keeps_the_previous_behaviour() -> None:
    """A scenario without a declaration behaves exactly as before.

    This is what keeps the change safe for every scenario set that has not been declared.

    **Validates: Requirements 33.1**
    """
    scenario = {
        "scenario_id": "SC-B-01",
        "acceptable_slots": ["target_roles"],
        "profile": {"skills": ["Python"], "years_experience": 2},
        "reference": {"hard": ["salary_min"]},  # a declaration WITHOUT an answer block
    }
    utterance, slot = SimulatedUser(scenario).answer(_clar(["target_roles"]))
    assert slot == "target_roles"
    assert "data analyst" in utterance.lower()


def test_a_scenario_dataclass_without_a_reference_attribute_is_tolerated() -> None:
    """``Scenario`` has no ``reference`` field; the lookup must not raise on it.

    **Validates: Requirements 33.1**
    """
    scenario = Scenario(
        scenario_id="SC-DC", scenario_type="clarification", difficulty="easy",
        memory_dependency="none", context_dependency="low", no_match_expected=False,
        clarification_expected=True, acceptable_slots=["target_roles"],
        expected_response="clarification",
    )
    result = SimulatedUser(scenario).answer(_clar(["target_roles"]))
    assert result is not None and result[1] == "target_roles"
