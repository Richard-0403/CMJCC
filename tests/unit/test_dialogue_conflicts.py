"""Unit tests for dialogue conflict detection and resolution (R21).

R21.1 requires a dialogue-conflict suite covering the three conflict types the
domain model names for candidate/profile disagreement, together with the
resolution each one carries:

===================  ==========================  ===========================
conflict_type        fields                      resolution
===================  ==========================  ===========================
``value_mismatch``   years_experience            ``ask_clarification``
                     experience_level            ``use_current_for_search``
``temporal_override``preferred_locations,        ``use_current_for_search``
                     salary_min
``scope_mismatch``   work_modes                  ``merge_values``
===================  ==========================  ===========================

Each type is checked twice: once at classification level
(``MemoryAgent.detect_conflicts``) for ``conflict_type`` / ``impact`` /
``resolution`` / rule id / evidence binding, and once end-to-end through
``CMJCC.run`` so the resolution is shown to actually take effect — on the
``ActiveSearchState`` for this search only, never on long-term
``CandidateState``, with the conflict recorded on ``DialogueState.conflicts``.
"""

from __future__ import annotations

import pytest

from jobrec.agents.memory_agent import MemoryAgent
from jobrec.domain.dialogue import DialogueState, PreferenceConflict
from jobrec.domain.enums import (
    ConfirmationStatus,
    ConstraintStrength,
    PersistenceScope,
)
from jobrec.domain.extraction import ExtractedPreference, ExtractedPreferenceSet
from jobrec.evidence_store import EvidenceStore
from jobrec.orchestration.cmjcc import CMJCC, CMJCCInput

#: A profile that carries a prior long-term value for every conflict-bearing field.
_PROFILE = {
    "candidate_id": "c",
    "skills": ["Python"],
    "target_roles": ["Data Analyst"],
    "preferred_locations": ["Penang"],
    "salary_min": 4000.0,
    "salary_currency": "MYR",
    "work_modes": ["remote"],
    "years_experience": 1.0,
    "experience_level": "junior",
}


def _pref(
    field_name: str,
    value,
    *,
    strength: ConstraintStrength = ConstraintStrength.SOFT,
    polarity: str = "positive",
) -> ExtractedPreference:
    """A confirmed current-turn statement scoped to THIS search ("for now")."""
    return ExtractedPreference(
        field_name=field_name,
        normalized_value=value,
        raw_text=f"{field_name} is {value} for now",
        confidence=0.9,
        confirmation_status=ConfirmationStatus.CONFIRMED,
        persistence_scope=PersistenceScope.ACTIVE_SEARCH,
        proposed_strength=strength,
        polarity=polarity,  # type: ignore[arg-type]
        temporal_scope="current_search",
    )


def _detect(config, prefs: list[ExtractedPreference], profile: dict | None = None):
    """Classify ``prefs`` against the long-term profile.

    Returns ``(candidate_state, conflicts, evidence_by_field)``. Incoming evidence
    is registered for real (no stubs) so conflicts can be checked to bind both
    sides of the disagreement.
    """
    mem = MemoryAgent(EvidenceStore(), config)
    cand = mem.create_candidate_state(dict(profile or _PROFILE))
    extraction = ExtractedPreferenceSet(utterance_id="u1", preferences=prefs)
    evidence_by_field: dict[str, list[str]] = {}
    for item in mem.build_dialogue_evidence(extraction, "s", "turn-0"):
        evidence_by_field.setdefault(item.field_name, []).append(item.evidence_id)
    conflicts = mem.detect_conflicts(cand, extraction, evidence_by_field)
    return cand, conflicts, evidence_by_field


def _run(config, prefs: list[ExtractedPreference], profile: dict | None = None):
    """Run the full CMJCC turn for ``prefs``; returns ``(candidate_state, output)``."""
    store = EvidenceStore()
    mem = MemoryAgent(store, config)
    cand = mem.create_candidate_state(dict(profile or _PROFILE))
    dialogue = DialogueState(
        session_id="s", candidate_id=cand.candidate_id, version=1, turns=[]
    )
    dialogue = mem.append_turn(dialogue, "candidate", "current turn")
    out = CMJCC(store, config).run(CMJCCInput(
        candidate_state=cand,
        dialogue_state=dialogue,
        extracted_preferences=ExtractedPreferenceSet(utterance_id="u1", preferences=prefs),
        catalog_snapshot_id="snap",
        config=config,
        run_id="run-conflicts",
    ))
    return cand, out


def _only(conflicts: list[PreferenceConflict], field: str) -> PreferenceConflict:
    """The single conflict raised for ``field`` (exactly one must exist)."""
    matches = [c for c in conflicts if c.field_name == field]
    assert len(matches) == 1, f"expected exactly one {field} conflict, got {matches}"
    return matches[0]


def _assert_binds_both_sides(
    conflict: PreferenceConflict, existing_ids: list[str], incoming_ids: list[str]
) -> None:
    """A conflict must cite the long-term evidence AND the current-turn evidence."""
    assert conflict.existing_evidence_ids == existing_ids
    assert conflict.incoming_evidence_ids == incoming_ids
    assert conflict.existing_evidence_ids and conflict.incoming_evidence_ids


def _active_values(state, attr: str) -> list:
    return [pv.value for pv in getattr(state, attr) if pv.is_active]


# ---------------------------------------------------------------- value mismatch


def test_years_mismatch_is_value_mismatch_resolved_by_clarification(config):
    """A factual years-of-experience disagreement is a high-impact value mismatch."""
    cand, conflicts, incoming = _detect(config, [_pref("years_experience", 3.0)])

    conflict = _only(conflicts, "years_experience")
    assert conflict.conflict_type == "value_mismatch"
    assert conflict.impact == "high"
    assert conflict.resolution == "ask_clarification"
    assert conflict.resolution_rule_id == "conflict.years"
    _assert_binds_both_sides(
        conflict,
        list(cand.years_experience.evidence_ids),
        incoming["years_experience"],
    )


def test_experience_level_mismatch_is_value_mismatch_using_current_for_search(config):
    """A stated experience level that differs from the profile is a value mismatch."""
    cand, conflicts, incoming = _detect(config, [_pref("experience_level", "senior")])

    conflict = _only(conflicts, "experience_level")
    assert conflict.conflict_type == "value_mismatch"
    assert conflict.impact == "medium"
    assert conflict.resolution == "use_current_for_search"
    assert conflict.resolution_rule_id == "conflict.level"
    _assert_binds_both_sides(
        conflict,
        list(cand.experience_level.evidence_ids),
        incoming["experience_level"],
    )


def test_years_clarification_blocks_override_and_is_recorded(config):
    """``ask_clarification`` keeps the profile value and surfaces a question.

    The stated years are neither written to long-term memory nor pushed into the
    active search; the field is flagged for clarification instead.
    """
    cand, out = _run(config, [_pref("years_experience", 3.0)])

    conflict = _only(out.conflicts, "years_experience")
    # Recorded on the new DialogueState version, and tracked as unresolved.
    assert conflict in out.dialogue_state.conflicts
    assert "years_experience" in out.dialogue_state.unresolved_slots
    # Active search keeps the profile value; the field is flagged for clarification.
    assert out.active_search_state.years_experience == 1.0
    assert "years_experience" in out.active_search_state.clarification_required_fields
    # A clarification question is raised for exactly this conflict.
    assert out.clarification_action is not None
    assert out.clarification_action.target_fields == ["years_experience"]
    assert conflict.conflict_id in out.clarification_action.related_conflict_ids
    # Long-term memory is untouched.
    assert out.candidate_state is cand
    assert cand.years_experience.value == 1.0


def test_experience_level_conflict_overrides_active_search_only(config):
    """``use_current_for_search`` applies the stated level to this search only."""
    cand, out = _run(config, [_pref("experience_level", "senior")])

    conflict = _only(out.conflicts, "experience_level")
    assert conflict in out.dialogue_state.conflicts
    # A resolved (non-clarification) conflict is not an unresolved slot.
    assert "experience_level" not in out.dialogue_state.unresolved_slots
    # The stated value drives the search ...
    assert out.active_search_state.experience_level == "senior"
    # ... while long-term memory still holds the profile level.
    assert out.candidate_state is cand
    assert cand.experience_level.value == "junior"


# ------------------------------------------------------------- temporal override


@pytest.mark.parametrize(
    ("strength", "expected_impact"),
    [(ConstraintStrength.HARD, "high"), (ConstraintStrength.SOFT, "medium")],
)
def test_location_change_is_temporal_override_scaled_by_strength(
    config, strength, expected_impact
):
    """A new location is a temporal override whose impact follows constraint strength."""
    cand, conflicts, incoming = _detect(
        config, [_pref("preferred_locations", "Kuala Lumpur", strength=strength)]
    )

    conflict = _only(conflicts, "preferred_locations")
    assert conflict.conflict_type == "temporal_override"
    assert conflict.impact == expected_impact
    assert conflict.resolution == "use_current_for_search"
    assert conflict.resolution_rule_id == "conflict.location"
    _assert_binds_both_sides(
        conflict,
        [eid for pv in cand.preferred_locations for eid in pv.evidence_ids],
        incoming["preferred_locations"],
    )


def test_salary_change_is_temporal_override_with_low_impact(config):
    """A different stated salary floor is a low-impact temporal override."""
    cand, conflicts, incoming = _detect(config, [_pref("salary_min", 6000.0)])

    conflict = _only(conflicts, "salary_min")
    assert conflict.conflict_type == "temporal_override"
    assert conflict.impact == "low"
    assert conflict.resolution == "use_current_for_search"
    assert conflict.resolution_rule_id == "conflict.salary"
    _assert_binds_both_sides(
        conflict, list(cand.salary_min.evidence_ids), incoming["salary_min"]
    )


def test_temporal_override_replaces_profile_value_for_this_search_only(config):
    """A temporal override replaces (not merges) the profile value on the search."""
    cand, out = _run(config, [
        _pref("preferred_locations", "Kuala Lumpur"),
        _pref("salary_min", 6000.0),
    ])

    for field in ("preferred_locations", "salary_min"):
        conflict = _only(out.conflicts, field)
        assert conflict.conflict_type == "temporal_override"
        assert conflict in out.dialogue_state.conflicts
        assert field not in out.dialogue_state.unresolved_slots
    # The overridden values replace the profile ones for this search ...
    assert out.active_search_state.preferred_locations == ["Kuala Lumpur"]
    assert out.active_search_state.salary_min == 6000.0
    # ... and long-term memory still holds only the profile values.
    assert out.candidate_state is cand
    assert _active_values(cand, "preferred_locations") == ["Penang"]
    assert cand.salary_min.value == 4000.0


# ---------------------------------------------------------------- scope mismatch


def test_additional_work_mode_is_scope_mismatch_resolved_by_merge(config):
    """An extra acceptable work mode widens scope rather than contradicting it."""
    cand, conflicts, incoming = _detect(config, [_pref("work_modes", "hybrid")])

    conflict = _only(conflicts, "work_modes")
    assert conflict.conflict_type == "scope_mismatch"
    assert conflict.impact == "low"
    assert conflict.resolution == "merge_values"
    assert conflict.resolution_rule_id == "conflict.work_mode"
    _assert_binds_both_sides(
        conflict,
        [eid for pv in cand.work_modes for eid in pv.evidence_ids],
        incoming["work_modes"],
    )


def test_scope_mismatch_merges_into_the_active_search_only(config):
    """``merge_values`` keeps both modes on the search; long-term memory is unchanged."""
    cand, out = _run(config, [_pref("work_modes", "hybrid")])

    conflict = _only(out.conflicts, "work_modes")
    assert conflict in out.dialogue_state.conflicts
    assert "work_modes" not in out.dialogue_state.unresolved_slots
    # Additive: the profile mode survives alongside the new one (unlike an override).
    assert set(out.active_search_state.work_modes) == {"remote", "hybrid"}
    # Long-term memory keeps only the profile mode.
    assert out.candidate_state is cand
    assert _active_values(cand, "work_modes") == ["remote"]


# ------------------------------------------------------- no-conflict and mixture


def test_no_conflict_when_statements_agree_or_have_no_prior_value(config):
    """Agreement, sub-threshold year drift, negations and unmapped fields raise nothing."""
    _, conflicts, _ = _detect(config, [
        _pref("preferred_locations", "Penang"),  # same as profile
        _pref("salary_min", 4000.0),  # same as profile
        _pref("work_modes", "remote"),  # already accepted
        _pref("experience_level", "junior"),  # same as profile
        _pref("years_experience", 1.5),  # drift below the 1.0-year threshold
        _pref("industries", "banking"),  # no prior long-term value
        _pref("preferred_locations", "Kuala Lumpur", polarity="negative"),  # exclusion
    ])

    assert conflicts == []


def test_all_three_conflict_types_are_recorded_together(config):
    """One turn can raise all three types; each keeps its own resolution."""
    cand, out = _run(config, [
        _pref("years_experience", 3.0),
        _pref("experience_level", "senior"),
        _pref("preferred_locations", "Kuala Lumpur"),
        _pref("salary_min", 6000.0),
        _pref("work_modes", "hybrid"),
    ])

    resolved = {c.field_name: (c.conflict_type, c.resolution) for c in out.conflicts}
    assert resolved == {
        "years_experience": ("value_mismatch", "ask_clarification"),
        "experience_level": ("value_mismatch", "use_current_for_search"),
        "preferred_locations": ("temporal_override", "use_current_for_search"),
        "salary_min": ("temporal_override", "use_current_for_search"),
        "work_modes": ("scope_mismatch", "merge_values"),
    }
    # Every conflict is recorded on the new DialogueState version, which is bumped.
    assert [c.conflict_id for c in out.dialogue_state.conflicts] == [
        c.conflict_id for c in out.conflicts
    ]
    assert out.dialogue_state.version == 3  # 1 + append_turn + this turn
    # Only the clarification-resolved field is unresolved.
    assert out.dialogue_state.unresolved_slots == ["years_experience"]
    # Resolutions took effect together, and none of them reached long-term memory.
    active = out.active_search_state
    assert active.years_experience == 1.0
    assert active.experience_level == "senior"
    assert active.preferred_locations == ["Kuala Lumpur"]
    assert active.salary_min == 6000.0
    assert set(active.work_modes) == {"remote", "hybrid"}
    assert out.candidate_state is cand
    assert cand.version == 1
