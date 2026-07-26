"""Unit tests for MemoryAgent.apply_confirmed_updates R4.11 conflict guard.

R4.11: IF a candidate statement conflicts with an existing long-term value and
the conflict resolution is not ``override``, THEN the MemoryAgent SHALL NOT
overwrite the long-term value (and SHALL record the conflict). Because
``PreferenceConflict.resolution`` currently has no ``override`` member, any
field with a detected conflict is preserved; a confirmed long-term preference on
a non-conflicting field is still written.
"""

from __future__ import annotations

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
from jobrec.utils.time import utcnow


def _pref(field_name: str, value, *, confidence: float = 0.95) -> ExtractedPreference:
    """A confirmed, long-term-scoped preference eligible for write-back."""
    return ExtractedPreference(
        field_name=field_name,
        normalized_value=value,
        raw_text=f"from now on {field_name}={value}",
        confidence=confidence,
        confirmation_status=ConfirmationStatus.CONFIRMED,
        persistence_scope=PersistenceScope.LONG_TERM,
        proposed_strength=ConstraintStrength.SOFT,
        temporal_scope="long_term",
    )


def _conflict(field_name: str, resolution: str = "use_current_for_search") -> PreferenceConflict:
    return PreferenceConflict(
        conflict_id=f"cf-{field_name}",
        field_name=field_name,
        existing_evidence_ids=["ev-existing"],
        incoming_evidence_ids=["ev-incoming"],
        conflict_type="value_mismatch",
        impact="medium",
        resolution=resolution,  # type: ignore[arg-type]
        resolution_rule_id="rule.test",
        created_at=utcnow(),
    )


def test_confirmed_long_term_on_conflicting_field_is_not_written(config):
    """A confirmed long-term write is skipped when its field has a conflict."""
    mem = MemoryAgent(store=EvidenceStore(), config=config)
    cand = mem.create_candidate_state(
        {"candidate_id": "c", "preferred_locations": ["Penang"]}
    )
    extraction = ExtractedPreferenceSet(
        utterance_id="u1",
        preferences=[_pref("preferred_locations", "Kuala Lumpur")],
    )
    conflicts = [_conflict("preferred_locations")]

    result = mem.apply_confirmed_updates(cand, extraction, conflicts)

    # Nothing writable -> same instance, version unchanged, value preserved.
    assert result is cand
    assert result.version == cand.version
    assert [p.value for p in result.preferred_locations if p.is_active] == ["Penang"]


def test_confirmed_long_term_on_non_conflicting_field_is_written(config):
    """A confirmed long-term write proceeds when its field has no conflict."""
    mem = MemoryAgent(store=EvidenceStore(), config=config)
    cand = mem.create_candidate_state(
        {"candidate_id": "c", "work_modes": ["remote"]}
    )
    extraction = ExtractedPreferenceSet(
        utterance_id="u1",
        preferences=[_pref("work_modes", "hybrid")],
    )

    result = mem.apply_confirmed_updates(cand, extraction, conflicts=[])

    # New version written; the new value is active.
    assert result is not cand
    assert result.version == cand.version + 1
    active_modes = {p.value for p in result.work_modes if p.is_active}
    assert "hybrid" in active_modes
    # Input is never mutated.
    assert cand.version == 1
    assert {p.value for p in cand.work_modes} == {"remote"}


def test_guard_skips_only_conflicting_field_and_writes_the_rest(config):
    """In one call, a conflicting field is preserved while a clean field is written."""
    mem = MemoryAgent(store=EvidenceStore(), config=config)
    cand = mem.create_candidate_state(
        {"candidate_id": "c", "preferred_locations": ["Penang"], "work_modes": ["remote"]}
    )
    extraction = ExtractedPreferenceSet(
        utterance_id="u1",
        preferences=[
            _pref("preferred_locations", "Kuala Lumpur"),  # conflicting -> blocked
            _pref("work_modes", "hybrid"),  # clean -> written
        ],
    )
    conflicts = [_conflict("preferred_locations")]

    result = mem.apply_confirmed_updates(cand, extraction, conflicts)

    assert result.version == cand.version + 1
    # Conflicting long-term value preserved (Kuala Lumpur never written).
    assert [p.value for p in result.preferred_locations if p.is_active] == ["Penang"]
    # Clean field written.
    assert "hybrid" in {p.value for p in result.work_modes if p.is_active}



def test_cmjcc_writes_back_from_now_on_statement_under_full_variant(config):
    """A confirmed "from now on ..." durable statement bumps the CandidateState
    version through the CMJCC path under the full variant (R4.2/4.6/4.8/4.9).

    The full variant resolves ``persist_confirmed_updates`` and
    ``use_persistent_memory`` (and ``use_current_turn``) to True, so CMJCC.run
    threads a new CandidateState version onto its output.
    """
    store = EvidenceStore()
    mem = MemoryAgent(store, config)
    # Candidate has no work_modes set, so the durable statement introduces a new
    # long-term value with no conflicting prior value.
    cand = mem.create_candidate_state(
        {"candidate_id": "c", "target_roles": ["data analyst"]}
    )
    dialogue = DialogueState(session_id="s", candidate_id="c", version=1, turns=[])
    extraction = ExtractedPreferenceSet(
        utterance_id="u1",
        preferences=[_pref("work_modes", "hybrid")],  # "from now on ..." -> long_term
    )

    cmjcc = CMJCC(store, config)
    out = cmjcc.run(CMJCCInput(
        candidate_state=cand,
        dialogue_state=dialogue,
        extracted_preferences=extraction,
        catalog_snapshot_id="cat-test",
        config=config,
        run_id="run-test",
    ))

    # CandidateState version incremented and carried on the output.
    assert out.candidate_state is not cand
    assert out.candidate_state.version == cand.version + 1
    assert "hybrid" in {p.value for p in out.candidate_state.work_modes if p.is_active}
    # The write-back is recorded as an evidence-log line on the single code path.
    assert any(
        e.event_type == "candidate_state_written" for e in out.evidence_log_entries
    )
    # Input candidate is never mutated.
    assert cand.version == 1
    assert cand.work_modes == []
