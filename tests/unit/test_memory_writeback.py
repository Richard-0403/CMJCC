"""Unit tests for MemoryAgent.apply_confirmed_updates R4.11 conflict guard.

R4.11: IF a candidate statement conflicts with an existing long-term value and
the conflict resolution is not ``override``, THEN the MemoryAgent SHALL NOT
overwrite the long-term value (and SHALL record the conflict). Because
``PreferenceConflict.resolution`` currently has no ``override`` member, any
field with a detected conflict is preserved; a confirmed long-term preference on
a non-conflicting field is still written.
"""

from __future__ import annotations

from datetime import timedelta
from typing import get_args

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from jobrec.agents.memory_agent import (
    _LIST_FIELDS,
    _SCALAR_FIELDS,
    MemoryAgent,
    resolve_scope,
)
from jobrec.domain.dialogue import DialogueState, PreferenceConflict
from jobrec.domain.enums import (
    ConfirmationStatus,
    ConstraintStrength,
    EvidenceSource,
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

# ---------------------------------------------------------------------------
# Property-based test (Property 1)
# ---------------------------------------------------------------------------

#: Writable CandidateState fields with a small set of well-typed, in-domain values.
#: Kept as an explicit domain so the generator never has to filter invalid inputs.
_LIST_FIELD_VALUES = {
    "skills_have": ["python", "sql", "excel", "tableau"],
    "target_roles": ["data analyst", "data engineer", "bi analyst"],
    "preferred_locations": ["Kuala Lumpur", "Penang", "Johor Bahru"],
    "work_modes": ["remote", "hybrid", "onsite"],
    "industries": ["banking", "retail", "technology"],
    "employment_types": ["full_time", "contract", "internship"],
    "work_authorizations": ["citizen", "permanent_resident"],
    "excluded_roles": ["sales executive", "call centre agent"],
    "excluded_locations": ["Sabah", "Sarawak"],
    "excluded_industries": ["gambling", "tobacco"],
}
_SCALAR_FIELD_VALUES = {
    "years_experience": [0.0, 1.0, 3.5, 8.0],
    "experience_level": ["entry", "junior", "mid", "senior"],
    "salary_min": [3000.0, 4500.0, 9000.0],
    "salary_currency": ["MYR", "SGD", "USD"],
    "education_level": ["diploma", "bachelor", "master"],
}
#: Flat (field, value) domain covering both list-valued and scalar CandidateState fields.
_FIELD_VALUE_PAIRS = [
    (field, value)
    for values in (_LIST_FIELD_VALUES, _SCALAR_FIELD_VALUES)
    for field, vals in values.items()
    for value in vals
]

#: A pre-populated profile so write-back also exercises the supersede paths, and an
#: empty one so it also exercises first-write paths.
_PROFILES = [
    {"candidate_id": "c"},
    {
        "candidate_id": "c",
        "skills": ["python"],
        "target_roles": ["data analyst"],
        "preferred_locations": ["Penang"],
        "work_modes": ["remote"],
        "years_experience": 2.0,
        "experience_level": "junior",
        "salary_min": 4000.0,
        "salary_currency": "MYR",
        "education_level": "bachelor",
    },
]

#: Reasons a confirmed preference does NOT resolve to a durable long-term write.
_BLOCKERS = ["current_search_scope", "unconfirmed", "low_confidence", "unknown_field"]


def _non_durable(pref: ExtractedPreference, blocker: str, threshold: float) -> ExtractedPreference:
    """Return a copy of ``pref`` that must never reach long-term memory."""
    if blocker == "current_search_scope":
        return pref.model_copy(update={
            "temporal_scope": "current_search",
            "persistence_scope": PersistenceScope.ACTIVE_SEARCH,
        })
    if blocker == "unconfirmed":
        return pref.model_copy(
            update={"confirmation_status": ConfirmationStatus.UNCONFIRMED}
        )
    if blocker == "low_confidence":
        return pref.model_copy(update={"confidence": max(0.0, threshold * 0.5)})
    return pref.model_copy(update={"field_name": "favourite_colour"})


# Feature: cmjcc-experiment-readiness, Property 1: Long-term write-back increments version
# monotonically and never mutates input
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    pairs=st.lists(st.sampled_from(_FIELD_VALUE_PAIRS), min_size=1, max_size=3),
    profile=st.sampled_from(_PROFILES),
    confidence=st.floats(min_value=0.72, max_value=1.0),
    blocker=st.sampled_from(_BLOCKERS),
)
def test_property_write_back_increments_version_and_never_mutates_input(
    config, pairs, profile, confidence, blocker
) -> None:
    """A durable write returns version + 1 (monotonic) and leaves the input untouched.

    **Validates: Requirements 4.2**
    """
    mem = MemoryAgent(store=EvidenceStore(), config=config)
    cand = mem.create_candidate_state(dict(profile))
    before = cand.model_copy(deep=True)
    extraction = ExtractedPreferenceSet(
        utterance_id="u1",
        preferences=[_pref(f, v, confidence=confidence) for f, v in pairs],
    )

    first = mem.apply_confirmed_updates(cand, extraction, conflicts=[])

    # A durable write yields a NEW state whose version is exactly input + 1 ...
    assert first is not cand
    assert first.version == cand.version + 1
    # ... and the input instance is untouched (version and every field value).
    assert cand == before

    # Version increases monotonically across repeated applications.
    first_before = first.model_copy(deep=True)
    second = mem.apply_confirmed_updates(first, extraction, conflicts=[])
    assert second.version == first.version + 1
    assert second.version > first.version > cand.version
    assert first == first_before

    # Nothing durable -> the SAME instance is returned, version unchanged.
    threshold = config.memory.clarification_confidence_threshold
    blocked = ExtractedPreferenceSet(
        utterance_id="u2",
        preferences=[
            _non_durable(_pref(f, v, confidence=confidence), blocker, threshold)
            for f, v in pairs
        ],
    )
    second_before = second.model_copy(deep=True)

    unchanged = mem.apply_confirmed_updates(second, blocked, conflicts=[])

    assert unchanged is second
    assert unchanged.version == second.version
    assert second == second_before

# ---------------------------------------------------------------------------
# Property-based test (Property 2)
# ---------------------------------------------------------------------------


def _assert_supersede_invariants(prior, result, pairs, now) -> None:
    """Every value replaced by the write is retired; the new value is the only active one."""
    for field, value in pairs:
        if field in _SCALAR_FIELDS:
            attr = _SCALAR_FIELDS[field]
            written = getattr(result, attr)
            # The newly written scalar is the active value, open-ended.
            assert written is not None
            assert written.value == value
            assert written.is_active is True
            assert written.effective_to is None
            assert written.effective_from == now

            history = result.metadata.get("superseded", {}).get(field, [])
            prior_history = prior.metadata.get("superseded", {}).get(field, [])
            prior_pv = getattr(prior, attr)
            if prior_pv is None:
                # Nothing to supersede -> no new history entry.
                assert len(history) == len(prior_history)
            else:
                assert len(history) == len(prior_history) + 1
                retired = history[-1]
                assert retired.value == prior_pv.value
                assert retired.is_active is False
                assert retired.effective_to == now
            # The whole retained history stays deactivated and closed off.
            assert all(h.is_active is False and h.effective_to is not None for h in history)
        else:
            attr = _LIST_FIELDS[field]
            prior_entries = list(getattr(prior, attr))
            entries = list(getattr(result, attr))
            # Exactly one write -> exactly one appended record.
            assert len(entries) == len(prior_entries) + 1

            same_value = [pv for pv in entries if pv.value == value]
            active_same = [pv for pv in same_value if pv.is_active]
            inactive_same = [pv for pv in same_value if not pv.is_active]
            # The new value is the single active record for that value.
            assert len(active_same) == 1
            assert active_same[0].effective_to is None
            assert active_same[0].effective_from == now
            # Every prior record carrying that value is now deactivated and closed off.
            prior_same = [pv for pv in prior_entries if pv.value == value]
            assert len(inactive_same) == len(prior_same)
            assert all(pv.effective_to is not None for pv in inactive_same)
            # The ones retired by THIS write are stamped with this update time.
            prior_active_same = [pv for pv in prior_same if pv.is_active]
            assert sum(pv.effective_to == now for pv in inactive_same) == len(prior_active_same)


# Feature: cmjcc-experiment-readiness, Property 2: Superseded values are deactivated with an
# effective_to timestamp
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    pairs=st.lists(
        st.sampled_from(_FIELD_VALUE_PAIRS),
        min_size=1,
        max_size=3,
        unique_by=lambda pair: pair[0],
    ),
    profile=st.sampled_from(_PROFILES),
    confidence=st.floats(min_value=0.72, max_value=1.0),
)
def test_property_superseded_values_are_deactivated_with_effective_to(
    config, pairs, profile, confidence
) -> None:
    """Overwritten values become inactive with ``effective_to == now``; the new value is active.

    **Validates: Requirements 4.3**
    """
    mem = MemoryAgent(store=EvidenceStore(), config=config)
    cand = mem.create_candidate_state(dict(profile))
    extraction = ExtractedPreferenceSet(
        utterance_id="u1",
        preferences=[_pref(f, v, confidence=confidence) for f, v in pairs],
    )

    first_at = utcnow()
    first = mem.apply_confirmed_updates(cand, extraction, conflicts=[], now=first_at)
    _assert_supersede_invariants(cand, first, pairs, first_at)

    # Re-applying the same durable write guarantees every generated field has a prior
    # active value, so the supersede path is exercised for all of them.
    second_at = first_at + timedelta(minutes=5)
    second = mem.apply_confirmed_updates(first, extraction, conflicts=[], now=second_at)
    _assert_supersede_invariants(first, second, pairs, second_at)

# ---------------------------------------------------------------------------
# Property-based test (Property 3)
# ---------------------------------------------------------------------------


def _records_for_field(state, field: str) -> list:
    """Every PreferenceValue record held for ``field``, including retired history."""
    if field in _SCALAR_FIELDS:
        pv = getattr(state, _SCALAR_FIELDS[field])
        records = [] if pv is None else [pv]
        records.extend(state.metadata.get("superseded", {}).get(field, []))
        return records
    return list(getattr(state, _LIST_FIELDS[field]))


def _assert_resolves_to_long_term_evidence(store: EvidenceStore, pv) -> None:
    """A written value must carry at least one id that resolves to long-term evidence."""
    assert pv.evidence_ids, "long-term value written with no evidence id"
    for evidence_id in pv.evidence_ids:
        item = store.get(evidence_id)
        # No dangling ids: the id resolves to an actually-registered EvidenceItem ...
        assert item is not None, f"dangling evidence id {evidence_id}"
        assert store.exists(evidence_id)
        assert item.evidence_id == evidence_id
        # ... which itself is scoped to long-term persistence.
        assert item.persistence_scope == PersistenceScope.LONG_TERM


# Feature: cmjcc-experiment-readiness, Property 3: Every long-term write is evidence-bound
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    pairs=st.lists(
        st.sampled_from(_FIELD_VALUE_PAIRS),
        min_size=1,
        max_size=3,
        unique_by=lambda pair: pair[0],
    ),
    profile=st.sampled_from(_PROFILES),
    confidence=st.floats(min_value=0.72, max_value=1.0),
)
def test_property_every_long_term_write_is_evidence_bound(
    config, pairs, profile, confidence
) -> None:
    """Each written long-term value carries evidence ids resolving to registered items.

    **Validates: Requirements 4.4, 4.10, 31.4**
    """
    store = EvidenceStore()
    mem = MemoryAgent(store=store, config=config)
    cand = mem.create_candidate_state(dict(profile))
    extraction = ExtractedPreferenceSet(
        utterance_id="u1",
        preferences=[_pref(f, v, confidence=confidence) for f, v in pairs],
    )

    first = mem.apply_confirmed_updates(cand, extraction, conflicts=[])
    # A second application exercises the supersede paths, whose retired records must
    # stay evidence-bound too.
    second = mem.apply_confirmed_updates(first, extraction, conflicts=[])

    for state in (first, second):
        for field, value in pairs:
            written = [pv for pv in _records_for_field(state, field) if pv.value == value]
            active = [pv for pv in written if pv.is_active]
            # Exactly one active record per written value, and it is evidence-bound.
            assert len(active) == 1
            _assert_resolves_to_long_term_evidence(store, active[0])
            # The bound evidence describes this very field/value and traces to dialogue
            # (an utterance or clarification), not to an unrelated source.
            item = store.get(active[0].evidence_ids[0])
            assert item is not None
            assert item.source == EvidenceSource.DIALOGUE
            assert item.field_name == field
            assert item.normalized_value == value
            assert item.confirmation_status == ConfirmationStatus.CONFIRMED

        # No long-term record anywhere in the state lacks backing evidence.
        for field in (*_LIST_FIELDS, *_SCALAR_FIELDS):
            for pv in _records_for_field(state, field):
                if pv.persistence_scope == PersistenceScope.LONG_TERM:
                    _assert_resolves_to_long_term_evidence(store, pv)

# ---------------------------------------------------------------------------
# Property-based test (Property 4)
# ---------------------------------------------------------------------------

#: (persistence_scope, temporal_scope) pairs covering all four declared scopes and every
#: temporal cue, including the cases where the two disagree ("from now on" on an
#: active-search preference, "this time only" on a long-term one).
_SCOPE_COMBOS = [
    (PersistenceScope.LONG_TERM, "long_term"),
    (PersistenceScope.LONG_TERM, "current_search"),
    (PersistenceScope.LONG_TERM, "session"),
    (PersistenceScope.LONG_TERM, "unknown"),
    (PersistenceScope.ACTIVE_SEARCH, "long_term"),
    (PersistenceScope.ACTIVE_SEARCH, "current_search"),
    (PersistenceScope.ACTIVE_SEARCH, "session"),
    (PersistenceScope.ACTIVE_SEARCH, "unknown"),
    (PersistenceScope.SESSION, "session"),
    (PersistenceScope.SESSION, "unknown"),
    (PersistenceScope.TURN_ONLY, "session"),
    (PersistenceScope.TURN_ONLY, "unknown"),
]


def _expected_resolved_scope(persistence: PersistenceScope, temporal: str) -> PersistenceScope:
    """R4.5 scope resolution, restated independently of the implementation."""
    if temporal == "long_term":  # "from now on ..."
        return PersistenceScope.LONG_TERM
    if temporal == "current_search":  # "this time only ..."
        return PersistenceScope.ACTIVE_SEARCH
    return persistence


#: One generated preference: (field, value) x confirmation status x scope combo x
#: confidence drawn from both sides of the clarification threshold (0.72).
_PREF_SPECS = st.lists(
    st.tuples(
        st.sampled_from(_FIELD_VALUE_PAIRS),
        st.sampled_from(list(ConfirmationStatus)),
        st.sampled_from(_SCOPE_COMBOS),
        st.one_of(
            st.floats(min_value=0.72, max_value=1.0),
            st.floats(min_value=0.0, max_value=0.71),
        ),
    ),
    min_size=1,
    max_size=4,
    unique_by=lambda spec: spec[0][0],
)


# Feature: cmjcc-experiment-readiness, Property 4: Only long-term-scoped, confirmed
# preferences write to long-term memory
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(specs=_PREF_SPECS, profile=st.sampled_from(_PROFILES))
def test_property_only_confirmed_long_term_preferences_are_persisted(
    config, specs, profile
) -> None:
    """A long-term write happens iff the preference is confirmed, resolves to
    ``long_term`` scope, and meets the confidence threshold.

    Unconfirmed, current-search-scoped ("this time only"), session/turn-only and
    sub-threshold preferences leave the persisted values and the version untouched.

    **Validates: Requirements 4.5, 4.7, 4.9**
    """
    threshold = config.memory.clarification_confidence_threshold
    mem = MemoryAgent(store=EvidenceStore(), config=config)
    cand = mem.create_candidate_state(dict(profile))
    before = cand.model_copy(deep=True)

    preferences: list[ExtractedPreference] = []
    checks: list[tuple[str, object, bool]] = []
    for (field, value), status, (persistence, temporal), confidence in specs:
        pref = _pref(field, value, confidence=confidence).model_copy(update={
            "confirmation_status": status,
            "persistence_scope": persistence,
            "temporal_scope": temporal,
        })
        preferences.append(pref)
        resolved = _expected_resolved_scope(persistence, temporal)
        # The agent's scope resolution matches the specified rule (R4.5).
        assert resolve_scope(pref) == resolved
        durable = (
            status == ConfirmationStatus.CONFIRMED
            and resolved == PersistenceScope.LONG_TERM
            and confidence >= threshold
        )
        checks.append((field, value, durable))

    extraction = ExtractedPreferenceSet(utterance_id="u1", preferences=preferences)
    result = mem.apply_confirmed_updates(cand, extraction, conflicts=[])

    if any(durable for *_, durable in checks):
        # At least one qualifying preference -> exactly one new version (R4.6).
        assert result is not cand
        assert result.version == cand.version + 1
    else:
        # Nothing qualifies -> long-term memory is untouched (R4.7/4.9).
        assert result is cand
        assert result.version == cand.version
    # The input state is never mutated, whatever the outcome.
    assert cand == before

    for field, value, durable in checks:
        prior_records = _records_for_field(cand, field)
        records = _records_for_field(result, field)
        if durable:
            # Written: exactly one new record, active, long-term and evidence-bound.
            assert len(records) == len(prior_records) + 1
            active = [pv for pv in records if pv.is_active and pv.value == value]
            assert len(active) == 1
            assert active[0].persistence_scope == PersistenceScope.LONG_TERM
            assert active[0].confirmation_status == ConfirmationStatus.CONFIRMED
            assert active[0].confidence >= threshold
            assert active[0].evidence_ids
        else:
            # Not written: the field's long-term records are byte-for-byte unchanged,
            # so the non-durable value never reaches long-term memory.
            assert records == prior_records
            assert not [
                pv for pv in records if pv.value == value and pv not in prior_records
            ]

# ---------------------------------------------------------------------------
# Property-based test (Property 5)
# ---------------------------------------------------------------------------

#: Every resolution the domain model actually permits, derived from the model so the
#: generator stays exhaustive if the Literal ever gains an ``override`` member.
_CONFLICT_RESOLUTIONS = sorted(
    get_args(PreferenceConflict.model_fields["resolution"].annotation)
)

#: Per-field value domains, derived from the shared (field, value) domain so a prior and a
#: distinct incoming value can be drawn for the SAME field without duplicating domains.
_FIELD_DOMAINS: dict[str, list] = {}
for _field, _value in _FIELD_VALUE_PAIRS:
    _FIELD_DOMAINS.setdefault(_field, []).append(_value)

#: One generated field: (field, [prior_value, incoming_value], conflicting?, resolution).
#: Mixing ``conflicting`` per field puts blocked and clean fields in the same call.
_CONFLICT_SPECS = st.lists(
    st.sampled_from(sorted(_FIELD_DOMAINS)).flatmap(
        lambda field: st.tuples(
            st.just(field),
            st.lists(
                st.sampled_from(_FIELD_DOMAINS[field]),
                min_size=2,
                max_size=2,
                unique=True,
            ),
            st.booleans(),
            st.sampled_from(_CONFLICT_RESOLUTIONS),
        )
    ),
    min_size=2,
    max_size=4,
    unique_by=lambda spec: spec[0],
)


def _active_records(state, field: str, value) -> list:
    """Active PreferenceValue records held for ``field`` carrying exactly ``value``."""
    return [
        pv for pv in _records_for_field(state, field) if pv.is_active and pv.value == value
    ]


# Feature: cmjcc-experiment-readiness, Property 5: Non-override conflicts never overwrite
# long-term memory
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(specs=_CONFLICT_SPECS, profile=st.sampled_from(_PROFILES))
def test_property_non_override_conflicts_never_overwrite_long_term_memory(
    config, specs, profile
) -> None:
    """A field carrying a non-``override`` conflict keeps its existing long-term value,
    while non-conflicting fields in the same call are still written.

    ``PreferenceConflict.resolution`` has no ``override`` member (asserted below), so in
    practice every detected conflict blocks its field's write -- this is checked across
    all six permitted resolutions. The guard suppresses the write only: the conflict is
    not consumed and stays recordable on ``DialogueState.conflicts``.

    **Validates: Requirements 4.11**
    """
    # The implementation's `resolution != "override"` predicate is total today: no
    # resolution can be "override", so no conflict can license an overwrite.
    assert "override" not in _CONFLICT_RESOLUTIONS

    mem = MemoryAgent(store=EvidenceStore(), config=config)
    cand = mem.create_candidate_state(dict(profile))

    # Seed a prior long-term value for every generated field (no conflicts yet) so the
    # guard is exercised against an existing value that it must not overwrite.
    prior = mem.apply_confirmed_updates(
        cand,
        ExtractedPreferenceSet(
            utterance_id="u0",
            preferences=[_pref(field, values[0]) for field, values, _, _ in specs],
        ),
        conflicts=[],
    )
    prior_snapshot = prior.model_copy(deep=True)

    # Incoming values are fully durable (confirmed, long-term, above threshold), so a
    # conflict is the ONLY thing that can stop them being written.
    incoming = ExtractedPreferenceSet(
        utterance_id="u1",
        preferences=[_pref(field, values[1]) for field, values, _, _ in specs],
    )
    conflicts = [
        _conflict(field, resolution)
        for field, _, conflicting, resolution in specs
        if conflicting
    ]
    conflicts_snapshot = [c.model_copy(deep=True) for c in conflicts]

    result = mem.apply_confirmed_updates(prior, incoming, conflicts)

    if any(not conflicting for _, _, conflicting, _ in specs):
        # At least one clean field -> exactly one new version.
        assert result is not prior
        assert result.version == prior.version + 1
    else:
        # Every field blocked -> no write at all, not even a version bump.
        assert result is prior
        assert result.version == prior.version
    # The input state is never mutated, whatever the conflict mix.
    assert prior == prior_snapshot

    for field, values, conflicting, _resolution in specs:
        prior_value, incoming_value = values
        if conflicting:
            # Long-term memory for this field is byte-for-byte unchanged ...
            assert _records_for_field(result, field) == _records_for_field(prior, field)
            # ... and the pre-existing value is still the single active one, so the
            # incoming value never overwrote it.
            assert len(_active_records(result, field, prior_value)) == 1
        else:
            # No conflict on this field -> the incoming value is written and active.
            assert len(_active_records(result, field, incoming_value)) == 1

    # Conflicts are untouched by the guard and remain recordable on the DialogueState
    # (R4.11 "SHALL record the conflict").
    assert conflicts == conflicts_snapshot
    dialogue = DialogueState(
        session_id="s", candidate_id="c", version=1, turns=[], conflicts=conflicts
    )
    assert [c.conflict_id for c in dialogue.conflicts] == [c.conflict_id for c in conflicts]

# ---------------------------------------------------------------------------
# Cross-session inheritance (R19.1)
# ---------------------------------------------------------------------------
#
# The tests above exercise write-back, Persistence_Scope handling and versioning on
# a single CandidateState. R19.1 also requires cross-session inheritance: a value
# written to long-term memory in one session must be picked up by a NEW session for
# the same candidate, while a value scoped to the current search must not be.
#
# These run end-to-end through ``AppService`` on the in-memory repository (the
# ``service`` fixture; no PostgreSQL required), because inheritance is a property of
# the repository loader + session wiring, not of ``apply_confirmed_updates`` alone:
# ``create_session`` records the candidate id and ``process_turn`` reloads the
# candidate's LATEST persisted version for every new session.


def _active_values(state, field: str) -> list:
    """Active (non-retired) values held for ``field`` in a CandidateState."""
    return [pv.value for pv in _records_for_field(state, field) if pv.is_active]


def test_long_term_write_back_is_inherited_by_a_new_session(service):
    """A durable value written in session 1 is loaded by a NEW session (R19.1).

    The second session starts with an empty dialogue, so the only channel through
    which "hybrid" can reach its active search is long-term candidate memory.
    """
    service.create_candidate(
        {"candidate_id": "c-cross", "skills": ["Python", "SQL"], "years_experience": 1}
    )
    first = service.create_session("c-cross", "full")

    # Session 1: a durable ("from now on ...") statement -> long-term write-back.
    r1 = service.process_turn(first, "From now on I only want hybrid roles.")
    assert r1.candidate_state.version == 2
    assert "hybrid" in _active_values(r1.candidate_state, "work_modes")

    # A NEW session for the SAME candidate.
    second = service.create_session("c-cross", "full")
    assert second != first

    r2 = service.process_turn(second, "Show me data analyst jobs in Kuala Lumpur.")

    # The new session has its own fresh dialogue -> nothing is carried over in-session.
    assert r2.dialogue_state.session_id == second
    assert len(r2.dialogue_state.turns) == 1
    # It nevertheless loaded the version written by session 1 (no re-write: this
    # utterance carries no durable statement, so the version is unchanged).
    assert r2.active_search_state.candidate_state_version == 2
    assert r2.candidate_state.version == 2
    assert "hybrid" in _active_values(r2.candidate_state, "work_modes")
    # ... and the inherited value shapes the new session's search even though the
    # utterance never mentions a work mode.
    assert "hybrid" in r2.active_search_state.work_modes

    # Version history is preserved: v1 (pre-write-back) is still retrievable and the
    # loader hands a new session the LATEST version.
    assert service.get_candidate("c-cross", version=1).work_modes == []
    assert service.get_candidate("c-cross").version == 2


def test_current_search_scoped_value_is_not_inherited_by_a_new_session(service):
    """A "this time only" value never persists, so a new session does not see it.

    The scope contrast to the test above: the same pipeline, the same fresh-session
    load, but a current-search-scoped statement leaves long-term memory untouched
    (R19.1 Persistence_Scope handling across sessions).
    """
    service.create_candidate(
        {"candidate_id": "c-scoped", "skills": ["Python", "SQL"], "years_experience": 1}
    )
    first = service.create_session("c-scoped", "full")

    r1 = service.process_turn(
        first, "This time only, show me remote jobs paying at least RM8000."
    )
    # It applies to THIS search ...
    assert "remote" in r1.active_search_state.work_modes
    assert r1.active_search_state.salary_min == 8000.0
    # ... but nothing was written to long-term memory (no new version).
    assert r1.candidate_state.version == 1
    assert _active_values(r1.candidate_state, "work_modes") == []
    assert r1.candidate_state.salary_min is None

    second = service.create_session("c-scoped", "full")
    r2 = service.process_turn(second, "Show me data analyst jobs in Kuala Lumpur.")

    # The new session inherits nothing from the previous session's search scope.
    assert r2.active_search_state.candidate_state_version == 1
    assert "remote" not in r2.active_search_state.work_modes
    assert r2.active_search_state.salary_min is None
    assert _active_values(r2.candidate_state, "work_modes") == []
    assert service.get_candidate("c-scoped").version == 1


def test_long_term_memory_accumulates_across_sessions(service):
    """A second session's write-back builds on the version it inherited (R19.1).

    Session 2 writes a different field, so the value from session 1 must still be
    active on the new version: long-term memory accumulates across sessions instead
    of each session starting from the original profile.
    """
    service.create_candidate(
        {"candidate_id": "c-accum", "skills": ["Python", "SQL"], "years_experience": 1}
    )
    first = service.create_session("c-accum", "full")
    r1 = service.process_turn(first, "From now on I only want hybrid roles.")
    assert r1.candidate_state.version == 2

    second = service.create_session("c-accum", "full")
    r2 = service.process_turn(second, "From now on I want to work in Penang.")

    # Built on the inherited v2, not on the original v1.
    assert r2.candidate_state.version == 3
    assert "Penang" in _active_values(r2.candidate_state, "preferred_locations")
    # The session-1 value survived the session-2 write.
    assert "hybrid" in _active_values(r2.candidate_state, "work_modes")
    # Both durable values are evidence-bound long-term records.
    for field in ("work_modes", "preferred_locations"):
        for pv in _records_for_field(r2.candidate_state, field):
            assert pv.persistence_scope == PersistenceScope.LONG_TERM
            assert pv.evidence_ids

    # A third session inherits the accumulated state.
    third = service.create_session("c-accum", "full")
    r3 = service.process_turn(third, "Show me data analyst jobs.")
    assert r3.active_search_state.candidate_state_version == 3
    assert "hybrid" in r3.active_search_state.work_modes
    assert "Penang" in r3.active_search_state.preferred_locations
