"""Unit tests for the pure scope-resolution helper (R4.5, R4.7)."""

from __future__ import annotations

import pytest

from jobrec.agents.memory_agent import resolve_scope
from jobrec.domain.enums import (
    ConfirmationStatus,
    ConstraintStrength,
    PersistenceScope,
)
from jobrec.domain.extraction import ExtractedPreference


def _pref(persistence_scope: PersistenceScope, temporal_scope: str) -> ExtractedPreference:
    return ExtractedPreference(
        field_name="preferred_locations",
        normalized_value="Kuala Lumpur",
        raw_text="I want to work in KL",
        confidence=0.9,
        confirmation_status=ConfirmationStatus.CONFIRMED,
        persistence_scope=persistence_scope,
        proposed_strength=ConstraintStrength.HARD,
        temporal_scope=temporal_scope,
    )


def test_long_term_temporal_resolves_long_term():
    p = _pref(PersistenceScope.ACTIVE_SEARCH, "long_term")
    assert resolve_scope(p) is PersistenceScope.LONG_TERM


def test_current_search_never_long_term_even_if_persistence_says_so():
    # persistence_scope says LONG_TERM, but "this time only" wins.
    p = _pref(PersistenceScope.LONG_TERM, "current_search")
    assert resolve_scope(p) is PersistenceScope.ACTIVE_SEARCH


@pytest.mark.parametrize(
    "persistence_scope",
    [
        PersistenceScope.LONG_TERM,
        PersistenceScope.SESSION,
        PersistenceScope.ACTIVE_SEARCH,
        PersistenceScope.TURN_ONLY,
    ],
)
@pytest.mark.parametrize("temporal_scope", ["session", "unknown"])
def test_session_and_unknown_fall_back_to_persistence_scope(persistence_scope, temporal_scope):
    p = _pref(persistence_scope, temporal_scope)
    assert resolve_scope(p) is persistence_scope


def test_helper_is_pure_and_does_not_mutate_input():
    p = _pref(PersistenceScope.SESSION, "current_search")
    before = p.model_dump()
    resolve_scope(p)
    assert p.model_dump() == before
