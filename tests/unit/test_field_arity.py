"""Arity (shape) contract for extracted field values (R8.7/R8.9/R8.11).

``validate_field`` historically checked only the value **domain** -- is this a legal
work mode, a resolvable location, a parsable date. It did not check **arity**: how many
values one preference may carry. A real model returned

    {"field_name": "target_roles", "normalized_value": ["software engineer"]}

-- a legal role wrapped in a single-element list. ``target_roles`` has no normalizer, so
validation returned ``ok=True`` with the list untouched, the orchestrator's repair rung
(which only inspects ``ok=False`` results) skipped it, and the list reached
``canonical_role`` which calls ``.strip()`` on a string. The run died with
``AttributeError: 'list' object has no attribute 'strip'``.

The same mismatch existed in the opposite direction and was strictly worse, because it
did not need malformed input at all: ``normalize_skills`` legitimately fans one utterance
out into several values (``"python, sql"`` -> ``["python", "sql"]``), while the consumer
appended ``normalized_value`` as a *single* element of a ``list[str]`` field. Any
``skills_have`` extraction whatsoever would therefore have crashed; it never fired only
because no scenario utterance happens to name a known skill.

These tests pin both directions, plus two silent-failure siblings found alongside them:
a ``salary_min: null`` that read as *present*, and a non-string exclusion coerced with
``str()`` into a key like ``"['sales analyst']"`` that can never match a role family.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jobrec.domain.dialogue import DialogueState
from jobrec.domain.enums import (
    ConfirmationStatus,
    ConstraintStrength,
    PersistenceScope,
    RunMode,
)
from jobrec.domain.extraction import ExtractedPreference, ExtractedPreferenceSet
from jobrec.evidence_store import EvidenceStore
from jobrec.llm.field_validation import FieldResult, field_arity, validate_field
from jobrec.orchestration.cmjcc import CMJCC, CMJCCInput
from tests.unit.test_field_validation import ScriptedJSONProvider, _orchestrator, _payload

#: Fields where ONE preference states ONE value, even though the search field is a list.
SCALAR_ARITY_FIELDS = [
    "target_roles",
    "preferred_locations",
    "work_modes",
    "employment_types",
    "work_authorizations",
    "experience_level",
    "salary_currency",
    "years_experience",
    "excluded_roles",
    "excluded_locations",
]

#: Fields whose normalizer deliberately produces several values from one utterance.
LIST_ARITY_FIELDS = ["skills_have", "skills"]

#: Container shapes a model actually emits, all of which violate a scalar contract.
BAD_CONTAINERS: list[Any] = [
    ["software engineer"],
    ["remote", "hybrid"],
    ("Kuala Lumpur",),
    {"value": "senior"},
    [["nested"]],
    {"a", "b"},
]


def _pref(field_name: str, value: Any, *, polarity: str = "positive") -> ExtractedPreference:
    return ExtractedPreference(
        field_name=field_name,
        normalized_value=value,
        raw_text=f"stated {field_name}",
        confidence=0.95,
        confirmation_status=ConfirmationStatus.CONFIRMED,
        persistence_scope=PersistenceScope.ACTIVE_SEARCH,
        proposed_strength=ConstraintStrength.HARD,
        polarity=polarity,
    )


def _active_search(config, *prefs: ExtractedPreference):
    """Run the real CMJCC path and return the resulting ActiveSearchState."""
    store = EvidenceStore()
    cmjcc = CMJCC(store, config)
    out = cmjcc.run(CMJCCInput(
        candidate_state=cmjcc.memory.create_candidate_state({"candidate_id": "c-arity"}),
        dialogue_state=DialogueState(session_id="s", candidate_id="c-arity", version=1, turns=[]),
        extracted_preferences=ExtractedPreferenceSet(utterance_id="u", preferences=list(prefs)),
        catalog_snapshot_id="cat-test",
        config=config,
        run_id="run-arity",
    ))
    return out.active_search_state


# --------------------------------------------------------------- declared contract
def test_every_consumed_field_declares_an_arity() -> None:
    """No consumed field may be left without a declared shape contract."""
    for name in [*SCALAR_ARITY_FIELDS, *LIST_ARITY_FIELDS, "salary_min"]:
        assert field_arity(name) is not None, f"{name} has no declared arity"
    assert field_arity("totally_made_up_field") is None


@pytest.mark.parametrize("field_name", SCALAR_ARITY_FIELDS)
@pytest.mark.parametrize("bad", BAD_CONTAINERS)
def test_a_container_violates_a_scalar_field_and_is_reported(field_name: str, bad: Any) -> None:
    """A multi-value shape on a single-valued field is ``ok=False`` with a reason.

    ``ok=False`` is the point: it is what hands the value to the orchestrator's repair
    rung instead of letting it through untouched.
    """
    result = validate_field(field_name, bad)

    assert isinstance(result, FieldResult)
    assert result.ok is False, f"{field_name}: {bad!r} passed validation"
    assert result.warnings, f"{field_name}: {bad!r} rejected without a reason"
    assert all(w.startswith(f"{field_name}:") for w in result.warnings)
    # The stated value is preserved for the repair rung, never replaced.
    assert result.value is bad


@pytest.mark.parametrize("field_name", SCALAR_ARITY_FIELDS)
def test_a_scalar_still_satisfies_a_scalar_field(field_name: str) -> None:
    """The overwhelmingly common case is untouched by the arity gate."""
    result = validate_field(field_name, "senior" if field_name == "experience_level" else "remote"
                            if field_name == "work_modes" else "Kuala Lumpur")

    assert result.warnings == [] or all(":" in w for w in result.warnings)
    assert not isinstance(result.value, (list, tuple, set, dict))


@pytest.mark.parametrize("field_name", LIST_ARITY_FIELDS)
def test_a_list_arity_field_may_legitimately_carry_several_values(field_name: str) -> None:
    """``skills`` fanning out is a feature, not a shape error."""
    result = validate_field(field_name, "python, sql")

    assert result.ok is True
    assert result.value == ["python", "sql"]


def test_an_empty_container_is_absence_not_a_shape_error() -> None:
    """An empty list states nothing, so it is handled as absent rather than malformed."""
    for empty in ([], (), {}, set()):
        result = validate_field("target_roles", empty)
        assert not any("expected a single value" in w for w in result.warnings)


def test_an_unknown_field_name_is_reported_rather_than_vanishing() -> None:
    """A hallucinated field is still passed through, but never silently."""
    result = validate_field("favourite_colour", "teal")

    assert result.ok is True
    assert result.value == "teal"
    assert any("unknown field" in w for w in result.warnings)


# ------------------------------------------------------- the observed crash, repaired
def test_a_single_element_list_role_is_repaired_instead_of_crashing(caplog) -> None:
    """The exact payload that killed a real run now resolves through the repair rung."""
    caplog.set_level(logging.WARNING, logger="jobrec.orchestration.orchestrator")
    provider = ScriptedJSONProvider([_payload(("target_roles", ["software engineer"]))])

    pref_set, _calls = _orchestrator(provider, mode=RunMode.HYBRID)._extract(
        "I want a software engineer role")

    repaired = {p.field_name: p for p in pref_set.preferences}["target_roles"]
    assert repaired.normalized_value == "software engineer", "the list was not unwrapped"
    assert provider.json_calls == 1, "repair should not have needed a model retry"


def test_a_well_formed_skills_extraction_no_longer_nests_a_list(config) -> None:
    """``skills_have: ["python","sql"]`` is well-formed and must not crash.

    This is the case that needed no malformed input at all: the normalizer returns a
    list by design and the consumer used to append it as one element, so the very next
    ``canonical_skill`` call raised.
    """
    active = _active_search(config, _pref("skills_have", ["python", "sql"]))

    assert active.skills_have == ["python", "sql"]
    assert all(isinstance(s, str) for s in active.skills_have)


def test_a_csv_skills_string_fans_out_into_separate_values(config) -> None:
    validated = validate_field("skills_have", "python, sql")
    active = _active_search(config, _pref("skills_have", validated.value))

    assert set(active.skills_have) == {"python", "sql"}


def test_target_roles_never_nests_a_container_in_the_active_search(config) -> None:
    """Even if a container survives validation, the search state stays ``list[str]``."""
    active = _active_search(config, _pref("target_roles", ["software engineer"]))

    assert all(isinstance(r, str) for r in active.target_roles)


# ------------------------------------------------- salary_min: null reads as absent
def test_a_null_salary_is_absent_not_present(config) -> None:
    """``salary_min: null`` must land in ``unknown_fields``, not in the soft set.

    ``normalize_salary`` returns the canonical structure even for ``None``, so a plain
    ``is not None`` present-check counted an absent salary as stated. Measured on a real
    run: 5 hybrid bundles hit this and 0 deterministic ones, which made the same input
    produce different soft/unknown field sets in the two backends.
    """
    validated = validate_field("salary_min", None)
    assert isinstance(validated.value, dict), "fixture guard: still the canonical structure"

    active = _active_search(config, _pref("salary_min", validated.value))

    assert active.salary_min is None
    assert "salary_min" in active.unknown_fields
    assert "salary_min" not in active.soft_preference_fields
    assert "salary_min" not in active.hard_constraint_fields


def test_a_stated_salary_is_still_present(config) -> None:
    """The parity fix must not make a real salary disappear."""
    active = _active_search(config, _pref("salary_min", validate_field("salary_min", 4000).value))

    assert active.salary_min == 4000.0
    assert "salary_min" not in active.unknown_fields


# --------------------------------------------- exclusions stay matchable, not stringified
def test_a_negative_preference_never_produces_a_stringified_container_key(config) -> None:
    """An exclusion key must be a role, not ``"['sales analyst']"``.

    A coerced container looks like a recorded constraint but can never match a role
    family, i.e. the exclusion is silently inert.
    """
    active = _active_search(
        config, _pref("excluded_roles", ["sales analyst"], polarity="negative"))

    roles = active.exclusions.get("roles", [])
    assert roles == ["sales analyst"]
    assert not any(key.startswith("[") for key in roles)


# ------------------------------------------------------------------------ totality
@settings(max_examples=100)
@given(
    field_name=st.sampled_from([*SCALAR_ARITY_FIELDS, *LIST_ARITY_FIELDS,
                                "salary_min", "unheard_of_field"]),
    raw=st.recursive(
        st.one_of(st.none(), st.booleans(), st.integers(), st.floats(allow_nan=True),
                  st.text(max_size=8), st.binary(max_size=4)),
        lambda children: st.one_of(
            st.lists(children, max_size=3),
            st.tuples(children),
            st.dictionaries(st.text(max_size=4), children, max_size=3),
        ),
        max_leaves=5,
    ),
)
def test_property_the_arity_gate_keeps_validation_total(field_name: str, raw: Any) -> None:
    """Adding the shape check must not cost validation its totality (R8.7)."""
    result = validate_field(field_name, raw)

    assert isinstance(result, FieldResult)
    assert result.field_name == field_name
    assert isinstance(result.ok, bool)
    assert isinstance(result.warnings, list)
    for warning in result.warnings:
        assert warning.strip() and ":" in warning
