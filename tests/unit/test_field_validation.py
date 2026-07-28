"""Field-level validation and normalization tests (R8).

Property 13 checks that ``normalize_salary`` is *shape-invariant*: the same stated
amount expressed as an int, a float, a numeric string, a formatted string, or an
object normalizes to the same canonical
``{min_salary, max_salary, currency, period}`` structure, without losing or
rescaling the amount.

Property 14 checks the two guarantees the module claims for *untrusted* model output:
validation is **total** (arbitrary garbage never raises) and it **never silently drops
a stated constraint** (an unrecoverable present value is always traceable via a
structured warning, a ``ok=False`` flag, or preservation of the original value).

Property 15 checks the *output type contract* of the non-salary normalizers: the enum
normalizers only ever emit a value from the controlled taxonomy vocabulary (never an
arbitrary passthrough string), ``normalize_skills`` only ever emits a deduplicated list of
clean canonical strings, ``normalize_location`` only ever emits a canonical location, and
``normalize_deadline`` only ever emits an ISO-8601 date - each alongside a list of
structured warnings.
"""

from __future__ import annotations

import logging
import re
from datetime import date, datetime
from typing import Any

from hypothesis import given, settings
from hypothesis import strategies as st

from jobrec.config import AppConfig
from jobrec.domain.enums import (
    ConfirmationStatus,
    ConstraintStrength,
    PersistenceScope,
    RunMode,
)
from jobrec.domain.extraction import ExtractedPreference, ExtractedPreferenceSet
from jobrec.llm.field_validation import (
    FieldResult,
    normalize_deadline,
    normalize_experience_level,
    normalize_location,
    normalize_salary,
    normalize_skills,
    normalize_work_mode,
    salary_amount,
    validate_extraction,
    validate_field,
)
from jobrec.llm.provider import LLMCallRecord
from jobrec.orchestration.cmjcc import _as_float
from jobrec.orchestration.orchestrator import ConversationOrchestrator
from jobrec.taxonomy import (
    EXPERIENCE_LEVELS,
    LOCATION_ALIASES,
    WORK_MODE_ALIASES,
    WORK_MODES,
    canonical_location,
    canonical_skill,
)
from tests.support.fault_injection import FaultInjectingProvider

SALARY_KEYS = {"min_salary", "max_salary", "currency", "period"}
VALID_PERIODS = {"hour", "month", "year", "unknown"}

#: Currency tokens the parser maps to a canonical code in both string and object form.
CURRENCY_TOKENS = ["RM", "MYR", "SGD", "USD"]
#: Unambiguous period phrases (full phrases avoid the "myr" / "yr" substring overlap).
PERIOD_PHRASES = ["per hour", "per month", "per year"]

#: Smart generator: plain amounts plus round thousands, so the "k" shorthand shape
#: (which states the same amount as ``N`` thousand) is always exercised.
AMOUNTS = st.one_of(
    st.integers(min_value=1, max_value=2_000_000),
    st.integers(min_value=1, max_value=900).map(lambda t: t * 1000),
)


def _single_amount_shapes(amount: int, currency: str, period: str) -> list[tuple[str, Any, bool]]:
    """Every accepted input shape stating exactly ``amount``.

    Returns ``(label, raw_value, states_units)`` triples; ``states_units`` marks the
    shapes that also state a currency and a period, so those can be compared.
    """
    shapes: list[tuple[str, Any, bool]] = [
        ("int", amount, False),
        ("float", float(amount), False),
        ("numeric-string", str(amount), False),
        ("formatted-string", f"{currency}{amount:,} {period}", True),
        ("object-amount", {"amount": amount, "currency": currency, "period": period}, True),
        (
            "object-min-max",
            {"min": amount, "max": amount, "currency": currency, "period": period},
            True,
        ),
        (
            "object-min-max-salary",
            {
                "min_salary": amount,
                "max_salary": amount,
                "currency": currency,
                "period": period,
            },
            True,
        ),
        (
            "object-string-amount",
            {"amount": f"{currency}{amount:,}", "currency": currency, "period": period},
            True,
        ),
    ]
    if amount % 1000 == 0:
        # "RM5k per month" states the same amount as "RM5,000 per month".
        shapes.append(("k-string", f"{currency}{amount // 1000}k {period}", True))
    return shapes


# Feature: cmjcc-experiment-readiness, Property 13: Salary normalization preserves the stated
# amount across input shapes
@settings(max_examples=100)
@given(
    amount=AMOUNTS,
    span=st.integers(min_value=0, max_value=500_000),
    currency=st.sampled_from(CURRENCY_TOKENS),
    period=st.sampled_from(PERIOD_PHRASES),
)
def test_property_salary_normalization_preserves_stated_amount_across_shapes(
    amount: int,
    span: int,
    currency: str,
    period: str,
) -> None:
    """Every accepted salary shape normalizes to the same canonical stated amount.

    **Validates: Requirements 8.2, 8.10**
    """
    seen_units: set[tuple[str | None, str]] = set()

    for label, raw, states_units in _single_amount_shapes(amount, currency, period):
        normalized, warnings = normalize_salary(raw)

        # R8.2: always the canonical structure, whatever the input shape.
        assert set(normalized) == SALARY_KEYS, f"{label}: unexpected salary keys"
        assert isinstance(warnings, list)
        assert normalized["period"] in VALID_PERIODS, f"{label}: period out of vocabulary"

        # R8.10: the stated amount survives the shape unchanged and unrescaled.
        assert normalized["min_salary"] == float(amount), f"{label}: min_salary lost the amount"
        assert normalized["max_salary"] == float(amount), f"{label}: max_salary lost the amount"
        assert isinstance(normalized["min_salary"], float)
        assert not warnings, f"{label}: recoverable shape emitted warnings {warnings}"

        # R8.10: exactly one salary parser - the orchestrator helper agrees.
        assert _as_float(raw) == float(amount), f"{label}: _as_float diverged from normalize_salary"

        if states_units:
            seen_units.add((normalized["currency"], normalized["period"]))

    # Shapes that state the same currency and period agree on the canonical pair.
    assert len(seen_units) == 1, f"units diverged across shapes: {seen_units}"

    # A stated range is likewise shape-invariant between string and object form.
    low, high = amount, amount + span
    from_string, string_warnings = normalize_salary(f"{currency}{low:,}-{high:,} {period}")
    from_object, object_warnings = normalize_salary(
        {"min": low, "max": high, "currency": currency, "period": period}
    )
    assert not string_warnings and not object_warnings
    assert from_string == from_object
    assert from_string["min_salary"] == float(low)
    assert from_string["max_salary"] == float(high)


def test_salary_amount_projects_the_canonical_structure_back_to_a_scalar() -> None:
    """``salary_amount`` is the one bridge from the salary structure to a number.

    ``validate_field`` stores the canonical structure on a salary preference, but the
    domain models (``ActiveSearchState.salary_min``, ``CandidateState.salary_min``) carry
    a scalar. Consumers must project through this helper instead of ``float()``, which
    raises on a dict; feeding it the structure it produced is idempotent, and an
    unrecoverable value yields ``None`` rather than raising (R8.2, R8.7).
    """
    structure = validate_field("salary_min", 4000).value

    assert salary_amount(structure) == 4000.0
    assert salary_amount(4000) == salary_amount("RM4000 per month") == 4000.0
    assert salary_amount(structure) == salary_amount(salary_amount(structure))
    assert salary_amount({"currency": "MYR", "period": "month"}) is None
    assert salary_amount("negotiable") is None
    assert salary_amount(None) is None


# --------------------------------------------------------------------------- #
# Property 14 support
# --------------------------------------------------------------------------- #
#: Salary fields whose normalized value is the canonical salary structure.
SALARY_FIELD_NAMES = {"salary", "salary_min", "salary_max", "salary_range"}

#: Every field the module dispatches on, plus one field with no dedicated
#: normalizer (which must still be handled, by pass-through).
VALIDATED_FIELDS = [
    "salary",
    "salary_min",
    "salary_max",
    "salary_range",
    "work_mode",
    "work_modes",
    "experience_level",
    "skills",
    "skills_have",
    "location",
    "preferred_locations",
    "excluded_locations",
    "deadline",
    "application_deadline",
    "unregistered_field",
]

#: Smart junk generator: the shapes an untrusted model actually emits - wrong types,
#: empty and whitespace strings, unicode, near-miss dates, and very large magnitudes.
_JUNK_SCALARS = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1_000_000_000, max_value=1_000_000_000),
    st.integers(min_value=10**12, max_value=10**24),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(max_size=16),
    st.binary(max_size=8),
    st.sampled_from(
        [
            "",
            "   ",
            "remote",
            "wfh",
            "sr",
            "KL",
            "Kuala Lumpur",
            "n/a",
            "unknown",
            "RM5k per month",
            "50,000-60,000",
            "2024-02-30",
            "31/13/2024",
            "中文",
            "€∞",
        ]
    ),
)

#: Nested junk: lists/tuples/sets/objects of junk, as a model might nest a field value.
JUNK = st.recursive(
    _JUNK_SCALARS,
    lambda children: st.one_of(
        st.lists(children, max_size=4),
        st.tuples(children),
        st.sets(st.text(max_size=6), max_size=3),
        st.dictionaries(
            st.sampled_from(
                [
                    "min",
                    "max",
                    "min_salary",
                    "max_salary",
                    "amount",
                    "value",
                    "currency",
                    "period",
                    "junk",
                ]
            ),
            children,
            max_size=4,
        ),
    ),
    max_leaves=5,
)


def _states_a_constraint(raw: Any) -> bool:
    """True when the raw value actually states something (i.e. is not absent/empty)."""
    if raw is None:
        return False
    if isinstance(raw, str):
        return bool(raw.strip())
    if isinstance(raw, (list, tuple, set, dict)):
        return len(raw) > 0
    return True


def _carries_value(field_name: str, value: Any) -> bool:
    """True when the validated result still carries a usable value for the field."""
    if field_name in SALARY_FIELD_NAMES and isinstance(value, dict):
        return value.get("min_salary") is not None or value.get("max_salary") is not None
    if isinstance(value, (list, tuple, set)):
        return bool(value)
    return value is not None


def _same(left: Any, right: Any) -> bool:
    """Identity-or-equality comparison (identity keeps NaN payloads comparable)."""
    if left is right:
        return True
    try:
        return bool(left == right)
    except Exception:  # pragma: no cover - defensive for exotic junk
        return False


def _preference(field_name: str, raw: Any) -> ExtractedPreference:
    return ExtractedPreference(
        field_name=field_name,
        normalized_value=raw,
        raw_text=f"stated {field_name}",
        confidence=0.5,
        confirmation_status=ConfirmationStatus.UNCONFIRMED,
        persistence_scope=PersistenceScope.ACTIVE_SEARCH,
        proposed_strength=ConstraintStrength.SOFT,
    )


# Feature: cmjcc-experiment-readiness, Property 14: Field validation is total and never silently
# drops a stated constraint
@settings(max_examples=100)
@given(
    field_name=st.sampled_from(VALIDATED_FIELDS),
    raw=JUNK,
    other_field=st.sampled_from(VALIDATED_FIELDS),
    other_raw=JUNK,
    magnitude=st.integers(min_value=100, max_value=400),
)
def test_property_field_validation_is_total_and_never_drops_a_stated_constraint(
    field_name: str,
    raw: Any,
    other_field: str,
    other_raw: Any,
    magnitude: int,
) -> None:
    """Validation absorbs arbitrary junk without raising and never loses a stated constraint.

    **Validates: Requirements 8.7, 8.9, 8.11, 31.9**
    """
    # ---- R8.7: total on arbitrary/garbage input (any raise fails the test) ----------
    result = validate_field(field_name, raw)

    assert isinstance(result, FieldResult)
    assert result.field_name == field_name
    assert isinstance(result.ok, bool)
    assert isinstance(result.warnings, list)

    # ---- R8.11: warnings are structured "field: reason" strings, never blank ---------
    for warning in result.warnings:
        assert isinstance(warning, str)
        assert warning.strip()
        assert ":" in warning, f"unstructured warning {warning!r}"

    if field_name in SALARY_FIELD_NAMES:
        assert set(result.value) == SALARY_KEYS
        assert result.value["period"] in VALID_PERIODS

    # ---- R8.9 / R31.9: a stated constraint always leaves a trace ---------------------
    # Either it normalized to a usable value, or it was preserved verbatim (no dedicated
    # normalizer), or the loss is flagged (ok=False) and/or explained by a warning.
    # It is never dropped to an unflagged, unexplained empty value.
    if _states_a_constraint(raw):
        assert (
            _carries_value(field_name, result.value)
            or result.value is raw
            or not result.ok
            or result.warnings
        ), f"{field_name}: stated value {raw!r} vanished with no warning and ok=True"

    # A clean ok=True result for a stated constraint must actually carry the value
    # (or preserve it, or explain what was adjusted).
    if result.ok and _states_a_constraint(raw) and not result.warnings:
        assert _carries_value(field_name, result.value) or result.value is raw, (
            f"{field_name}: ok=True but stated value {raw!r} produced no value"
        )

    # ---- R8.9: whole-set validation keeps every preference and every warning ---------
    pref_set = ExtractedPreferenceSet(
        utterance_id="u-prop14",
        preferences=[_preference(field_name, raw), _preference(other_field, other_raw)],
        extraction_warnings=["pre-existing warning"],
    )
    original_values = [p.normalized_value for p in pref_set.preferences]

    new_set, results = validate_extraction(pref_set)

    assert len(results) == 2
    assert len(new_set.preferences) == 2
    # Prior warnings survive and every field warning is recorded, never swallowed.
    assert new_set.extraction_warnings[0] == "pre-existing warning"
    for field_result in results:
        for warning in field_result.warnings:
            assert warning in new_set.extraction_warnings

    for original, updated, field_result in zip(
        original_values, new_set.preferences, results, strict=True
    ):
        # Constraint metadata is untouched; only the value is normalized.
        assert updated.confirmation_status is ConfirmationStatus.UNCONFIRMED
        assert updated.raw_text == f"stated {updated.field_name}"
        if field_result.ok:
            assert _same(updated.normalized_value, field_result.value)
        else:
            # Unrecoverable: the originally stated value is preserved, not discarded.
            assert _same(updated.normalized_value, original), (
                f"{updated.field_name}: stated value {original!r} was dropped"
            )

    # The input set is never mutated.
    assert pref_set.extraction_warnings == ["pre-existing warning"]
    assert [p.normalized_value for p in pref_set.preferences] == original_values

    # ---- R8.7: very large numbers are absorbed too, in scalar and object form --------
    huge = 10**magnitude
    for name in ("salary", "salary_range", "skills", "unregistered_field"):
        assert isinstance(validate_field(name, huge), FieldResult)
        assert isinstance(validate_field(name, {"min": huge, "max": huge}), FieldResult)


# --------------------------------------------------------------------------- #
# Property 15 support
# --------------------------------------------------------------------------- #
#: The controlled vocabularies the enum normalizers are allowed to emit (R8.3).
KNOWN_WORK_MODES = set(WORK_MODES)
KNOWN_LEVELS = set(EXPERIENCE_LEVELS)
#: Canonical location names the taxonomy resolves aliases to (R8.5).
KNOWN_LOCATIONS = set(LOCATION_ALIASES.values())

#: ISO-8601 calendar date, the single deadline output form (R8.6).
_ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

#: Dates kept inside the range platform ``strftime`` formats reliably.
DEADLINE_DATES = st.dates(min_value=date(1970, 1, 1), max_value=date(2100, 12, 31))

#: Experience-level aliases whose normalized form is actually resolvable (the taxonomy
#: also lists hyphenated aliases, which normalize to a spaced form and stay unknown).
RESOLVABLE_LEVEL_TOKENS = [
    "intern",
    "internship",
    "entry",
    "graduate",
    "junior",
    "jr",
    "mid",
    "intermediate",
    "senior",
    "sr",
    "lead",
    "principal",
]

#: Smart generator: what a model actually emits for these fields - valid tokens, near-miss
#: phrasings, CSV skill blobs, blanks, punctuation-only strings, wrong types and nested
#: shapes (reusing the Property 14 junk generator for the untrusted-shape space).
FIELD_INPUTS = st.one_of(
    JUNK,
    st.sampled_from(
        [
            "Remote",
            "on-site",
            "WFH",
            "prefer remote work",
            "hybridd",
            "Sr",
            "entry-level",
            "principal",
            "python, sql; excel",
            "Power BI/Tableau",
            "  ",
            ",",
            "!!!",
            "kl",
            "Kuala  Lumpur",
            "Nowhere City",
            "2026-03-01",
            "2026-03-01T09:30:00",
            "01/03/2026",
            "March 1, 2026",
            "next friday",
            "2026-02-30",
        ]
    ),
    st.lists(st.one_of(st.text(max_size=8), st.none(), st.integers(), st.booleans()), max_size=5),
    DEADLINE_DATES,
    st.datetimes(
        min_value=datetime(1970, 1, 1),
        max_value=datetime(2100, 12, 31),
    ),
)


def _assert_structured_warnings(field_name: str, warnings: Any) -> None:
    """Every normalizer reports problems as a list of ``"field: reason"`` strings (R8.11)."""
    assert isinstance(warnings, list), f"{field_name}: warnings must be a list"
    for warning in warnings:
        assert isinstance(warning, str), f"{field_name}: warning {warning!r} is not a string"
        assert warning.strip(), f"{field_name}: blank warning"
        assert warning.startswith(f"{field_name}:"), f"{field_name}: unstructured {warning!r}"


def _assert_enum_contract(
    field_name: str,
    raw: Any,
    value: Any,
    warnings: Any,
    vocabulary: set[str],
) -> None:
    """An enum normalizer emits ``None`` or a vocabulary member - never a passthrough."""
    _assert_structured_warnings(field_name, warnings)
    assert value is None or value in vocabulary, f"{field_name}: {value!r} escaped the vocabulary"
    if raw is None:
        assert value is None and not warnings, f"{field_name}: absent value invented {value!r}"
    elif not isinstance(raw, str):
        # Wrong type is never coerced into the vocabulary; the loss is always explained.
        assert value is None, f"{field_name}: coerced {type(raw).__name__} into {value!r}"
        assert warnings, f"{field_name}: wrong type dropped without a warning"
    elif value is None:
        assert warnings, f"{field_name}: unknown value {raw!r} dropped without a warning"


# Feature: cmjcc-experiment-readiness, Property 15: Enum, skills, location, and deadline
# normalizers produce well-typed output
@settings(max_examples=100)
@given(
    raw=FIELD_INPUTS,
    mode_token=st.sampled_from(sorted(WORK_MODE_ALIASES)),
    level_token=st.sampled_from(RESOLVABLE_LEVEL_TOKENS),
    style=st.integers(min_value=0, max_value=2),
    location_token=st.sampled_from(sorted(LOCATION_ALIASES)),
    day=DEADLINE_DATES,
)
def test_property_enum_skills_location_deadline_normalizers_are_well_typed(
    raw: Any,
    mode_token: str,
    level_token: str,
    style: int,
    location_token: str,
    day: date,
) -> None:
    """Each non-salary normalizer emits only its declared output type, whatever the input.

    **Validates: Requirements 8.3, 8.4, 8.5, 8.6**
    """
    # ---- R8.3: enums are constrained to the fixed taxonomy vocabularies --------------
    mode, mode_warnings = normalize_work_mode(raw)
    _assert_enum_contract("work_mode", raw, mode, mode_warnings, KNOWN_WORK_MODES)

    level, level_warnings = normalize_experience_level(raw)
    _assert_enum_contract("experience_level", raw, level, level_warnings, KNOWN_LEVELS)

    # ---- R8.4: skills are always a list of clean, canonical, deduplicated strings -----
    skills, skills_warnings = normalize_skills(raw)
    _assert_structured_warnings("skills", skills_warnings)
    assert isinstance(skills, list), f"skills: got {type(skills).__name__}"
    for skill in skills:
        assert isinstance(skill, str), f"skills: non-string entry {skill!r}"
        assert skill, "skills: empty entry"
        assert skill.strip() == skill, f"skills: unstripped entry {skill!r}"
        assert skill == skill.lower(), f"skills: uncanonical case {skill!r}"
        assert skill == canonical_skill(skill), f"skills: {skill!r} is not canonical"
    assert len(set(skills)) == len(skills), f"skills: duplicate entries in {skills}"
    if raw is None:
        assert skills == [] and not skills_warnings

    # ---- R8.5: location is None or a canonical (already-normalized) location ----------
    location, location_warnings = normalize_location(raw)
    _assert_structured_warnings("location", location_warnings)
    if location is not None:
        assert isinstance(location, str), f"location: got {type(location).__name__}"
        assert location.strip() == location and location, f"location: unclean {location!r}"
        assert location == canonical_location(location), f"location: {location!r} is not canonical"
    if raw is None:
        assert location is None and not location_warnings
    elif not isinstance(raw, str):
        assert location is None, f"location: coerced {type(raw).__name__} into {location!r}"
        assert location_warnings, "location: wrong type dropped without a warning"
    elif location is None:
        assert location_warnings, f"location: {raw!r} dropped without a warning"

    # ---- R8.6: deadline is None or a single unified ISO-8601 date form ----------------
    deadline, deadline_warnings = normalize_deadline(raw)
    _assert_structured_warnings("deadline", deadline_warnings)
    if deadline is not None:
        assert isinstance(deadline, str), f"deadline: got {type(deadline).__name__}"
        assert _ISO_DATE.match(deadline), f"deadline: {deadline!r} is not an ISO date"
        assert date.fromisoformat(deadline).isoformat() == deadline
        assert not deadline_warnings, f"deadline: parsed value warned {deadline_warnings}"
    if raw is None:
        assert deadline is None and not deadline_warnings
    elif not isinstance(raw, (str, date, datetime)):
        assert deadline is None, f"deadline: coerced {type(raw).__name__} into {deadline!r}"
        assert deadline_warnings, "deadline: wrong type dropped without a warning"

    # ---- Controlled tokens resolve into the vocabulary, not merely "not crash" --------
    styled_mode = [mode_token, mode_token.upper(), f"  {mode_token.title()}  "][style]
    resolved_mode, resolved_mode_warnings = normalize_work_mode(styled_mode)
    assert resolved_mode == WORK_MODE_ALIASES[mode_token], f"work_mode: {styled_mode!r} unresolved"
    assert not resolved_mode_warnings

    styled_level = [level_token, level_token.upper(), f"  {level_token.title()}  "][style]
    resolved_level, resolved_level_warnings = normalize_experience_level(styled_level)
    assert resolved_level in KNOWN_LEVELS, f"experience_level: {styled_level!r} unresolved"
    assert not resolved_level_warnings

    resolved_location, resolved_location_warnings = normalize_location(location_token.title())
    assert resolved_location in KNOWN_LOCATIONS, f"location: {location_token!r} unresolved"
    assert not resolved_location_warnings

    # ---- Every stated date form collapses to the same ISO output ----------------------
    date_shapes: list[Any] = [
        day,
        datetime(day.year, day.month, day.day, 9, 30),
        day.isoformat(),
        f"{day.isoformat()}T09:30:00",
        f"{day:%d/%m/%Y}",
        f"{day:%d %B %Y}",
        f"{day:%B %d, %Y}",
    ]
    for shape in date_shapes:
        parsed, parsed_warnings = normalize_deadline(shape)
        assert parsed == day.isoformat(), f"deadline: {shape!r} lost the stated date"
        assert not parsed_warnings, f"deadline: {shape!r} warned {parsed_warnings}"

# --------------------------------------------------------------------------- #
# Task 5.7: concrete repair -> retry -> rule-fallback ordering and warning emission
#
# These are example-based (not property) tests of the recovery ladder the
# orchestrator's ``_extract`` actually implements: field validation right after the
# lenient parse, then schema repair, then a *single* bounded model retry (hybrid
# only), then the rule fallback - with a warning/error log at every step and a
# stated-but-unrecoverable value preserved rather than dropped (R8.8, R8.11).
# --------------------------------------------------------------------------- #

ORCHESTRATOR_LOGGER = "jobrec.orchestration.orchestrator"

#: Utterance the rule extractor *can* parse (work mode + location), so the rule
#: fallback has a real value to substitute for an unrecoverable model value.
UTTERANCE = "I want a remote job in Kuala Lumpur"

#: Log fragments marking each rung of the ladder, in the order R8.8 mandates.
STEP_REPAIR = "attempting schema repair"
STEP_RETRY = "attempting one bounded model retry"
STEP_FALLBACK = "applying rule fallback"


class ScriptedJSONProvider:
    """Provider that replays a fixed sequence of extraction payloads.

    Each ``complete_json`` call returns the next scripted payload (the last payload
    repeats once the script is exhausted) and counts the call, so tests can assert
    exactly how many model calls the recovery ladder made. ``complete_text`` echoes
    the fallback, matching the provider protocol.
    """

    def __init__(self, payloads: list[dict]) -> None:
        if not payloads:
            raise ValueError("at least one payload is required")
        self._payloads = payloads
        self.name = "scripted"
        self.model = "scripted-v1"
        self.json_calls = 0

    def complete_json(self, prompt: str, *, purpose: str) -> tuple[dict, LLMCallRecord]:
        payload = self._payloads[min(self.json_calls, len(self._payloads) - 1)]
        self.json_calls += 1
        record = LLMCallRecord(
            call_id=f"call-{self.json_calls}",
            purpose=purpose,
            prompt=prompt,
            raw_response="<scripted>",
            parsed_ok=True,
            latency_ms=0.0,
            provider=self.name,
            model=self.model,
        )
        return payload, record

    def complete_text(
        self, prompt: str, *, purpose: str, fallback: str = ""
    ) -> tuple[str, LLMCallRecord]:
        record = LLMCallRecord(
            call_id="call-text",
            purpose=purpose,
            prompt=prompt,
            raw_response=fallback,
            parsed_ok=True,
            latency_ms=0.0,
            provider=self.name,
            model=self.model,
        )
        return fallback, record

    def manifest(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model}


def _payload(*fields: tuple[str, Any]) -> dict:
    """A minimal model extraction payload stating ``(field_name, value)`` pairs."""
    return {
        "preferences": [
            {"field_name": name, "normalized_value": value, "raw_text": str(value)}
            for name, value in fields
        ]
    }


def _orchestrator(
    provider: Any,
    *,
    mode: RunMode = RunMode.HYBRID,
    max_retries: int = 5,
) -> ConversationOrchestrator:
    """An orchestrator wired only for ``_extract`` (empty catalog, no database)."""
    config = AppConfig()
    config.llm.mode = mode
    config.llm.max_retries = max_retries
    return ConversationOrchestrator(
        config, [], "snapshot-test", "catalog-hash-test", provider=provider
    )


def _steps(caplog: Any) -> list[tuple[int, str]]:
    """The orchestrator's ``(level, message)`` log entries, in emission order."""
    return [
        (record.levelno, record.getMessage())
        for record in caplog.records
        if record.name == ORCHESTRATOR_LOGGER
    ]


def _step_index(steps: list[tuple[int, str]], needle: str) -> int:
    for i, (_level, message) in enumerate(steps):
        if needle in message:
            return i
    raise AssertionError(f"no log entry matched {needle!r} in {[m for _, m in steps]}")


def _by_field(pref_set: ExtractedPreferenceSet) -> dict[str, ExtractedPreference]:
    return {p.field_name: p for p in pref_set.preferences}


def test_deterministic_mode_skips_the_model_and_the_bounded_retry(caplog) -> None:
    """Deterministic runs use the rule extractor only - no model call, no retry.

    The bounded retry rung is hybrid-only (R8.8), so in deterministic mode the
    provider is never touched even though a repairable model payload is scripted,
    and every field is attributed to the rule extractor.
    """
    caplog.set_level(logging.WARNING, logger=ORCHESTRATOR_LOGGER)
    provider = ScriptedJSONProvider([_payload(("work_modes", ["remote"]))])

    pref_set, calls = _orchestrator(provider, mode=RunMode.DETERMINISTIC)._extract(UTTERANCE)

    assert provider.json_calls == 0, "deterministic mode called the model"
    assert calls == []
    assert pref_set.preferences, "rule extractor produced nothing for the utterance"
    assert all(p.metadata["extraction_method"] == "rule" for p in pref_set.preferences)
    # No recovery ladder ran, so nothing was logged about repair/retry/fallback.
    assert _steps(caplog) == []


def test_schema_repair_runs_first_and_short_circuits_before_any_model_retry(caplog) -> None:
    """A repairable shape is fixed by schema repair, with no second model call.

    ``["remote"]`` fails enum validation but is unambiguously coercible, so the first
    rung of the ladder resolves it: exactly one model call is made, the repair is
    logged as a warning, and the value stays attributed to the model (R8.8).
    """
    caplog.set_level(logging.WARNING, logger=ORCHESTRATOR_LOGGER)
    provider = ScriptedJSONProvider([_payload(("work_modes", ["remote"]))])

    pref_set, calls = _orchestrator(provider)._extract(UTTERANCE)

    assert provider.json_calls == 1, "repair should not trigger a model retry"
    assert len(calls) == 1
    repaired = _by_field(pref_set)["work_modes"]
    assert repaired.normalized_value == "remote"
    assert repaired.metadata["extraction_method"] == "llm"
    # R13.1: the rung that produced the value is persisted alongside the method.
    assert repaired.metadata["extraction_source"] == "repaired"

    steps = _steps(caplog)
    assert any(STEP_REPAIR in message for _level, message in steps)
    assert any("work_modes: repaired via schema coercion" in message for _level, message in steps)
    # The later rungs were never reached.
    assert not any(STEP_RETRY in message for _level, message in steps)
    assert not any(STEP_FALLBACK in message for _level, message in steps)
    assert "work_modes: repaired via schema coercion" in pref_set.extraction_warnings


def test_repair_then_one_bounded_retry_then_rule_fallback_run_in_that_order(caplog) -> None:
    """The ladder is ordered repair -> single retry -> rule fallback, each logged.

    The payload mixes a repairable field (``["remote"]``) with an unrecoverable one
    (a free-text deadline). Repair fixes the first field, the still-failing field
    triggers exactly one bounded model retry - one extra call, not ``max_retries``
    worth - and the rule fallback closes the ladder (R8.8, R8.11).
    """
    caplog.set_level(logging.WARNING, logger=ORCHESTRATOR_LOGGER)
    provider = ScriptedJSONProvider(
        [
            _payload(("work_modes", ["remote"]), ("deadline", "next friday")),
            _payload(("work_modes", "remote"), ("deadline", "next friday")),
        ]
    )

    pref_set, calls = _orchestrator(provider, max_retries=5)._extract(UTTERANCE)

    # Bounded: the initial call plus exactly one retry, regardless of max_retries.
    assert provider.json_calls == 2, "the in-ladder model retry is not bounded to one"
    assert len(calls) == 2

    steps = _steps(caplog)
    repair_at = _step_index(steps, STEP_REPAIR)
    retry_at = _step_index(steps, STEP_RETRY)
    fallback_at = _step_index(steps, STEP_FALLBACK)
    assert repair_at < retry_at < fallback_at, f"ladder ran out of order: {steps}"
    # Repair is attempted before the retry is even considered.
    assert _step_index(steps, "repaired via schema coercion") < retry_at

    # R8.8: a warning or error accompanies every rung.
    assert steps[repair_at][0] == logging.WARNING
    assert steps[retry_at][0] == logging.WARNING
    assert steps[fallback_at][0] == logging.ERROR

    prefs = _by_field(pref_set)
    assert prefs["work_modes"].normalized_value == "remote"
    # R8.11: the unrecoverable field is explained, not silently dropped.
    assert prefs["deadline"].normalized_value == "next friday"
    assert any("deadline" in w for w in pref_set.extraction_warnings)


def test_rule_fallback_supplies_the_value_when_the_model_value_stays_unrecoverable(
    caplog,
) -> None:
    """An unrepairable enum value falls back to the rule-extracted value, unconfirmed.

    ``"hybirdd"`` is neither a valid enum member nor coercible, and the retry returns
    the same payload, so the last rung substitutes the rule extractor's value, retags
    the field as rule-derived, and marks it ``UNCONFIRMED`` with an error log (R8.8).
    """
    caplog.set_level(logging.WARNING, logger=ORCHESTRATOR_LOGGER)
    provider = ScriptedJSONProvider([_payload(("work_modes", "hybirdd"))])

    pref_set, _calls = _orchestrator(provider)._extract(UTTERANCE)

    assert provider.json_calls == 2, "repair and the bounded retry both ran once"
    fallen_back = _by_field(pref_set)["work_modes"]
    assert fallen_back.normalized_value == "remote", "rule fallback value not applied"
    assert fallen_back.metadata["extraction_method"] == "rule"
    assert fallen_back.metadata["extraction_source"] == "rule_fallback"
    assert fallen_back.confirmation_status is ConfirmationStatus.UNCONFIRMED

    steps = _steps(caplog)
    fallback_message = "work_modes: LLM value unrecoverable; used rule-based fallback"
    assert any(
        fallback_message in message and level == logging.ERROR for level, message in steps
    ), f"rule fallback was not logged as an error: {steps}"
    assert any(fallback_message in w for w in pref_set.extraction_warnings)


def test_unrecoverable_value_is_preserved_with_a_warning_instead_of_vanishing(caplog) -> None:
    """A stated value with no rule fallback survives as an unconfirmed constraint.

    The rule extractor knows nothing about ``deadline``, so the last rung cannot
    substitute a value. R8.11 requires the loss to be reported and R8.9 requires the
    stated constraint to stay, so the raw value is kept and two structured warnings
    (the normalization reason and the preservation notice) are recorded.
    """
    caplog.set_level(logging.WARNING, logger=ORCHESTRATOR_LOGGER)
    provider = ScriptedJSONProvider([_payload(("deadline", "next friday"))])

    pref_set, _calls = _orchestrator(provider)._extract(UTTERANCE)

    preserved = _by_field(pref_set)["deadline"]
    assert preserved.normalized_value == "next friday", "stated constraint was dropped"
    assert preserved.confirmation_status is ConfirmationStatus.UNCONFIRMED
    # R13.1: an unnormalizable stated value is reported as such, not as normalized.
    assert preserved.metadata["extraction_source"] == "unresolved"

    warnings = pref_set.extraction_warnings
    assert "deadline: unrecognized date format 'next friday'" in warnings
    assert (
        "deadline: value could not be normalized; preserved as unconfirmed constraint" in warnings
    )
    assert any(
        "preserved as unconfirmed constraint" in message and level >= logging.WARNING
        for level, message in _steps(caplog)
    )


def test_repeated_model_errors_fall_back_to_rules_after_a_bounded_number_of_attempts(
    caplog,
) -> None:
    """Transient model failures are retried a bounded number of times, then rules win.

    With ``max_retries=2`` the provider is called at most three times; once the model
    keeps failing the extraction falls back to the rule extractor with a warning
    rather than fabricating a value (R8.8).
    """
    caplog.set_level(logging.WARNING, logger=ORCHESTRATOR_LOGGER)
    provider = FaultInjectingProvider(fail_times=3)

    pref_set, calls = _orchestrator(provider, max_retries=2)._extract(UTTERANCE)

    assert provider.attempts == 3, "retry budget was not max_retries + 1"
    # No SUCCESSFUL call is recorded, yet all three spent attempts are: asserting an
    # empty list made the retry budget invisible in the archive.
    assert len(calls) == 3, "every spent attempt must be recorded"
    assert all(c.metadata["failed"] is True and not c.parsed_ok for c in calls)
    assert [c.metadata["attempts"] for c in calls] == [1, 2, 3]
    assert len({c.call_id for c in calls}) == 3, "attempt records must not collide"
    assert pref_set.preferences, "rule fallback produced nothing"
    assert all(p.metadata["extraction_method"] == "rule" for p in pref_set.preferences)
    assert any(
        "model call failed; falling back to rule extractor" in message
        and level >= logging.WARNING
        for level, message in _steps(caplog)
    )
