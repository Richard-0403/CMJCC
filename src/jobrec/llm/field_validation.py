"""Field-level validation and normalization for LLM-extracted preferences.

This module is the single source of truth for normalizing preference field
values, regardless of whether they were produced by the rule extractor or by a
model in hybrid mode. Model output is untrusted and arbitrarily shaped
(numbers, strings, nested objects, wrong types, invalid enums, missing values),
so every normalizer here is **total**: it returns a value-or-None together with
structured warnings and never raises.

Design contract (see design.md, R8):
- ``normalize_salary`` accepts int/float/str/object and returns the canonical
  ``{min_salary, max_salary, currency, period}`` structure.
- Enum-typed fields (``work_mode``, ``experience_level``) are constrained via
  the controlled vocabularies in :mod:`jobrec.taxonomy`.
- ``normalize_skills`` coerces scalars/CSV strings/lists into ``list[str]``.
- ``normalize_location`` canonicalizes via :func:`jobrec.taxonomy.canonical_location`.
- ``normalize_deadline`` parses common date forms into an ISO-8601 date string.
- A stated constraint is never silently dropped: an unrecoverable present value
  yields a structured warning rather than a silent ``None``.

Higher layers (orchestrator ``_extract``, ``cmjcc._as_float``) delegate here so
there is exactly one salary parser and one field-normalization code path.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any

from ..domain.extraction import ExtractedPreferenceSet
from ..taxonomy import (
    canonical_level,
    canonical_location,
    canonical_skill,
    canonical_work_mode,
)

# Defaults used when a salary is stated without an explicit currency/period.
# Currency is intentionally left as ``None`` (rather than guessed) so downstream
# monthly-MYR normalization treats it as unknown; a warning records the gap.
DEFAULT_CURRENCY: str | None = None
DEFAULT_PERIOD: str = "unknown"

_VALID_PERIODS = {"hour", "month", "year", "unknown"}

# Currency tokens -> ISO-ish currency codes (mirrors the rule extractor).
_CURRENCY_MAP = {
    "rm": "MYR", "myr": "MYR", "ringgit": "MYR",
    "sgd": "SGD", "s$": "SGD",
    "usd": "USD", "$": "USD", "us$": "USD",
    "eur": "EUR", "€": "EUR",
}

# Period phrases -> canonical period token.
_PERIOD_MAP = {
    "hour": "hour", "hourly": "hour", "hr": "hour", "/hr": "hour", "per hour": "hour",
    "month": "month", "monthly": "month", "mo": "month", "/mo": "month", "per month": "month",
    "year": "year", "yearly": "year", "annual": "year", "annually": "year",
    "annum": "year", "yr": "year", "/yr": "year", "per year": "year", "pa": "year",
}


@dataclass
class FieldResult:
    """The outcome of validating/normalizing a single extracted field.

    ``value`` is the normalized value (or ``None`` when unrecoverable).
    ``ok`` is True when a present value was normalized without loss.
    ``warnings`` carries structured, human-readable reasons (never silent).
    ``source`` records how the value was produced.
    """

    field_name: str
    value: Any
    ok: bool
    warnings: list[str] = field(default_factory=list)
    source: str = "normalized"  # "normalized" | "repaired" | "rule_fallback"


# --------------------------------------------------------------------------- #
# Salary
# --------------------------------------------------------------------------- #
def _coerce_amount(value: Any) -> float | None:
    """Best-effort numeric coercion from a scalar/string, tolerating currency."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return _amount_from_string(value)
    return None


def _amount_from_string(text: str) -> float | None:
    """Extract the first numeric amount from a string, honouring a 'k' suffix."""
    m = re.search(r"(\d[\d,]*(?:\.\d+)?)\s*([kK])?", text)
    if not m:
        return None
    try:
        amount = float(m.group(1).replace(",", ""))
    except (TypeError, ValueError):
        return None
    if m.group(2):
        amount *= 1000
    return amount


def _amounts_from_string(text: str) -> list[float]:
    """Extract all numeric amounts from a string, honouring 'k' suffixes."""
    amounts: list[float] = []
    for m in re.finditer(r"(\d[\d,]*(?:\.\d+)?)\s*([kK])?", text):
        try:
            amount = float(m.group(1).replace(",", ""))
        except (TypeError, ValueError):
            continue
        if m.group(2):
            amount *= 1000
        amounts.append(amount)
    return amounts


def _currency_from_string(text: str) -> str | None:
    low = text.lower()
    for token, code in _CURRENCY_MAP.items():
        if token in low:
            return code
    return None


def _period_from_string(text: str) -> str | None:
    low = text.lower()
    # Longer phrases first so "per month" wins over "mo".
    for token in sorted(_PERIOD_MAP, key=len, reverse=True):
        if token in low:
            return _PERIOD_MAP[token]
    return None


def _salary_structure(
    min_salary: float | None,
    max_salary: float | None,
    currency: str | None,
    period: str,
) -> dict:
    return {
        "min_salary": min_salary,
        "max_salary": max_salary,
        "currency": currency,
        "period": period if period in _VALID_PERIODS else DEFAULT_PERIOD,
    }


def normalize_salary(raw: Any) -> tuple[dict, list[str]]:
    """Normalize any salary shape into ``{min_salary, max_salary, currency, period}``.

    Accepts an int/float (``min == max``), a string (e.g. ``"RM50000"`` or
    ``"50k-60k/month"``), or an object (e.g. ``{"amount": 50000, "period": "month"}``
    or ``{"min": 5000, "max": 8000, "currency": "MYR"}``). Never raises; a present
    but unparseable value yields a warning while still returning the structure.
    """
    warnings: list[str] = []
    currency: str | None = DEFAULT_CURRENCY
    period: str = DEFAULT_PERIOD

    if raw is None:
        return _salary_structure(None, None, currency, period), warnings

    # ---- object / dict shape ------------------------------------------------
    if isinstance(raw, dict):
        currency = _clean_currency(raw.get("currency")) or currency
        raw_period = raw.get("period")
        if isinstance(raw_period, str):
            period = _PERIOD_MAP.get(raw_period.strip().lower(), raw_period.strip().lower())
            if period not in _VALID_PERIODS:
                warnings.append(f"salary: unknown period '{raw_period}', defaulted to '{DEFAULT_PERIOD}'")
                period = DEFAULT_PERIOD
        min_salary = _coerce_amount(
            raw.get("min_salary") if raw.get("min_salary") is not None else raw.get("min")
        )
        max_salary = _coerce_amount(
            raw.get("max_salary") if raw.get("max_salary") is not None else raw.get("max")
        )
        if min_salary is None and max_salary is None:
            amount = _coerce_amount(
                raw.get("amount")
                if raw.get("amount") is not None
                else (raw.get("value") if raw.get("value") is not None else raw.get("salary"))
            )
            if amount is not None:
                min_salary = max_salary = amount
        if min_salary is None and max_salary is None:
            warnings.append("salary: object provided without a recognizable amount")
        return _salary_structure(min_salary, max_salary, currency, period), warnings

    # ---- numeric shape ------------------------------------------------------
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        amount = float(raw)
        return _salary_structure(amount, amount, currency, period), warnings

    # ---- string shape -------------------------------------------------------
    if isinstance(raw, str):
        currency = _currency_from_string(raw) or currency
        period = _period_from_string(raw) or period
        amounts = _amounts_from_string(raw)
        if not amounts:
            warnings.append(f"salary: no numeric amount found in '{raw}'")
            return _salary_structure(None, None, currency, period), warnings
        if len(amounts) == 1:
            return _salary_structure(amounts[0], amounts[0], currency, period), warnings
        low, high = min(amounts[0], amounts[1]), max(amounts[0], amounts[1])
        return _salary_structure(low, high, currency, period), warnings

    # ---- wrong type ---------------------------------------------------------
    warnings.append(f"salary: unsupported type {type(raw).__name__}")
    return _salary_structure(None, None, currency, period), warnings


def _clean_currency(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    token = value.strip().lower()
    return _CURRENCY_MAP.get(token, value.strip().upper())


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #
def normalize_work_mode(raw: Any) -> tuple[str | None, list[str]]:
    """Constrain a work mode to the fixed enumeration (onsite/hybrid/remote)."""
    if raw is None:
        return None, []
    if not isinstance(raw, str):
        return None, [f"work_mode: expected string, got {type(raw).__name__}"]
    canon = canonical_work_mode(raw)
    if canon is None:
        return None, [f"work_mode: unknown value '{raw}'"]
    return canon, []


def normalize_experience_level(raw: Any) -> tuple[str | None, list[str]]:
    """Constrain an experience level to the fixed taxonomy enumeration."""
    if raw is None:
        return None, []
    if not isinstance(raw, str):
        return None, [f"experience_level: expected string, got {type(raw).__name__}"]
    canon = canonical_level(raw)
    if canon is None:
        return None, [f"experience_level: unknown value '{raw}'"]
    return canon, []


# --------------------------------------------------------------------------- #
# Skills
# --------------------------------------------------------------------------- #
def normalize_skills(raw: Any) -> tuple[list[str], list[str]]:
    """Coerce a scalar, CSV string, or list into a list of canonical skills."""
    warnings: list[str] = []
    if raw is None:
        return [], warnings
    items: list[Any]
    if isinstance(raw, str):
        items = [part for part in re.split(r"[,;/]", raw) if part.strip()]
        if not items and raw.strip():
            items = [raw]
    elif isinstance(raw, (list, tuple, set)):
        items = list(raw)
    else:
        return [], [f"skills: unsupported type {type(raw).__name__}"]

    out: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            warnings.append(f"skills: dropped non-string entry {item!r}")
            continue
        canon = canonical_skill(item)
        if canon and canon not in out:
            out.append(canon)
    return out, warnings


# --------------------------------------------------------------------------- #
# Location
# --------------------------------------------------------------------------- #
def normalize_location(raw: Any) -> tuple[str | None, list[str]]:
    """Canonicalize a location string via the taxonomy."""
    if raw is None:
        return None, []
    if not isinstance(raw, str):
        return None, [f"location: expected string, got {type(raw).__name__}"]
    if not raw.strip():
        return None, ["location: empty value"]
    canon = canonical_location(raw)
    if canon is None:
        return None, [f"location: could not canonicalize '{raw}'"]
    return canon, []


# --------------------------------------------------------------------------- #
# Deadline
# --------------------------------------------------------------------------- #
_DATE_FORMATS = [
    "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%Y/%m/%d",
    "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y", "%B %d %Y", "%b %d %Y",
]


def normalize_deadline(raw: Any) -> tuple[str | None, list[str]]:
    """Parse a deadline into an ISO-8601 (YYYY-MM-DD) date string."""
    if raw is None:
        return None, []
    if isinstance(raw, datetime):
        return raw.date().isoformat(), []
    if isinstance(raw, date):
        return raw.isoformat(), []
    if not isinstance(raw, str):
        return None, [f"deadline: expected date/string, got {type(raw).__name__}"]
    text = raw.strip()
    if not text:
        return None, ["deadline: empty value"]
    # Fast path: leading ISO date (tolerate trailing time/zone).
    iso = re.match(r"(\d{4}-\d{2}-\d{2})", text)
    if iso:
        try:
            return date.fromisoformat(iso.group(1)).isoformat(), []
        except ValueError:
            pass
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(text, fmt).date().isoformat(), []
        except ValueError:
            continue
    return None, [f"deadline: unrecognized date format '{raw}'"]


# --------------------------------------------------------------------------- #
# Dispatch
# --------------------------------------------------------------------------- #
# Fields whose raw value should be normalized, mapped to their normalizer.
# Salary is handled specially (structured result).
_NORMALIZERS = {
    "work_modes": normalize_work_mode,
    "work_mode": normalize_work_mode,
    "experience_level": normalize_experience_level,
    "skills_have": normalize_skills,
    "skills": normalize_skills,
    "preferred_locations": normalize_location,
    "location": normalize_location,
    "excluded_locations": normalize_location,
    "deadline": normalize_deadline,
    "application_deadline": normalize_deadline,
}

_SALARY_FIELDS = {"salary", "salary_min", "salary_max", "salary_range"}


def validate_field(field_name: str, raw: Any) -> FieldResult:
    """Validate/normalize a single field value, never raising.

    A present value that cannot be normalized produces ``ok=False`` with a
    structured warning (the caller preserves the stated constraint rather than
    dropping it silently).
    """
    if field_name in _SALARY_FIELDS:
        value, warnings = normalize_salary(raw)
        recovered = value.get("min_salary") is not None or value.get("max_salary") is not None
        ok = recovered or raw is None
        return FieldResult(field_name, value, ok, warnings)

    normalizer = _NORMALIZERS.get(field_name)
    if normalizer is None:
        # No dedicated normalizer: pass through unchanged (still total).
        return FieldResult(field_name, raw, True, [])

    value, warnings = normalizer(raw)
    if normalizer is normalize_skills:
        present = raw is not None and raw != "" and raw != []
        ok = bool(value) or not present
        return FieldResult(field_name, value, ok, warnings)

    present = raw is not None and raw != ""
    ok = value is not None or not present
    return FieldResult(field_name, value, ok, warnings)


def validate_extraction(
    pref_set: ExtractedPreferenceSet,
) -> tuple[ExtractedPreferenceSet, list[FieldResult]]:
    """Validate/normalize every preference in a set.

    Returns a new :class:`ExtractedPreferenceSet` whose preferences carry
    normalized ``normalized_value`` fields (a stated constraint is preserved as
    its original value when it cannot be normalized), plus the per-field
    :class:`FieldResult` list. Any warnings are appended to
    ``extraction_warnings``. The input set is never mutated.
    """
    results: list[FieldResult] = []
    new_prefs = []
    extra_warnings: list[str] = []

    for pref in pref_set.preferences:
        result = validate_field(pref.field_name, pref.normalized_value)
        results.append(result)
        if result.warnings:
            extra_warnings.extend(result.warnings)
        # Preserve the stated constraint: use the normalized value when the
        # field was normalized OK, otherwise keep the original raw value so it
        # is never silently dropped.
        new_value = result.value if result.ok else pref.normalized_value
        new_prefs.append(pref.model_copy(update={"normalized_value": new_value}))

    new_set = pref_set.model_copy(
        update={
            "preferences": new_prefs,
            "extraction_warnings": [*pref_set.extraction_warnings, *extra_warnings],
        }
    )
    return new_set, results
