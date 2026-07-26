"""Candidate Understanding Agent: natural-language -> ExtractedPreferenceSet.

This module provides a deterministic, rule-based extractor used in the
``deterministic`` run mode and as the fallback in ``hybrid`` mode. It parses an
utterance into structured preferences with text spans, confidence, polarity,
proposed constraint strength and temporal scope.

The extractor never makes final eligibility decisions; it only surfaces
structured evidence for the memory agent and CMJCC to reconcile.
"""

from __future__ import annotations

import re

from ..domain.enums import ConfirmationStatus, ConstraintStrength, PersistenceScope
from ..domain.extraction import ExtractedPreference, ExtractedPreferenceSet
from ..taxonomy import (
    EXPERIENCE_LEVEL_ALIASES,
    ROLE_FAMILIES,
    SKILL_SYNONYMS,
    canonical_level,
    canonical_role,
    canonical_skill,
)
from ..utils.hashing import content_id

# Language cues that map to constraint strength.
_HARD_CUES = ["must", "only", "cannot", "can't", "at least", "minimum", "required", "no less than", "absolutely"]
_SOFT_CUES = ["prefer", "ideally", "would like", "better", "nice to have", "hopefully"]
_FLEX_CUES = ["can consider", "flexible", "open to", "also fine", "is fine", "is ok", "okay", "acceptable", "would also"]
_UNSURE_CUES = ["maybe", "probably", "not sure", "might", "perhaps"]
_NEG_CUES = ["don't want", "do not want", "not interested", "exclude", "avoid", "no ", "not "]
# Threshold cues that make a salary/experience minimum a hard constraint.
_THRESHOLD_CUES = ["at least", "minimum", "above", "over", "more than", "no less than", "must"]

# Temporal-scope phrase cues. Durable phrases ("from now on") mark a preference as
# ``long_term`` so it can be written back to long-term memory; one-off phrases
# ("this time only") pin it to the ``current_search`` and never persist. When no
# cue is present the scope defaults to ``current_search`` (unchanged behaviour).
_DURABLE_TEMPORAL_CUES = ["from now on", "always", "going forward", "in general", "permanently"]
_ONE_OFF_TEMPORAL_CUES = ["this time", "just this search", "for now", "only this"]

_CURRENCY_MAP = {
    "rm": "MYR", "myr": "MYR", "ringgit": "MYR",
    "sgd": "SGD", "s$": "SGD",
    "usd": "USD", "$": "USD", "us$": "USD",
    "eur": "EUR", "€": "EUR",
}

_WORK_MODES = ["remote", "hybrid", "onsite", "on-site", "on site"]

_LOCATIONS = [
    "kuala lumpur", "kl", "penang", "johor bahru", "johor", "cyberjaya",
    "singapore", "selangor", "malaysia",
]
_LOCATION_CANON = {
    "kl": "Kuala Lumpur",
    "kuala lumpur": "Kuala Lumpur",
    "penang": "Penang",
    "johor bahru": "Johor Bahru",
    "johor": "Johor Bahru",
    "cyberjaya": "Cyberjaya",
    "singapore": "Singapore",
    "selangor": "Selangor",
    "malaysia": "Malaysia",
}

EXTRACTOR_NAME = "rule_based_extractor"
EXTRACTOR_VERSION = "1.0.0"


def _strength_for(window: str) -> ConstraintStrength:
    w = window.lower()
    if any(cue in w for cue in _HARD_CUES):
        return ConstraintStrength.HARD
    if any(cue in w for cue in _UNSURE_CUES):
        return ConstraintStrength.UNKNOWN
    if any(cue in w for cue in _SOFT_CUES) or any(cue in w for cue in _FLEX_CUES):
        return ConstraintStrength.SOFT
    return ConstraintStrength.SOFT


def _confirmation_for(strength: ConstraintStrength, window: str) -> ConfirmationStatus:
    if any(cue in window.lower() for cue in _UNSURE_CUES):
        return ConfirmationStatus.UNCONFIRMED
    return ConfirmationStatus.CONFIRMED


def _temporal_for(window: str) -> str:
    """Resolve a temporal scope from phrase cues in ``window``.

    Durable cues ("from now on", "always", ...) take precedence and map to
    ``long_term``; one-off cues ("this time", "for now", ...) map to
    ``current_search``. With no cue the scope defaults to ``current_search`` so
    existing behaviour is preserved.
    """
    w = window.lower()
    if any(cue in w for cue in _DURABLE_TEMPORAL_CUES):
        return "long_term"
    if any(cue in w for cue in _ONE_OFF_TEMPORAL_CUES):
        return "current_search"
    return "current_search"


def _find(text: str, needle: str) -> tuple[int, int] | None:
    idx = text.find(needle)
    if idx < 0:
        return None
    return idx, idx + len(needle)


class CandidateUnderstandingAgent:
    """Deterministic rule-based intent extractor."""

    name = "candidate_understanding_agent"

    def extract(self, text: str, utterance_id: str | None = None) -> ExtractedPreferenceSet:
        """Extract structured preferences from a single utterance."""
        lower = text.lower()
        utterance_id = utterance_id or content_id("utt", text)
        prefs: list[ExtractedPreference] = []
        ambiguous: list[str] = []
        warnings: list[str] = []
        # Utterance-level temporal scope from phrase cues; used as the default for
        # every extracted preference unless a call site overrides it.
        default_temporal = _temporal_for(lower)

        def add(
            field: str,
            value,
            raw: str,
            span: tuple[int, int] | None,
            strength: ConstraintStrength,
            polarity: str = "positive",
            confidence: float = 0.85,
            confirmation: ConfirmationStatus | None = None,
            temporal: str | None = None,
        ) -> None:
            prefs.append(
                ExtractedPreference(
                    field_name=field,
                    normalized_value=value,
                    raw_text=raw,
                    span_start=span[0] if span else None,
                    span_end=span[1] if span else None,
                    confidence=confidence,
                    confirmation_status=confirmation or _confirmation_for(strength, raw),
                    persistence_scope=PersistenceScope.ACTIVE_SEARCH,
                    proposed_strength=strength,
                    polarity=polarity,
                    temporal_scope=temporal if temporal is not None else default_temporal,
                )
            )

        # --- experience level -------------------------------------------------
        for alias in sorted(EXPERIENCE_LEVEL_ALIASES, key=len, reverse=True):
            m = re.search(rf"\b{re.escape(alias)}\b", lower)
            if m:
                level = canonical_level(alias)
                if level:
                    add("experience_level", level, alias, (m.start(), m.end()), ConstraintStrength.SOFT)
                    break

        # --- years of experience ---------------------------------------------
        ym = re.search(r"(\d+(?:\.\d+)?)\s*(?:\+)?\s*years?", lower)
        if ym:
            window = lower[max(0, ym.start() - 25): ym.end()]
            add("years_experience", float(ym.group(1)), ym.group(0), (ym.start(), ym.end()), _strength_for(window))

        # --- roles ------------------------------------------------------------
        for canonical, aliases in ROLE_FAMILIES.items():
            for alias in aliases:
                span = _find(lower, alias)
                if span:
                    window = _clause(lower, span[0], span[1])
                    polarity = "negative" if _is_negated(lower, span[0]) else "positive"
                    field = "excluded_roles" if polarity == "negative" else "target_roles"
                    add(field, canonical_role(canonical), alias, span,
                        ConstraintStrength.HARD if polarity == "negative" else _strength_for(window),
                        polarity=polarity)
                    break

        # --- skills (word-boundary match to avoid spurious substrings) -------
        seen_skills: set[str] = set()
        for canonical, aliases in SKILL_SYNONYMS.items():
            for alias in aliases:
                m = re.search(rf"(?<![a-z0-9]){re.escape(alias)}(?![a-z0-9])", lower)
                if m:
                    canon = canonical_skill(canonical)
                    if canon in seen_skills:
                        break
                    seen_skills.add(canon)
                    add("skills_have", canon, alias, (m.start(), m.end()), ConstraintStrength.SOFT,
                        confidence=0.8)
                    break

        # --- salary -----------------------------------------------------------
        for m in re.finditer(
            r"(rm|myr|sgd|usd|s\$|us\$|\$|€)?\s*(\d[\d,]*)\s*(k)?\b(?:\s*(?:per|/)\s*(month|year|hour))?",
            lower,
        ):
            cur_raw, num_raw, kilo = m.group(1), m.group(2), m.group(3)
            window = lower[max(0, m.start() - 30): m.end() + 10]
            amount = float(num_raw.replace(",", ""))
            if kilo:
                amount *= 1000
            has_money_ctx = any(
                x in window for x in ["salary", "rm", "myr", "sgd", "usd", "$", "€",
                                      "pay", "wage", "per month", "per year", "k "]
            )
            # A bare number in the typical monthly-salary range (and not a duration
            # like "3 years") is treated as a salary figure.
            looks_like_salary = (
                1000 <= amount <= 200000
                and "year" not in window and "experience" not in window and "month" not in window.replace("per month", "")
            )
            if not has_money_ctx and not cur_raw and not kilo and not looks_like_salary:
                continue
            if amount < 100:  # too small to be a salary; likely years/count
                continue
            currency = _CURRENCY_MAP.get(cur_raw.strip()) if cur_raw else None
            if currency is None:
                if "rm" in window or "myr" in window or "ringgit" in window:
                    currency = "MYR"
            # A stated minimum ("at least / above / minimum") is a hard threshold.
            if any(cue in window for cue in _THRESHOLD_CUES):
                strength = ConstraintStrength.HARD
            else:
                strength = _strength_for(window)
            add("salary_min", amount, m.group(0).strip(), (m.start(), m.end()), strength)
            if currency:
                add("salary_currency", currency, cur_raw or currency, (m.start(), m.end()),
                    ConstraintStrength.NOT_APPLICABLE)
            else:
                ambiguous.append("salary_currency")
                warnings.append("salary amount detected without an explicit currency")
            break

        # --- work mode --------------------------------------------------------
        for wm in _WORK_MODES:
            span = _find(lower, wm)
            if span:
                canon = "onsite" if wm in {"on-site", "on site", "onsite"} else wm
                window = _clause(lower, span[0], span[1])
                polarity = "negative" if _is_negated(lower, span[0]) else "positive"
                add("work_modes", canon, wm, span, _strength_for(window), polarity=polarity)
                break

        # --- locations --------------------------------------------------------
        for loc in sorted(_LOCATIONS, key=len, reverse=True):
            span = _find(lower, loc)
            if span:
                canon = _LOCATION_CANON[loc]
                window = _clause(lower, span[0], span[1])
                polarity = "negative" if _is_negated(lower, span[0]) else "positive"
                field = "excluded_locations" if polarity == "negative" else "preferred_locations"
                add(field, canon, loc, span, _strength_for(window), polarity=polarity)
                break

        return ExtractedPreferenceSet(
            utterance_id=utterance_id,
            preferences=prefs,
            detected_language="en",
            ambiguous_fields=sorted(set(ambiguous)),
            extraction_warnings=warnings,
        )


def _is_negated(text: str, start: int) -> bool:
    """Heuristic: is there a negation cue shortly before ``start``?"""
    window = text[max(0, start - 25): start]
    return any(cue in window for cue in _NEG_CUES)


_SENT_DELIMS = ".!?;"


def _clause(text: str, start: int, end: int) -> str:
    """Return the clause (from the last sentence delimiter) up to ``end``.

    This lets sentence-level modifiers such as "only" / "must" influence the
    strength of a constraint even when they are not immediately adjacent to it.
    """
    cut = 0
    for i in range(start - 1, -1, -1):
        if text[i] in _SENT_DELIMS:
            cut = i + 1
            break
    return text[cut:end]
