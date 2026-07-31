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
#: Cues that make a CATEGORICAL constraint binding -- a location, a work mode.
#:
#: Threshold wording ("at least", "minimum", "no less than") is deliberately NOT here, and
#: used to be. A threshold quantifies a NUMBER; it says nothing about the categorical
#: fields sharing its clause. While it lived in this list, "business analyst in Kuala
#: Lumpur at least RM4000" hardened the role and the location as well as the salary, and
#: "Something hybrid with at least RM4000" hardened the work mode -- in both cases turning
#: a stated preference into a filter that silently removed candidates. Threshold cues are
#: matched separately, against the numeric field they actually modify: see
#: :data:`_THRESHOLD_CUES`.
_HARD_CUES = ["must", "only", "cannot", "can't", "required", "absolutely",
              "mandatory", "non-negotiable"]
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


def _word_span(text: str, needle: str) -> tuple[int, int] | None:
    """Span of ``needle`` as a whole token, or ``None``.

    Alphanumeric boundaries rather than ``\\b`` so hyphenated aliases ("on-site") match as
    one token instead of breaking at the hyphen.
    """
    match = re.search(rf"(?<![a-z0-9]){re.escape(needle)}(?![a-z0-9])", text)
    return (match.start(), match.end()) if match else None


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
            window = _cue_window(lower, ym.start(), ym.end())
            add("years_experience", float(ym.group(1)), ym.group(0), (ym.start(), ym.end()), _strength_for(window))

        # --- roles ------------------------------------------------------------
        for canonical, aliases in ROLE_FAMILIES.items():
            for alias in aliases:
                span = _find(lower, alias)
                if span:
                    polarity = "negative" if _is_negated(lower, span[0]) else "positive"
                    field = "excluded_roles" if polarity == "negative" else "target_roles"
                    # A stated target role defines the RELEVANCE SCOPE, not a hard filter,
                    # so a positive role is never proposed as binding. This is the declared
                    # semantics of the frozen reference set, not a convenience: across all
                    # 42 scenarios the declared hard fields are only salary_min,
                    # preferred_locations and work_modes -- target_roles is declared hard
                    # zero times, including in scenarios whose text says "I only want a
                    # data analyst role". Role exclusivity is enforced by scope (a role
                    # mismatch caps the graded relevance at 0), which is why hardening it
                    # here would double-count the same intent and shrink the candidate pool
                    # a second time.
                    #
                    # A NEGATIVE role is different: it becomes excluded_roles, and
                    # exclusions are a hard mechanism by design.
                    strength = (ConstraintStrength.HARD if polarity == "negative"
                                else ConstraintStrength.SOFT)
                    add(field, canonical_role(canonical), alias, span, strength,
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
            # Two windows, because they answer different questions. ``window`` is a fixed
            # character span used to decide whether this number is money at all and which
            # currency it is in -- that evidence can sit outside the clause ("salary" in an
            # earlier clause still tells us the number is a salary). ``cue_window`` is the
            # clause, used only for strength, so a neighbouring field's cue cannot decide
            # whether this threshold is binding.
            window = lower[max(0, m.start() - 30): m.end() + 10]
            cue_window = _cue_window(lower, m.start(), m.end())
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
            if any(cue in cue_window for cue in _THRESHOLD_CUES):
                strength = ConstraintStrength.HARD
            else:
                strength = _strength_for(cue_window)
            add("salary_min", amount, m.group(0).strip(), (m.start(), m.end()), strength)
            if currency:
                add("salary_currency", currency, cur_raw or currency, (m.start(), m.end()),
                    ConstraintStrength.NOT_APPLICABLE)
            else:
                ambiguous.append("salary_currency")
                warnings.append("salary amount detected without an explicit currency")
            break

        # --- work mode --------------------------------------------------------
        # Every stated mode, not just the first. This loop used to ``break`` on its first
        # match, so "remote or hybrid" yielded ["remote"] -- which does not merely lose
        # information, it asserts the candidate ruled hybrid OUT. Dedupe is by canonical
        # value because three aliases collapse onto "onsite".
        seen_modes: set[str] = set()
        for wm in _WORK_MODES:
            span = _word_span(lower, wm)
            if span is None:
                continue
            canon = "onsite" if wm in {"on-site", "on site", "onsite"} else wm
            if canon in seen_modes:
                continue
            seen_modes.add(canon)
            window = _cue_window(lower, span[0], span[1])
            polarity = "negative" if _is_negated(lower, span[0]) else "positive"
            add("work_modes", canon, wm, span, _strength_for(window), polarity=polarity)

        # --- locations --------------------------------------------------------
        # Same fix, plus word-boundary matching. Substring matching was survivable while
        # the loop stopped at the first hit; without the ``break`` the two-letter alias
        # "kl" would match inside unrelated words ("weekly") and invent a location.
        seen_locations: set[str] = set()
        for loc in sorted(_LOCATIONS, key=len, reverse=True):
            span = _word_span(lower, loc)
            if span is None:
                continue
            canon = _LOCATION_CANON[loc]
            if canon in seen_locations:
                continue
            seen_locations.add(canon)
            window = _cue_window(lower, span[0], span[1])
            polarity = "negative" if _is_negated(lower, span[0]) else "positive"
            field = "excluded_locations" if polarity == "negative" else "preferred_locations"
            add(field, canon, loc, span, _strength_for(window), polarity=polarity)

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

#: Clause boundaries for cue scoping. Commas are included alongside sentence delimiters
#: because a comma is what separates two independently-modified assertions in the
#: utterances this extractor sees: in "Cyberjaya only, ideally around RM7400" the "only"
#: belongs to the location and the "ideally" to the salary, and any window that spans the
#: comma gets both wrong at once.
_CLAUSE_DELIMS = _SENT_DELIMS + ","

#: Coordinating words that begin a new assertion without a comma. A cue on the far side of
#: one of these modifies that assertion, not the previous one -- "onsite is required though
#: hybrid would also work" states a requirement and then a concession.
_CLAUSE_BREAKERS = (" and ", " but ", " though ", " although ", " while ", " however ",
                    " whereas ", " plus ")


def _cue_window(text: str, start: int, end: int) -> str:
    """Return the clause AROUND ``[start:end]``, for strength classification.

    Replaces an earlier helper that ran from the last sentence delimiter up to ``end``.
    That had two consequences, and they pulled in opposite directions, which is why both
    had to be fixed together rather than one at a time.

    It ended at the value, so a POST-positioned cue was structurally invisible: "onsite
    only" produced the window "onsite" and was classified SOFT, while "at least RM4000" in
    the same sentence was correctly HARD only because English puts that cue in front of
    its number. Widening the window forward fixes that.

    It also began at the sentence delimiter, so the window held every earlier field in the
    sentence and one field's cue decided another's strength. Widening forward without
    narrowing backward would have made that worse, so the window is now bounded on BOTH
    sides by :data:`_CLAUSE_DELIMS` and :data:`_CLAUSE_BREAKERS`.

    The window deliberately still extends beyond the matched token: a cue is a property of
    the clause, not of the word next to it. What it must not do is cross into a clause that
    modifies something else.
    """
    left = 0
    for index in range(start - 1, -1, -1):
        if text[index] in _CLAUSE_DELIMS:
            left = index + 1
            break
    right = len(text)
    for index in range(end, len(text)):
        if text[index] in _CLAUSE_DELIMS:
            right = index
            break

    for breaker in _CLAUSE_BREAKERS:
        # A breaker before the value moves the left edge in; one after it moves the right
        # edge in. Only the closest on each side matters.
        cut = text.rfind(breaker, left, start)
        if cut != -1:
            left = max(left, cut + len(breaker))
        cut = text.find(breaker, end, right)
        if cut != -1:
            right = min(right, cut)
    return text[left:right]
