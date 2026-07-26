"""Scenario-level simulated user for the clarification dialogue loop (R7.1).

The :class:`SimulatedUser` answers a system clarification question deterministically
(no LLM) using the scenario *reference*: the ``acceptable_slots`` the scenario declares
as answerable, the candidate ``profile`` shipped with the scenario, and sensible
per-field defaults for slots the scenario says are answerable but does not pin to a
concrete value.

Given a :class:`~jobrec.domain.dialogue.ClarificationAction` (or an equivalent mapping),
it maps the clarification's ``target_fields`` / ``reason_code`` to a natural-language
answer utterance that, when re-extracted by the rule extractor, supplies the missing
constraint. It returns ``None`` when it cannot answer the clarification -- modelling a
user who cannot or will not answer, which forces the loop to terminate.

This module is intentionally pure and deterministic: it feeds the clarification loop in
task 10.2 but does not implement the loop, the dialogue trace, or scoring.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .scenarios import Scenario

# Per-field fallback values used when a scenario declares a slot answerable (via
# ``acceptable_slots``) but the profile does not pin a concrete value. These are the
# canonical, catalog-consistent defaults for the evaluation domain.
_DEFAULTS: dict[str, Any] = {
    "target_roles": "data analyst",
    "preferred_locations": "Kuala Lumpur",
    "work_modes": "hybrid",
    "salary_currency": "MYR",
    "salary_min": 4000,
    "experience_level": "junior",
}


def _clar_attr(clarification: Any, name: str, default: Any) -> Any:
    """Read ``name`` from a ClarificationAction object or a mapping equivalently."""
    if isinstance(clarification, Mapping):
        return clarification.get(name, default)
    return getattr(clarification, name, default)


class SimulatedUser:
    """Answers clarification questions from a scenario reference.

    Accepts either a :class:`Scenario` object or the raw scenario ``dict`` (as loaded
    from the scenario JSONL, which carries a ``profile`` mapping that the ``Scenario``
    dataclass does not expose). This keeps the simulated user usable directly by the
    experiment runner regardless of which representation it holds.
    """

    def __init__(self, scenario: Scenario | Mapping[str, Any]) -> None:
        self.scenario = scenario
        if isinstance(scenario, Mapping):
            self.acceptable_slots: list[str] = list(scenario.get("acceptable_slots", []))
            self.profile: dict[str, Any] = dict(scenario.get("profile", {}))
            self.expected_response: str = (
                scenario.get("expects", {}).get("response_type")
                or scenario.get("expected_response", "recommendation")
            )
        else:
            self.acceptable_slots = list(getattr(scenario, "acceptable_slots", []))
            # The Scenario dataclass has no ``profile`` field; tolerate its absence.
            self.profile = dict(getattr(scenario, "profile", {}) or {})
            self.expected_response = getattr(scenario, "expected_response", "recommendation")

    # -- public API -------------------------------------------------------------

    def answer(
        self,
        clarification: Any,
        asked_slots: set[str] | None = None,
    ) -> tuple[str, str] | None:
        """Return ``(utterance, slot)`` answering ``clarification``, or ``None``.

        The clarification's ``target_fields`` are considered in order. A field is
        answerable when the scenario declares it in ``acceptable_slots`` or the profile
        carries a concrete value for it. Among answerable fields, one that has not yet
        been asked (per ``asked_slots``) is preferred, so the loop makes progress; if
        every answerable field has already been asked, the first answerable field is
        still returned so the loop's repeated-slot guard can act.

        Returns ``None`` when none of the target fields are answerable from the scenario
        reference (a user who cannot/won't answer), which terminates the loop.
        """
        asked = asked_slots or set()
        target_fields = list(_clar_attr(clarification, "target_fields", []) or [])
        if not target_fields:
            return None

        answerable = [f for f in target_fields if self._is_answerable(f)]
        if not answerable:
            return None

        # Prefer a not-yet-asked answerable slot to make progress.
        slot = next((f for f in answerable if f not in asked), answerable[0])
        value = self._value_for(slot)
        if value is None:
            return None
        utterance = self._utterance_for(slot, value)
        if utterance is None:
            return None
        return utterance, slot

    # -- internals --------------------------------------------------------------

    def _profile_value(self, field: str) -> Any:
        """Return a scalar value for ``field`` from the profile, or ``None``."""
        if field not in self.profile:
            return None
        raw = self.profile[field]
        if isinstance(raw, (list, tuple)):
            return raw[0] if raw else None
        return raw

    def _is_answerable(self, field: str) -> bool:
        """A field is answerable if the scenario lists it or the profile supplies it."""
        if field in self.acceptable_slots and field in _DEFAULTS:
            return True
        if field in self.acceptable_slots and self._profile_value(field) is not None:
            return True
        return self._profile_value(field) is not None

    def _value_for(self, field: str) -> Any:
        """Resolve the answer value: profile first, then the scenario-domain default."""
        val = self._profile_value(field)
        if val is not None:
            return val
        if field in self.acceptable_slots:
            return _DEFAULTS.get(field)
        return None

    def _utterance_for(self, field: str, value: Any) -> str | None:
        """Phrase ``value`` for ``field`` so the rule extractor re-extracts it."""
        if field in {"target_roles", "excluded_roles"}:
            return f"I'm looking for a {str(value).lower()} role."
        if field in {"preferred_locations", "excluded_locations"}:
            return f"For this search, {value} please."
        if field == "work_modes":
            return f"{str(value).capitalize()} is fine."
        if field == "salary_currency":
            return f"The salary figure I mentioned is in {value}."
        if field == "salary_min":
            return f"At least RM{int(float(value))} per month."
        if field == "years_experience":
            return f"I have {value} years of experience."
        if field == "experience_level":
            return f"I'm at a {value} level."
        if field == "work_authorizations":
            return "I have the right to work here."
        # Unknown field: emit a generic confirmation naming the value.
        return f"{value}."
