"""Deterministic clarification policy.

The policy (not the LLM) decides *whether* and *what* to clarify. The LLM may
later phrase the chosen slot as natural language, but slot selection is
deterministic and driven by:

    priority = expected_decision_impact * uncertainty * coverage_gain

At most 1-2 high-value questions are asked per turn.
"""

from __future__ import annotations

from ..config import AppConfig
from ..domain.dialogue import ClarificationAction, PreferenceConflict
from ..domain.job import ActiveSearchState
from ..utils.hashing import content_id
from ..utils.time import utcnow

# Impact weights per field (how much resolving it changes the decision).
_FIELD_IMPACT = {
    "years_experience": 0.9,
    "preferred_locations": 0.8,
    "salary_currency": 0.85,
    "target_roles": 0.95,
    "work_authorizations": 0.8,
    "salary_min": 0.6,
    "work_modes": 0.4,
    "experience_level": 0.5,
}


class ClarificationPolicy:
    """Chooses at most one clarification action for the current turn."""

    name = "clarification_policy"

    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def decide(
        self,
        active: ActiveSearchState,
        conflicts: list[PreferenceConflict],
        ambiguous_fields: list[str],
    ) -> ClarificationAction | None:
        """Return a clarification action, or None if none is warranted."""
        candidates: list[tuple[float, str, str, list[str], list[str]]] = []

        # 1) High-impact unresolved conflicts (e.g. factual year mismatch).
        for c in conflicts:
            if c.resolution == "ask_clarification" and c.impact == "high":
                impact = _FIELD_IMPACT.get(c.field_name, 0.5)
                score = impact * 1.0 * 0.9
                candidates.append(
                    (score, c.field_name, f"conflict_{c.conflict_type}", [c.conflict_id],
                     self._options_for(c.field_name, active))
                )

        # 2) Missing role target -> retrieval would be far too broad.
        if not active.target_roles:
            candidates.append(
                (_FIELD_IMPACT["target_roles"] * 1.0 * 1.0, "target_roles",
                 "missing_role_target", [], [])
            )

        # 3) Ambiguous unit/currency for a stated salary.
        if "salary_currency" in ambiguous_fields and active.salary_min is not None:
            candidates.append(
                (_FIELD_IMPACT["salary_currency"] * 0.9 * 0.8, "salary_currency",
                 "ambiguous_salary_currency", [], ["MYR", "SGD", "USD"])
            )

        # 4) Location temporal override marked as clarification-required.
        for field in active.clarification_required_fields:
            impact = _FIELD_IMPACT.get(field, 0.5)
            candidates.append(
                (impact * 0.8 * 0.7, field, "clarification_required_field", [],
                 self._options_for(field, active))
            )

        if not candidates:
            return None

        candidates.sort(key=lambda t: -t[0])
        score, field, reason, conflict_ids, options = candidates[0]
        return ClarificationAction(
            clarification_id=content_id("clar", field, reason),
            target_fields=[field],
            reason_code=reason,
            priority_score=round(score, 4),
            question_text=self._phrase(field, reason, options, active),
            options=options,
            related_conflict_ids=conflict_ids,
            created_at=utcnow(),
        )

    def _options_for(self, field: str, active: ActiveSearchState) -> list[str]:
        if field == "preferred_locations":
            return active.preferred_locations or []
        if field == "work_modes":
            return ["onsite", "hybrid", "remote"]
        return []

    def _phrase(self, field: str, reason: str, options: list[str], active: ActiveSearchState) -> str:
        """Deterministic fallback phrasing (LLM may replace this in hybrid mode)."""
        if reason == "missing_role_target":
            return "What kind of role are you looking for (for example: data analyst, business analyst)?"
        if field == "years_experience":
            return (
                "Your profile and your message state different years of experience. "
                "How many years of experience should I use for this search?"
            )
        if field == "salary_currency":
            opts = " / ".join(options) if options else "MYR"
            return f"You mentioned a salary figure. Which currency is that in ({opts})?"
        if field == "preferred_locations":
            opts = ", ".join(options) if options else "the stated location"
            return f"Should I search only in {opts}, or consider other locations too?"
        return f"Could you confirm your preference for {field.replace('_', ' ')}?"
