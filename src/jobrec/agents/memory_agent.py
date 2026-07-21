"""Candidate Profile Memory Agent.

Manages CandidateState (stable / confirmed long-term info) and DialogueState
(turn-level evidence). It applies evidence priority, detects conflicts, and
performs version control. It never writes low-confidence inferences into
long-term memory automatically, and current-search overrides never silently
overwrite long-term values.
"""

from __future__ import annotations

from datetime import datetime

from ..config import AppConfig
from ..domain.candidate import CandidateState
from ..domain.dialogue import DialogueState, DialogueTurn, PreferenceConflict
from ..domain.enums import (
    ConfirmationStatus,
    ConstraintStrength,
    EvidenceSource,
    PersistenceScope,
)
from ..domain.evidence import EvidenceItem, PreferenceValue
from ..domain.extraction import ExtractedPreference, ExtractedPreferenceSet
from ..evidence_store import EvidenceStore
from ..utils.hashing import content_id
from ..utils.time import utcnow

# Which extracted field maps to which CandidateState attribute (list vs scalar).
_LIST_FIELDS = {
    "skills_have": "skills",
    "target_roles": "target_roles",
    "preferred_locations": "preferred_locations",
    "work_modes": "work_modes",
    "industries": "industries",
    "employment_types": "employment_types",
    "work_authorizations": "work_authorizations",
    "excluded_roles": "excluded_roles",
    "excluded_locations": "excluded_locations",
    "excluded_industries": "excluded_industries",
}
_SCALAR_FIELDS = {
    "years_experience": "years_experience",
    "experience_level": "experience_level",
    "salary_min": "salary_min",
    "salary_currency": "salary_currency",
    "education_level": "education_level",
}


class MemoryAgent:
    """Owns state creation, evidence generation, conflict detection, versioning."""

    name = "memory_agent"

    def __init__(self, store: EvidenceStore, config: AppConfig) -> None:
        self.store = store
        self.config = config

    # ------------------------------------------------------------- profile
    def create_candidate_state(self, profile: dict) -> CandidateState:
        """Build CandidateState v1 from a structured profile, with evidence."""
        candidate_id = profile["candidate_id"]
        now = utcnow()

        def pv(field: str, value, confidence: float = 0.95) -> PreferenceValue:
            ev = self.store.register_field(
                EvidenceSource.PROFILE, candidate_id, field, value,
                confidence=confidence, confirmation=ConfirmationStatus.CONFIRMED,
                scope=PersistenceScope.LONG_TERM,
            )
            return PreferenceValue(
                value=value, evidence_ids=[ev.evidence_id],
                confirmation_status=ConfirmationStatus.CONFIRMED,
                persistence_scope=PersistenceScope.LONG_TERM,
                effective_from=now, confidence=confidence, is_active=True,
            )

        def pv_list(field: str, values) -> list[PreferenceValue]:
            return [pv(field, v) for v in (values or [])]

        return CandidateState(
            candidate_id=candidate_id,
            version=1,
            updated_at=now,
            skills=pv_list("skills", profile.get("skills")),
            education_level=pv("education_level", profile["education_level"]) if profile.get("education_level") else None,
            years_experience=pv("years_experience", float(profile["years_experience"])) if profile.get("years_experience") is not None else None,
            experience_level=pv("experience_level", profile["experience_level"]) if profile.get("experience_level") else None,
            target_roles=pv_list("target_roles", profile.get("target_roles")),
            preferred_locations=pv_list("preferred_locations", profile.get("preferred_locations")),
            salary_min=pv("salary_min", float(profile["salary_min"])) if profile.get("salary_min") is not None else None,
            salary_currency=pv("salary_currency", profile["salary_currency"]) if profile.get("salary_currency") else None,
            work_modes=pv_list("work_modes", profile.get("work_modes")),
            industries=pv_list("industries", profile.get("industries")),
            employment_types=pv_list("employment_types", profile.get("employment_types")),
            work_authorizations=pv_list("work_authorizations", profile.get("work_authorizations")),
            excluded_roles=pv_list("excluded_roles", profile.get("excluded_roles")),
            excluded_industries=pv_list("excluded_industries", profile.get("excluded_industries")),
            excluded_locations=pv_list("excluded_locations", profile.get("excluded_locations")),
        )

    # --------------------------------------------------------------- turns
    def append_turn(
        self,
        dialogue: DialogueState,
        speaker: str,
        text: str,
        evidence_ids: list[str] | None = None,
        action_type: str | None = None,
    ) -> DialogueState:
        """Return a NEW DialogueState version with the turn appended."""
        turn_index = len(dialogue.turns)
        turn = DialogueTurn(
            turn_id=content_id("turn", dialogue.session_id, str(turn_index)),
            session_id=dialogue.session_id,
            turn_index=turn_index,
            speaker=speaker,  # type: ignore[arg-type]
            text=text,
            created_at=utcnow(),
            evidence_ids=evidence_ids or [],
            action_type=action_type,
        )
        return dialogue.model_copy(update={
            "version": dialogue.version + 1,
            "turns": [*dialogue.turns, turn],
        })

    def latest_turn_id(self, dialogue: DialogueState) -> str | None:
        return dialogue.turns[-1].turn_id if dialogue.turns else None

    def register_profile_evidence(self, candidate: CandidateState) -> None:
        """Re-register a candidate's profile evidence into the current store.

        Evidence ids are content-addressed, so re-registering the same profile
        field/value reproduces the identical id (idempotent). This ensures
        profile-derived claims resolve even when the CandidateState was created
        with a different (throwaway) store.
        """
        cid = candidate.candidate_id

        def reg(field: str, pv) -> None:
            self.store.register_field(
                EvidenceSource.PROFILE, cid, field, pv.value,
                confidence=pv.confidence, confirmation=pv.confirmation_status,
                scope=pv.persistence_scope,
            )

        list_attrs = {
            "skills": candidate.skills, "target_roles": candidate.target_roles,
            "preferred_locations": candidate.preferred_locations,
            "work_modes": candidate.work_modes, "industries": candidate.industries,
            "employment_types": candidate.employment_types,
            "work_authorizations": candidate.work_authorizations,
            "excluded_roles": candidate.excluded_roles,
            "excluded_industries": candidate.excluded_industries,
            "excluded_locations": candidate.excluded_locations,
        }
        for field, values in list_attrs.items():
            for pv in values:
                reg(field, pv)
        for field, pv in {
            "education_level": candidate.education_level,
            "years_experience": candidate.years_experience,
            "experience_level": candidate.experience_level,
            "salary_min": candidate.salary_min,
            "salary_currency": candidate.salary_currency,
        }.items():
            if pv is not None:
                reg(field, pv)

    # ------------------------------------------------------------ evidence
    def build_dialogue_evidence(
        self, extraction: ExtractedPreferenceSet, session_id: str, turn_id: str
    ) -> list[EvidenceItem]:
        """Turn extracted preferences into registered EvidenceItems."""
        from ..agents.candidate_understanding import EXTRACTOR_NAME, EXTRACTOR_VERSION

        items: list[EvidenceItem] = []
        for pref in extraction.preferences:
            span = (
                (pref.span_start, pref.span_end)
                if pref.span_start is not None and pref.span_end is not None
                else None
            )
            item = self.store.register_field(
                EvidenceSource.DIALOGUE, session_id, pref.field_name, pref.normalized_value,
                confidence=pref.confidence, confirmation=pref.confirmation_status,
                scope=pref.persistence_scope, raw_text=pref.raw_text, turn_id=turn_id,
                span=span, extractor_name=EXTRACTOR_NAME, extractor_version=EXTRACTOR_VERSION,
            )
            items.append(item)
        return items

    # ----------------------------------------------------------- conflicts
    def detect_conflicts(
        self,
        candidate: CandidateState,
        extraction: ExtractedPreferenceSet,
        evidence_by_field: dict[str, list[str]],
    ) -> list[PreferenceConflict]:
        """Detect conflicts between incoming dialogue evidence and long-term state."""
        conflicts: list[PreferenceConflict] = []
        now = utcnow()

        for pref in extraction.preferences:
            if pref.polarity == "negative":
                continue
            existing_ids, existing_vals, kind = self._existing(candidate, pref.field_name)
            if not existing_vals:
                continue
            incoming_ids = evidence_by_field.get(pref.field_name, [])

            conflict = self._classify_conflict(
                pref, existing_vals, existing_ids, incoming_ids, now
            )
            if conflict is not None:
                conflicts.append(conflict)
        return conflicts

    def _existing(self, candidate: CandidateState, field: str):
        if field in _LIST_FIELDS:
            attr = getattr(candidate, _LIST_FIELDS[field])
            vals = [pvv.value for pvv in attr]
            ids = [eid for pvv in attr for eid in pvv.evidence_ids]
            return ids, vals, "list"
        if field in _SCALAR_FIELDS:
            pvv = getattr(candidate, _SCALAR_FIELDS[field])
            if pvv is None:
                return [], [], "scalar"
            return list(pvv.evidence_ids), [pvv.value], "scalar"
        return [], [], "none"

    def _classify_conflict(
        self,
        pref: ExtractedPreference,
        existing_vals: list,
        existing_ids: list[str],
        incoming_ids: list[str],
        now: datetime,
    ) -> PreferenceConflict | None:
        field = pref.field_name
        incoming = pref.normalized_value

        def mk(conflict_type, impact, resolution, rule_id) -> PreferenceConflict:
            return PreferenceConflict(
                conflict_id=content_id("cf", field, str(incoming), *existing_ids),
                field_name=field,
                existing_evidence_ids=existing_ids,
                incoming_evidence_ids=incoming_ids,
                conflict_type=conflict_type,
                impact=impact,
                resolution=resolution,
                resolution_rule_id=rule_id,
                created_at=now,
            )

        # Years of experience: a factual mismatch -> must clarify, never override.
        if field == "years_experience":
            try:
                if abs(float(incoming) - float(existing_vals[0])) >= 1.0:
                    return mk("value_mismatch", "high", "ask_clarification", "conflict.years")
            except (TypeError, ValueError):
                return None
            return None

        # Location: current explicit statement overrides for THIS search only.
        if field == "preferred_locations":
            if incoming not in existing_vals:
                impact = "high" if pref.proposed_strength == ConstraintStrength.HARD else "medium"
                return mk("temporal_override", impact, "use_current_for_search", "conflict.location")
            return None

        # Salary: most recent explicit statement controls the active search.
        if field == "salary_min":
            if existing_vals and float(incoming) != float(existing_vals[0]):
                return mk("temporal_override", "low", "use_current_for_search", "conflict.salary")
            return None

        # Work mode: additive preference -> merge values.
        if field == "work_modes":
            if incoming not in existing_vals:
                return mk("scope_mismatch", "low", "merge_values", "conflict.work_mode")
            return None

        # Experience level mismatch.
        if field == "experience_level":
            if existing_vals and incoming != existing_vals[0]:
                return mk("value_mismatch", "medium", "use_current_for_search", "conflict.level")
            return None

        return None
