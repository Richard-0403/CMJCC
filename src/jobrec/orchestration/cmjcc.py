"""Candidate-Memory and Job-Context Connector (CMJCC).

The CMJCC is a state-coordination / orchestration component, NOT a recommender.
It validates state, merges evidence into an active-search view, resolves
conflicts, decides clarifications, and produces a constraint bundle. It does not
generate natural language, run retrieval, or call an LLM for eligibility.

Determinism: for the same state versions, input text, config and model
responses, the output is identical (content-addressed ids prevent duplicate
writes for the same turn).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from ..agents.job_context_agent import JobContextAgent
from ..agents.memory_agent import MemoryAgent
from ..config import AppConfig
from ..domain.candidate import CandidateState
from ..domain.constraints import JobContextState
from ..domain.dialogue import ClarificationAction, DialogueState, PreferenceConflict
from ..domain.enums import ConstraintStrength
from ..domain.extraction import ExtractedPreferenceSet
from ..domain.handoff import EvidenceLogEntry
from ..domain.job import ActiveSearchState
from ..evidence_store import EvidenceStore
from ..llm.field_validation import salary_amount
from ..utils.hashing import content_id
from ..utils.time import utcnow
from .clarification_policy import ClarificationPolicy
from .feature_flags import FeatureFlags

# Active-search fields that are lists (merged as unions) vs scalars (overridden).
_ACTIVE_LIST_FIELDS = [
    "target_roles", "skills_have", "preferred_locations",
    "work_modes", "employment_types", "work_authorizations",
]
_ACTIVE_SCALAR_FIELDS = ["salary_min", "salary_currency", "experience_level", "years_experience"]

# Map CandidateState attributes -> active-search field names.
_PROFILE_TO_ACTIVE_LIST = {
    "target_roles": "target_roles",
    "skills": "skills_have",
    "preferred_locations": "preferred_locations",
    "work_modes": "work_modes",
    "employment_types": "employment_types",
    "work_authorizations": "work_authorizations",
}
_PROFILE_TO_ACTIVE_SCALAR = {
    "salary_min": "salary_min",
    "salary_currency": "salary_currency",
    "experience_level": "experience_level",
    "years_experience": "years_experience",
}


@dataclass
class CMJCCInput:
    candidate_state: CandidateState
    dialogue_state: DialogueState
    extracted_preferences: ExtractedPreferenceSet
    catalog_snapshot_id: str
    config: AppConfig
    run_id: str


@dataclass
class CMJCCOutput:
    candidate_state: CandidateState
    dialogue_state: DialogueState
    active_search_state: ActiveSearchState
    job_context_state: JobContextState | None
    conflicts: list[PreferenceConflict]
    clarification_action: ClarificationAction | None
    evidence_log_entries: list[EvidenceLogEntry] = field(default_factory=list)


class CMJCC:
    """Coordinates candidate memory and job context into an inspectable view."""

    name = "cmjcc"

    def __init__(self, store: EvidenceStore, config: AppConfig) -> None:
        self.store = store
        self.config = config
        self.flags = FeatureFlags.from_config(config)
        self.memory = MemoryAgent(store, config)
        self.clarifier = ClarificationPolicy(config)
        self.job_context = JobContextAgent(config)

    def run(self, inp: CMJCCInput) -> CMJCCOutput:
        logs: list[EvidenceLogEntry] = []
        now = utcnow()

        def log(stage: str, event: str, inputs=None, outputs=None, rule=None) -> None:
            logs.append(EvidenceLogEntry(
                log_id=content_id("log", inp.run_id, stage, event, str(len(logs))),
                run_id=inp.run_id, stage=stage, event_type=event, actor=self.name,
                source_ids=[], input_object_ids=inputs or [], output_object_ids=outputs or [],
                rule_id=rule, status="success", created_at=utcnow(),
            ))

        # 1) Build dialogue evidence for the current turn (if in scope).
        turn_id = self.memory.latest_turn_id(inp.dialogue_state) or "no-turn"
        evidence_by_field: dict[str, list[str]] = {}
        if self.flags.use_current_turn:
            items = self.memory.build_dialogue_evidence(
                inp.extracted_preferences, inp.dialogue_state.session_id, turn_id
            )
            for it in items:
                evidence_by_field.setdefault(it.field_name, []).append(it.evidence_id)
            log("understanding", "dialogue_evidence_built",
                outputs=[i.evidence_id for i in items], rule="cmjcc.build_evidence")

        # 2) Conflict detection against long-term profile.
        conflicts: list[PreferenceConflict] = []
        if self.flags.use_current_turn and self.flags.use_persistent_memory:
            conflicts = self.memory.detect_conflicts(
                inp.candidate_state, inp.extracted_preferences, evidence_by_field
            )
            log("validating", "conflicts_detected",
                outputs=[c.conflict_id for c in conflicts], rule="cmjcc.detect_conflicts")

        # 2b) Long-term write-back into CandidateState (single shared code path).
        # Only the current turn's confirmed, durable ("from now on ...") preferences
        # are eligible. We respect use_current_turn so nothing is written back when the
        # current utterance is out of scope for the variant; persist_confirmed_updates
        # and use_persistent_memory gate the mechanism per the resolved flag matrix.
        candidate_state = inp.candidate_state
        if (
            self.flags.persist_confirmed_updates
            and self.flags.use_persistent_memory
            and self.flags.use_current_turn
        ):
            updated = self.memory.apply_confirmed_updates(
                inp.candidate_state, inp.extracted_preferences, conflicts, now
            )
            if updated is not inp.candidate_state:
                candidate_state = updated
                log("memory_updated", "candidate_state_written",
                    outputs=[f"{updated.candidate_id}:v{updated.version}"],
                    rule="cmjcc.writeback")

        # 3) Build the active-search view by merging profile + current turn.
        active = self._build_active_search(inp, evidence_by_field, conflicts, now)
        log("context_built", "active_search_built", outputs=[active.active_search_id],
            rule="cmjcc.build_active_search")

        # 4) Clarification decision (deterministic policy).
        clarification = self.clarifier.decide(
            active, conflicts, inp.extracted_preferences.ambiguous_fields
        )
        if clarification is not None:
            log("clarification_required", "clarification_selected",
                outputs=[clarification.clarification_id], rule="cmjcc.clarify")

        # 5) Constraint bundle (only under explicit orchestration).
        job_context: JobContextState | None = None
        if self.flags.explicit_constraint_orchestration:
            job_context = self.job_context.build_context(active, inp.catalog_snapshot_id)
            log("context_built", "constraints_built", outputs=[job_context.context_id],
                rule="cmjcc.build_constraints")

        # 6) Persist conflicts into a new DialogueState version.
        unresolved = sorted({
            f for c in conflicts if c.resolution in ("ask_clarification", "unresolved")
            for f in [c.field_name]
        })
        dialogue = inp.dialogue_state.model_copy(update={
            "version": inp.dialogue_state.version + 1,
            "conflicts": [*inp.dialogue_state.conflicts, *conflicts],
            "unresolved_slots": sorted(set(inp.dialogue_state.unresolved_slots) | set(unresolved)),
            "active_search_id": active.active_search_id,
        })

        return CMJCCOutput(
            candidate_state=candidate_state,
            dialogue_state=dialogue,
            active_search_state=active,
            job_context_state=job_context,
            conflicts=conflicts,
            clarification_action=clarification,
            evidence_log_entries=logs,
        )

    # --------------------------------------------------------------- merge
    def _build_active_search(
        self,
        inp: CMJCCInput,
        evidence_by_field: dict[str, list[str]],
        conflicts: list[PreferenceConflict],
        now: datetime,
    ) -> ActiveSearchState:
        cand = inp.candidate_state
        list_values: dict[str, list] = {f: [] for f in _ACTIVE_LIST_FIELDS}
        scalar_values: dict[str, object] = {f: None for f in _ACTIVE_SCALAR_FIELDS}
        field_evidence: dict[str, list[str]] = {}
        strengths: dict[str, ConstraintStrength] = {}
        exclusions: dict[str, list[str]] = {"roles": [], "locations": [], "industries": []}
        clar_required: list[str] = []

        def add_ev(field: str, ids: list[str]) -> None:
            if not ids:
                return
            field_evidence.setdefault(field, [])
            for i in ids:
                if i not in field_evidence[field]:
                    field_evidence[field].append(i)

        # ---- profile contributions (long-term) --------------------------
        if self.flags.use_profile:
            for prof_attr, active_field in _PROFILE_TO_ACTIVE_LIST.items():
                for pv in getattr(cand, prof_attr):
                    if pv.value not in list_values[active_field]:
                        list_values[active_field].append(pv.value)
                    add_ev(active_field, pv.evidence_ids)
                    strengths.setdefault(active_field, ConstraintStrength.SOFT)
            for prof_attr, active_field in _PROFILE_TO_ACTIVE_SCALAR.items():
                pv = getattr(cand, prof_attr)
                if pv is not None:
                    scalar_values[active_field] = pv.value
                    add_ev(active_field, pv.evidence_ids)
                    strengths.setdefault(active_field, ConstraintStrength.SOFT)
            # profile exclusions
            for pv in cand.excluded_roles:
                exclusions["roles"].append(str(pv.value).lower())
            for pv in cand.excluded_locations:
                exclusions["locations"].append(str(pv.value).lower())
            for pv in cand.excluded_industries:
                exclusions["industries"].append(str(pv.value).lower())

        # ---- current-turn contributions (active-search overrides) --------
        year_conflict_fields = {c.field_name for c in conflicts if c.resolution == "ask_clarification"}
        if self.flags.use_current_turn:
            # locations from current turn OVERRIDE profile locations for this search.
            current_locations = [
                p.normalized_value for p in inp.extracted_preferences.preferences
                if p.field_name == "preferred_locations" and p.polarity == "positive"
            ]
            if current_locations:
                list_values["preferred_locations"] = list(dict.fromkeys(current_locations))

            for pref in inp.extracted_preferences.preferences:
                f = pref.field_name
                ids = evidence_by_field.get(f, [])
                if pref.polarity == "negative":
                    if f in ("excluded_roles", "target_roles"):
                        exclusions["roles"].append(str(pref.normalized_value).lower())
                    elif f in ("excluded_locations", "preferred_locations"):
                        exclusions["locations"].append(str(pref.normalized_value).lower())
                    elif f == "work_modes":
                        pass  # negative work mode handled as exclusion of that mode
                    continue

                if f in _ACTIVE_LIST_FIELDS:
                    if f != "preferred_locations":  # locations already overridden above
                        if pref.normalized_value not in list_values[f]:
                            list_values[f].append(pref.normalized_value)
                    add_ev(f, ids)
                    strengths[f] = _stronger(strengths.get(f), pref.proposed_strength)
                elif f in _ACTIVE_SCALAR_FIELDS:
                    # years_experience with a factual conflict is NOT silently overridden.
                    if f == "years_experience" and f in year_conflict_fields:
                        clar_required.append(f)
                        add_ev(f, ids)
                        continue
                    scalar_values[f] = pref.normalized_value
                    add_ev(f, ids)
                    strengths[f] = _stronger(strengths.get(f), pref.proposed_strength)

        # location override that conflicts and is hard -> keep for search, no forced clarify
        for c in conflicts:
            if c.resolution == "ask_clarification" and c.field_name not in clar_required:
                clar_required.append(c.field_name)

        # ---- classify hard / soft / unknown -----------------------------
        present_fields = [f for f in _ACTIVE_LIST_FIELDS if list_values[f]]
        present_fields += [f for f in _ACTIVE_SCALAR_FIELDS if scalar_values[f] is not None]

        hard_fields: list[str] = []
        soft_fields: list[str] = []
        if self.flags.explicit_constraint_orchestration:
            for f in present_fields:
                st = strengths.get(f, ConstraintStrength.SOFT)
                if st == ConstraintStrength.HARD:
                    hard_fields.append(f)
                else:
                    soft_fields.append(f)
            if any(exclusions.values()):
                hard_fields.append("exclusions")
        else:
            # no_context: everything is a soft relevance feature, no explicit hard.
            soft_fields = list(present_fields)

        # unknown = fields the search cares about but has no value for
        all_possible = set(_ACTIVE_LIST_FIELDS) | set(_ACTIVE_SCALAR_FIELDS)
        unknown_fields = sorted(all_possible - set(present_fields))

        if any(exclusions.values()):
            field_evidence.setdefault("exclusions", [])

        # Canonicalise + dedupe roles and skills so profile ("Python") and
        # dialogue ("python") do not produce duplicates.
        from ..taxonomy import canonical_role, canonical_skill

        list_values["target_roles"] = list(dict.fromkeys(
            canonical_role(r) for r in list_values["target_roles"]))
        list_values["skills_have"] = list(dict.fromkeys(
            canonical_skill(s) for s in list_values["skills_have"]))

        active_id = content_id(
            "as", inp.dialogue_state.session_id, cand.version,
            inp.dialogue_state.version, self.flags.variant,
        )
        return ActiveSearchState(
            active_search_id=active_id,
            session_id=inp.dialogue_state.session_id,
            candidate_id=cand.candidate_id,
            candidate_state_version=cand.version,
            dialogue_state_version=inp.dialogue_state.version,
            target_roles=list_values["target_roles"],
            skills_have=list_values["skills_have"],
            preferred_locations=list_values["preferred_locations"],
            salary_min=_as_float(scalar_values["salary_min"]),
            salary_currency=scalar_values["salary_currency"],
            work_modes=list_values["work_modes"],
            experience_level=scalar_values["experience_level"],
            years_experience=_as_float(scalar_values["years_experience"]),
            employment_types=list_values["employment_types"],
            work_authorizations=list_values["work_authorizations"],
            exclusions={k: sorted(set(v)) for k, v in exclusions.items() if v},
            hard_constraint_fields=sorted(set(hard_fields)),
            soft_preference_fields=sorted(set(soft_fields)),
            unknown_fields=unknown_fields,
            clarification_required_fields=sorted(set(clar_required)),
            field_evidence_map=field_evidence,
            generated_at=now,
        )


def _stronger(a: ConstraintStrength | None, b: ConstraintStrength) -> ConstraintStrength:
    order = {ConstraintStrength.UNKNOWN: 0, ConstraintStrength.NOT_APPLICABLE: 0,
             ConstraintStrength.SOFT: 1, ConstraintStrength.HARD: 2}
    if a is None:
        return b
    return a if order.get(a, 0) >= order.get(b, 0) else b


def _as_float(value) -> float | None:
    """Coerce a value to float, tolerating LLM variation.

    LLMs sometimes return a salary as an object like {"amount": 50000,
    "period": "month"} or a string like "RM50000", and hybrid-mode field
    validation replaces a stated salary with the canonical
    ``{min_salary, max_salary, currency, period}`` structure. Rather than
    re-implementing salary parsing here, we delegate to :func:`salary_amount` —
    the single salary parser/projection in the codebase — so a hard salary
    constraint is never silently dropped due to output shape.
    """
    return salary_amount(value)
