"""DialogueState, turns, and preference conflicts."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DialogueTurn(BaseModel):
    """A single utterance from the candidate or the system."""

    model_config = ConfigDict(extra="forbid")

    turn_id: str
    session_id: str
    turn_index: int
    speaker: Literal["candidate", "system"]
    text: str
    created_at: datetime
    evidence_ids: list[str] = Field(default_factory=list)
    action_type: str | None = None


class PreferenceConflict(BaseModel):
    """A detected conflict between existing and incoming evidence."""

    model_config = ConfigDict(extra="forbid")

    conflict_id: str
    field_name: str
    existing_evidence_ids: list[str] = Field(default_factory=list)
    incoming_evidence_ids: list[str] = Field(default_factory=list)
    conflict_type: Literal[
        "value_mismatch",
        "hard_soft_mismatch",
        "scope_mismatch",
        "negation_conflict",
        "temporal_override",
    ]
    impact: Literal["low", "medium", "high"]
    resolution: Literal[
        "use_current_for_search",
        "keep_profile",
        "ask_clarification",
        "merge_values",
        "reject_incoming",
        "unresolved",
    ]
    resolution_rule_id: str
    created_at: datetime


class ClarificationAction(BaseModel):
    """A deterministic request to ask the candidate one high-value question."""

    model_config = ConfigDict(extra="forbid")

    clarification_id: str
    target_fields: list[str] = Field(default_factory=list)
    reason_code: str
    priority_score: float = 0.0
    question_text: str
    options: list[str] = Field(default_factory=list)
    related_conflict_ids: list[str] = Field(default_factory=list)
    created_at: datetime


class DialogueState(BaseModel):
    """Session-scoped dialogue history and unresolved items."""

    model_config = ConfigDict(extra="forbid")

    session_id: str
    candidate_id: str
    version: int
    turns: list[DialogueTurn] = Field(default_factory=list)
    unresolved_slots: list[str] = Field(default_factory=list)
    conflicts: list[PreferenceConflict] = Field(default_factory=list)
    active_search_id: str | None = None
