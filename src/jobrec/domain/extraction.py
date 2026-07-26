"""Extraction output models produced by the Candidate Understanding Agent."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .enums import ConfirmationStatus, ConstraintStrength, PersistenceScope


class ExtractedPreference(BaseModel):
    """One structured preference extracted from an utterance."""

    model_config = ConfigDict(extra="forbid")

    field_name: str
    normalized_value: Any
    raw_text: str
    span_start: int | None = None
    span_end: int | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    confirmation_status: ConfirmationStatus
    persistence_scope: PersistenceScope
    proposed_strength: ConstraintStrength
    polarity: Literal["positive", "negative"] = "positive"
    temporal_scope: Literal["current_search", "session", "long_term", "unknown"] = "current_search"
    # Free-form provenance bag. Notably carries ``extraction_method`` ("rule"|"llm")
    # recorded by the orchestrator so downstream metrics can attribute each field to
    # the deterministic rule extractor or the model (R8.8/8.12).
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedPreferenceSet(BaseModel):
    """The full set of preferences extracted from a single utterance."""

    model_config = ConfigDict(extra="forbid")

    utterance_id: str
    preferences: list[ExtractedPreference] = Field(default_factory=list)
    detected_language: str = "en"
    ambiguous_fields: list[str] = Field(default_factory=list)
    extraction_warnings: list[str] = Field(default_factory=list)
