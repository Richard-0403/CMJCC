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
    #: What the utterance does to the field, as an explicit operation rather than something
    #: inferred from the merge order.
    #:
    #: ``add`` states a value (the default, and what every extraction did implicitly
    #: before). ``relax`` WITHDRAWS a requirement: the candidate says it is no longer
    #: binding, and it is the only operation permitted to turn HARD into SOFT. That
    #: distinction is the whole reason this field exists -- strength merging is monotone by
    #: design, because two statements inside one turn should combine to the stronger of the
    #: two, and without an explicit relax signal that same rule makes a hard constraint
    #: permanent for the rest of the session no matter what the candidate says next.
    #:
    #: A ``relax`` may carry ``normalized_value=None``: "I am flexible on work mode" names
    #: the field without restating a value.
    operation: Literal["add", "replace", "remove", "relax", "confirm"] = "add"
    #: The turn that actually said this. ``None`` means the current turn.
    #:
    #: Prior-turn preferences are re-derived by re-parsing history, and everything built
    #: from them used to be stamped with the CURRENT turn id -- so a value the candidate
    #: stated three turns ago produced evidence attributed to now. Carrying the origin lets
    #: per-turn provenance stay truthful while the re-parsing itself is still in place.
    origin_turn_id: str | None = None
    #: The EvidenceItem this preference produced, set once the turn that stated it
    #: registered its evidence.
    #:
    #: Carried on the preference so a prior turn's contribution to a later search state
    #: can cite the evidence the ORIGINAL turn created. Without it the only way to get an
    #: evidence id for a historical preference was to re-register it under the current
    #: turn, which minted a second item for one statement and attributed it to a turn that
    #: never said it.
    evidence_id: str | None = None
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
