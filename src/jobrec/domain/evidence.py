"""Evidence and preference-value models.

Every field that enters the active search view must be backed by at least one
``EvidenceItem``. Raw text and the normalised value are both retained so that
explanations can be traced back to their source.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .enums import ConfirmationStatus, EvidenceSource, PersistenceScope


class EvidenceItem(BaseModel):
    """A single, traceable piece of evidence about a candidate or job field."""

    model_config = ConfigDict(extra="forbid")

    evidence_id: str
    source: EvidenceSource
    source_object_id: str
    field_name: str
    raw_text: str | None = None
    normalized_value: Any = None
    confidence: float = Field(ge=0.0, le=1.0)
    confirmation_status: ConfirmationStatus
    persistence_scope: PersistenceScope
    observed_at: datetime
    turn_id: str | None = None
    text_span_start: int | None = None
    text_span_end: int | None = None
    extractor_name: str | None = None
    extractor_version: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PreferenceValue(BaseModel):
    """A candidate preference value with its supporting evidence and scope."""

    model_config = ConfigDict(extra="forbid")

    value: Any
    evidence_ids: list[str] = Field(default_factory=list)
    confirmation_status: ConfirmationStatus
    persistence_scope: PersistenceScope
    effective_from: datetime
    effective_to: datetime | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    is_active: bool = True
