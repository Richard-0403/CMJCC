"""In-memory evidence registry.

Every ``EvidenceItem`` produced anywhere in the pipeline is registered here so
that downstream components (constraint checks, ranking features, response
claims) can reference evidence by id, and the claim validator can verify that an
id resolves to a real piece of evidence. It can be mirrored to the database.
"""

from __future__ import annotations

from .domain.enums import ConfirmationStatus, EvidenceSource, PersistenceScope
from .domain.evidence import EvidenceItem
from .utils.hashing import content_id
from .utils.time import utcnow


class EvidenceStore:
    """A simple id -> EvidenceItem registry scoped to a session/run."""

    def __init__(self) -> None:
        self._items: dict[str, EvidenceItem] = {}

    def add(self, item: EvidenceItem) -> EvidenceItem:
        self._items[item.evidence_id] = item
        return item

    def add_many(self, items: list[EvidenceItem]) -> list[EvidenceItem]:
        for item in items:
            self.add(item)
        return items

    def get(self, evidence_id: str) -> EvidenceItem | None:
        return self._items.get(evidence_id)

    def exists(self, evidence_id: str) -> bool:
        return evidence_id in self._items

    def all(self) -> list[EvidenceItem]:
        return list(self._items.values())

    # ---- factory helpers -------------------------------------------------
    def register_field(
        self,
        source: EvidenceSource,
        source_object_id: str,
        field_name: str,
        normalized_value: object,
        *,
        confidence: float,
        confirmation: ConfirmationStatus,
        scope: PersistenceScope,
        raw_text: str | None = None,
        turn_id: str | None = None,
        span: tuple[int, int] | None = None,
        extractor_name: str | None = None,
        extractor_version: str | None = None,
    ) -> EvidenceItem:
        """Create, register and return an EvidenceItem with a content id."""
        evidence_id = content_id(
            "ev", source, source_object_id, field_name, str(normalized_value), turn_id or ""
        )
        item = EvidenceItem(
            evidence_id=evidence_id,
            source=source,
            source_object_id=source_object_id,
            field_name=field_name,
            raw_text=raw_text,
            normalized_value=normalized_value,
            confidence=confidence,
            confirmation_status=confirmation,
            persistence_scope=scope,
            observed_at=utcnow(),
            turn_id=turn_id,
            text_span_start=span[0] if span else None,
            text_span_end=span[1] if span else None,
            extractor_name=extractor_name,
            extractor_version=extractor_version,
        )
        return self.add(item)
