"""Structured-output parsing and validation for extraction.

Model JSON is validated against the Pydantic ExtractedPreferenceSet schema.
Parsing failures raise LLMInvalidJSON so the orchestrator can retry a limited
number of times and then fall back to the rule extractor (never fabricate).
"""

from __future__ import annotations

import json

from ..domain.extraction import ExtractedPreferenceSet
from .provider import LLMInvalidJSON


def parse_extraction(payload: dict | str) -> ExtractedPreferenceSet:
    """Validate a JSON payload into an ExtractedPreferenceSet."""
    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise LLMInvalidJSON(f"could not decode model JSON: {exc}") from exc
    try:
        return ExtractedPreferenceSet.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - surfaced as explicit error
        raise LLMInvalidJSON(f"extraction failed schema validation: {exc}") from exc


def parse_extraction_lenient(payload: dict | str, utterance: str = "") -> ExtractedPreferenceSet:
    """Parse a minimal model JSON into an ExtractedPreferenceSet, filling the
    bookkeeping fields (confirmation/persistence/scope/confidence) with defaults.

    LLMs reliably produce field_name / normalized_value / raw_text /
    proposed_strength / polarity, but not the internal provenance fields. We fill
    those deterministically so a well-formed model response is actually used
    instead of being rejected and falling back to rules.
    """
    from ..domain.enums import ConfirmationStatus, ConstraintStrength, PersistenceScope
    from ..domain.extraction import ExtractedPreference
    from ..utils.hashing import content_id

    if isinstance(payload, str):
        try:
            payload = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise LLMInvalidJSON(f"could not decode model JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise LLMInvalidJSON("model JSON is not an object")

    raw_prefs = payload.get("preferences", [])
    if not isinstance(raw_prefs, list):
        raise LLMInvalidJSON("'preferences' is not a list")

    valid_strength = {s.value for s in ConstraintStrength}
    prefs: list[ExtractedPreference] = []
    for rp in raw_prefs:
        if not isinstance(rp, dict) or "field_name" not in rp or "normalized_value" not in rp:
            continue
        strength = str(rp.get("proposed_strength", "soft")).lower()
        if strength not in valid_strength:
            strength = "soft"
        polarity = "negative" if str(rp.get("polarity", "positive")).lower() == "negative" else "positive"
        temporal = str(rp.get("temporal_scope", "current_search")).lower()
        if temporal not in {"current_search", "session", "long_term", "unknown"}:
            temporal = "current_search"
        confirmation = (ConfirmationStatus.UNCONFIRMED if strength == "unknown"
                        else ConfirmationStatus.CONFIRMED)
        try:
            confidence = float(rp.get("confidence", 0.85))
        except (TypeError, ValueError):
            confidence = 0.85
        confidence = min(max(confidence, 0.0), 1.0)
        prefs.append(ExtractedPreference(
            field_name=str(rp["field_name"]),
            normalized_value=rp["normalized_value"],
            raw_text=str(rp.get("raw_text", "")),
            confidence=confidence,
            confirmation_status=confirmation,
            persistence_scope=PersistenceScope.ACTIVE_SEARCH,
            proposed_strength=ConstraintStrength(strength),
            polarity=polarity,
            temporal_scope=temporal,
        ))
    return ExtractedPreferenceSet(
        utterance_id=payload.get("utterance_id") or content_id("utt", utterance),
        preferences=prefs,
        detected_language=payload.get("detected_language", "en"),
        ambiguous_fields=payload.get("ambiguous_fields", []) if isinstance(payload.get("ambiguous_fields"), list) else [],
        extraction_warnings=payload.get("extraction_warnings", []) if isinstance(payload.get("extraction_warnings"), list) else [],
    )
