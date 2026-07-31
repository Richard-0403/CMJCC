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


def _stated_values(field_name: str, raw: object) -> list:
    """The values a model's answer states for ``field_name``, one per preference.

    A list on a SCALAR-arity field is fanned out; a list on a list-arity field is left
    whole, because that field's normalizer is the thing meant to expand it. Anything else is
    a single value. An empty container yields one ``None``, so "the model named the field
    without a value" still reaches the ordinary absent-value handling instead of vanishing.
    """
    from .field_validation import field_arity

    if not isinstance(raw, list | tuple | set | frozenset):
        return [raw]
    values = list(raw)
    if not values:
        return [None]
    if field_arity(field_name) == "list":
        return [values]
    # A ONE-item list is left alone. The recovery ladder already unwraps it correctly, marks
    # the value ``repaired`` and keeps it attributed to the model, and that rung is covered by
    # its own tests -- fanning it out here would make the repair path unreachable and remove
    # the coverage without fixing anything. Only 2+ items are fanned out, which is exactly
    # the shape repair cannot resolve and which was therefore being discarded.
    if len(values) == 1:
        return [values]
    return values


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
        field_name = str(rp["field_name"])
        # One value per preference, which is the representation the rest of the pipeline is
        # built on: a scalar-arity field holds ONE stated value and several values become
        # several preferences (the rule extractor already fans out this way).
        #
        # Models do not honour that. They answer "onsite or hybrid" as a single preference
        # whose value is a two-item list, and field validation then rejected the shape: a
        # one-item list could be unwrapped by schema repair, two items could not, so the
        # whole field fell through to the rule extractor and was recorded UNCONFIRMED. The
        # model had extracted the constraint correctly and the value was discarded anyway --
        # measured on a hybrid smoke, where SC-D-08's ``work_modes`` came back as
        # ``rule_fallback`` while the model's answer named both modes.
        #
        # Fanning out here keeps the model's extraction and produces exactly the shape the
        # contract asks for, instead of widening the contract to admit lists.
        for value in _stated_values(field_name, rp["normalized_value"]):
            prefs.append(ExtractedPreference(
                field_name=field_name,
                normalized_value=value,
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
