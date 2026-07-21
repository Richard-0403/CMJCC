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
