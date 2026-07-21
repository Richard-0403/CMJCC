"""Deterministic mock LLM provider.

Used in ``deterministic`` mode and in tests. For structured extraction it
delegates to the rule-based extractor so that ``hybrid`` runs with the mock
provider are fully reproducible. For text phrasing it returns the supplied
fallback verbatim (no stochastic generation).
"""

from __future__ import annotations

from typing import Any

from ..agents.candidate_understanding import CandidateUnderstandingAgent
from ..utils.hashing import content_id
from .provider import LLMCallRecord


class MockLLMProvider:
    """A deterministic, offline provider."""

    def __init__(self) -> None:
        self.name = "mock"
        self.model = "mock-deterministic-v1"
        self._extractor = CandidateUnderstandingAgent()

    def complete_json(self, prompt: str, *, purpose: str) -> tuple[dict, LLMCallRecord]:
        # The prompt embeds the utterance after the marker "Utterance:".
        utterance = prompt.split("Utterance:", 1)[-1].strip() if "Utterance:" in prompt else prompt
        result = self._extractor.extract(utterance)
        payload = result.model_dump(mode="json")
        record = LLMCallRecord(
            call_id=content_id("call", purpose, utterance),
            purpose=purpose, prompt=prompt, raw_response="<mock-json>",
            parsed_ok=True, latency_ms=0.0, provider=self.name, model=self.model,
        )
        return payload, record

    def complete_text(self, prompt: str, *, purpose: str, fallback: str = "") -> tuple[str, LLMCallRecord]:
        record = LLMCallRecord(
            call_id=content_id("call", purpose, fallback or prompt),
            purpose=purpose, prompt=prompt, raw_response=fallback,
            parsed_ok=True, latency_ms=0.0, provider=self.name, model=self.model,
        )
        return fallback, record

    def manifest(self) -> dict[str, Any]:
        return {"provider": self.name, "model": self.model, "mode": "deterministic"}
