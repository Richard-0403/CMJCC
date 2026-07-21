"""LLM provider protocol and shared types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol


class LLMError(Exception):
    """Base class for LLM adapter failures (maps to explicit error codes)."""


class LLMTimeout(LLMError):
    pass


class LLMInvalidJSON(LLMError):
    pass


@dataclass
class LLMCallRecord:
    """A recorded model call (for logging and replay)."""

    call_id: str
    purpose: str
    prompt: str
    raw_response: str
    parsed_ok: bool
    latency_ms: float
    provider: str
    model: str
    metadata: dict[str, Any] = field(default_factory=dict)


class LLMProvider(Protocol):
    """Minimal provider interface used by the orchestration layer."""

    name: str
    model: str

    def complete_json(self, prompt: str, *, purpose: str) -> tuple[dict, LLMCallRecord]:
        """Return a parsed JSON object plus a call record. Raises LLMError."""
        ...

    def complete_text(self, prompt: str, *, purpose: str, fallback: str = "") -> tuple[str, LLMCallRecord]:
        """Return generated text plus a call record. Never raises for phrasing."""
        ...

    def manifest(self) -> dict[str, Any]:
        """Return a manifest describing the provider/model for reproducibility."""
        ...
