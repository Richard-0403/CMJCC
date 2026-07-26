"""Fault-injection helpers for the R10 failure-path tests.

This support module provides small, deterministic building blocks used by the
failure-path unit and integration tests (Requirement 10). None of these helpers
are tests themselves; they only fabricate faulty inputs so that the real
validation and recovery paths can be exercised:

* :class:`FaultInjectingProvider` -- an :class:`~jobrec.llm.provider.LLMProvider`
  that raises :class:`~jobrec.llm.provider.LLMTimeout` for the first
  ``fail_times`` calls and then succeeds by delegating to a wrapped provider.
  Used for the timeout-with-retry and partial-failure-with-recovery cases
  (R10.4), typically together with :func:`jobrec.llm.retry.retry_call`.
* claim factories that fabricate
  :class:`~jobrec.domain.recommendation.ResponseClaim` objects with dangling or
  missing evidence ids so the claim validator's drop/flag path is exercised
  (R10.1, R10.2, R10.6).
* handoff-payload factories that omit required fields or violate the schema so
  :class:`~jobrec.domain.handoff.AgentHandoff` validation fails (R10.3, R10.7).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from jobrec.domain.handoff import AgentHandoff
from jobrec.domain.recommendation import ResponseClaim
from jobrec.llm.mock_provider import MockLLMProvider
from jobrec.llm.provider import LLMCallRecord, LLMError, LLMProvider, LLMTimeout
from jobrec.utils.hashing import content_id
from jobrec.utils.time import utcnow

# An evidence id that is never registered in any EvidenceStore. Claims that
# reference it must be dropped/flagged as unsupported by ``validate_claims``.
DANGLING_EVIDENCE_ID = "ev-dangling-0000000000000000"

# Required (no-default) fields of ``AgentHandoff``, derived from the model so
# the factories below stay in sync if the schema changes.
REQUIRED_HANDOFF_FIELDS: tuple[str, ...] = tuple(
    name for name, info in AgentHandoff.model_fields.items() if info.is_required()
)


# --------------------------------------------------------------------------- provider
class FaultInjectingProvider:
    """LLM provider that fails a fixed number of times, then succeeds.

    The first ``fail_times`` calls to :meth:`complete_json` raise ``error``
    (:class:`~jobrec.llm.provider.LLMTimeout` by default); every subsequent call
    delegates to ``delegate`` (a deterministic :class:`MockLLMProvider` by
    default) so the returned shape stays valid. This lets tests drive the
    bounded-retry path (:func:`jobrec.llm.retry.retry_call`) and the
    partial-failure-with-recovery path (R10.4).

    ``complete_text`` never injects a fault: per the provider protocol, text
    phrasing must not raise. It delegates straight through.
    """

    def __init__(
        self,
        fail_times: int,
        *,
        delegate: LLMProvider | None = None,
        error: type[LLMError] = LLMTimeout,
    ) -> None:
        if fail_times < 0:
            raise ValueError("fail_times must be >= 0")
        self.fail_times = fail_times
        self._delegate: LLMProvider = delegate or MockLLMProvider()
        self._error = error
        self.name = f"fault-injecting:{self._delegate.name}"
        self.model = self._delegate.model
        # Observability counters for assertions in tests.
        self.attempts = 0
        self.failures = 0

    def reset(self) -> None:
        """Reset the failure/attempt counters so the provider can be reused."""
        self.attempts = 0
        self.failures = 0

    def _maybe_fail(self, purpose: str) -> None:
        self.attempts += 1
        if self.failures < self.fail_times:
            self.failures += 1
            raise self._error(
                f"injected {self._error.__name__} "
                f"({self.failures}/{self.fail_times}) for purpose={purpose!r}"
            )

    def complete_json(self, prompt: str, *, purpose: str) -> tuple[dict, LLMCallRecord]:
        self._maybe_fail(purpose)
        return self._delegate.complete_json(prompt, purpose=purpose)

    def complete_text(
        self, prompt: str, *, purpose: str, fallback: str = ""
    ) -> tuple[str, LLMCallRecord]:
        return self._delegate.complete_text(prompt, purpose=purpose, fallback=fallback)

    def manifest(self) -> dict[str, Any]:
        manifest = dict(self._delegate.manifest())
        manifest.update(
            {
                "provider": self.name,
                "fault_injection": {
                    "fail_times": self.fail_times,
                    "error": self._error.__name__,
                },
            }
        )
        return manifest


# ----------------------------------------------------------------------------- claims
def make_claim(
    *,
    claim_type: str = "ranking_reason",
    text: str = "This role matches your stated preference.",
    evidence_ids: Iterable[str] | None = None,
) -> ResponseClaim:
    """Build a :class:`ResponseClaim` with an arbitrary evidence-id set.

    The default ``evidence_ids`` is empty, which is itself an unsupported claim
    (no source). Pass explicit ids for supported or dangling variants.
    """
    ids = list(evidence_ids) if evidence_ids is not None else []
    return ResponseClaim(
        claim_id=content_id("claim", "fault", claim_type, text, *ids),
        claim_type=claim_type,  # type: ignore[arg-type]
        text=text,
        evidence_ids=ids,
    )


def make_dangling_claim(
    *,
    claim_type: str = "ranking_reason",
    text: str = "This role matches evidence that was never registered.",
    evidence_id: str = DANGLING_EVIDENCE_ID,
) -> ResponseClaim:
    """A claim whose evidence id does not resolve in any ``EvidenceStore`` (R10.1)."""
    return make_claim(claim_type=claim_type, text=text, evidence_ids=[evidence_id])


def make_unsupported_claim(
    *,
    claim_type: str = "job_attribute",
    text: str = "This posting offers a benefit that was never stated.",
) -> ResponseClaim:
    """A claim with no evidence ids at all -- a missing source (R10.2, R10.6)."""
    return make_claim(claim_type=claim_type, text=text, evidence_ids=[])


# --------------------------------------------------------------------------- handoffs
def valid_handoff_payload(**overrides: Any) -> dict[str, Any]:
    """A complete, schema-valid ``AgentHandoff`` payload used as a mutation base.

    Callers mutate the returned dict (or use the helpers below) to produce the
    failure variants required by R10.3.
    """
    payload: dict[str, Any] = {
        "handoff_id": "handoff-fault-1",
        "run_id": "run-fault-1",
        "from_component": "orchestrator",
        "to_component": "explanation_agent",
        "contract_name": "recommendation_decision",
        "input_schema_version": "1.0",
        "output_schema_version": "1.0",
        "attempted_at": utcnow(),
        "completed_at": None,
        "validation_passed": False,
        "status": "attempted",
        "error_code": None,
    }
    payload.update(overrides)
    return payload


def missing_field_handoff_payload(
    omit: str | Iterable[str] = "contract_name", **overrides: Any
) -> dict[str, Any]:
    """A handoff payload with one or more *required* fields removed (R10.3).

    Constructing ``AgentHandoff(**payload)`` from the result raises a pydantic
    ``ValidationError`` because a required field is absent.
    """
    payload = valid_handoff_payload(**overrides)
    fields = [omit] if isinstance(omit, str) else list(omit)
    for name in fields:
        payload.pop(name, None)
    return payload


def schema_invalid_handoff_payload(**overrides: Any) -> dict[str, Any]:
    """A handoff payload that violates the schema via an invalid ``status`` (R10.3).

    All required fields are present (distinct from the missing-field case), but
    ``status`` is not one of the permitted literal values, so
    ``AgentHandoff(**payload)`` fails validation.
    """
    payload = valid_handoff_payload(**overrides)
    payload["status"] = "not-a-valid-status"
    return payload
