"""Enumerations shared across the domain model.

These enums encode the controlled vocabularies used throughout the system.
Schema validation rejects unknown enum members, so any new value must be added
here explicitly (see contract tests).
"""

from __future__ import annotations

from enum import StrEnum


class EvidenceSource(StrEnum):
    """Where a piece of evidence originated."""

    PROFILE = "profile"
    DIALOGUE = "dialogue"
    CLARIFICATION = "clarification"
    JOB_POSTING = "job_posting"
    SYSTEM_RULE = "system_rule"
    MODEL_INFERENCE = "model_inference"


class ConfirmationStatus(StrEnum):
    """How strongly a value has been confirmed."""

    CONFIRMED = "confirmed"
    UNCONFIRMED = "unconfirmed"
    INFERRED = "inferred"
    REJECTED = "rejected"


class PersistenceScope(StrEnum):
    """Lifetime / scope of a preference value."""

    LONG_TERM = "long_term"
    SESSION = "session"
    ACTIVE_SEARCH = "active_search"
    TURN_ONLY = "turn_only"


class ConstraintStrength(StrEnum):
    """Whether a condition filters (hard) or only scores (soft)."""

    HARD = "hard"
    SOFT = "soft"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class ConstraintOutcome(StrEnum):
    """Result of evaluating a single constraint against a job."""

    PASS = "pass"
    FAIL = "fail"
    UNKNOWN = "unknown"
    NOT_APPLICABLE = "not_applicable"


class WorkflowState(StrEnum):
    """States of the conversation orchestration state machine."""

    RECEIVED = "received"
    UNDERSTANDING = "understanding"
    VALIDATING = "validating"
    CLARIFICATION_REQUIRED = "clarification_required"
    MEMORY_UPDATED = "memory_updated"
    CONTEXT_BUILT = "context_built"
    RETRIEVED = "retrieved"
    FILTERED = "filtered"
    RANKED = "ranked"
    EXPLAINED = "explained"
    COMPLETED = "completed"
    NO_MATCH = "no_match"
    FAILED = "failed"


class ResponseType(StrEnum):
    """The kind of response produced for a turn."""

    RECOMMENDATION = "recommendation"
    CLARIFICATION = "clarification"
    NO_MATCH = "no_match"
    ERROR = "error"


class ExperimentVariant(StrEnum):
    """Supported experiment configurations / ablations."""

    FULL = "full"
    PROFILE_ONLY = "profile_only"
    ONE_SHOT = "one_shot"
    NO_MEMORY = "no_memory"
    NO_CONTEXT = "no_context"


class RunMode(StrEnum):
    """How LLM-dependent behaviour is executed."""

    DETERMINISTIC = "deterministic"
    HYBRID = "hybrid"
    REPLAY = "replay"


class UnknownPolicy(StrEnum):
    """What to do when a job field required by a hard constraint is missing."""

    FAIL = "fail"
    PASS = "pass"
    PENALIZE = "penalize"
    CLARIFY = "clarify"


class ErrorCode(StrEnum):
    """Explicit failure codes. The system never silently degrades."""

    INVALID_INPUT_SCHEMA = "INVALID_INPUT_SCHEMA"
    CATALOG_NOT_READY = "CATALOG_NOT_READY"
    EXTRACTION_FAILED = "EXTRACTION_FAILED"
    MODEL_TIMEOUT = "MODEL_TIMEOUT"
    MODEL_INVALID_JSON = "MODEL_INVALID_JSON"
    STATE_VERSION_CONFLICT = "STATE_VERSION_CONFLICT"
    EVIDENCE_NOT_FOUND = "EVIDENCE_NOT_FOUND"
    CONSTRAINT_EVALUATION_ERROR = "CONSTRAINT_EVALUATION_ERROR"
    RANKING_CONFIGURATION_ERROR = "RANKING_CONFIGURATION_ERROR"
    UNSUPPORTED_RESPONSE_CLAIM = "UNSUPPORTED_RESPONSE_CLAIM"
    NO_ELIGIBLE_JOB = "NO_ELIGIBLE_JOB"
    HANDOFF_VALIDATION_FAILED = "HANDOFF_VALIDATION_FAILED"
    INTERNAL_ERROR = "INTERNAL_ERROR"
