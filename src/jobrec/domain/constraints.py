"""Constraint definitions, job-context state and eligibility results."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from .enums import ConstraintOutcome, ConstraintStrength, UnknownPolicy


class ConstraintDefinition(BaseModel):
    """A single constraint mapped from a candidate condition."""

    model_config = ConfigDict(extra="forbid")

    constraint_id: str
    field_name: str
    operator: Literal[
        "eq", "neq", "in", "not_in", "gte", "lte",
        "overlap", "contains_all", "contains_any", "not_expired",
    ]
    expected_value: Any
    strength: ConstraintStrength
    weight: float = 0.0
    evidence_ids: list[str] = Field(default_factory=list)
    unknown_policy: UnknownPolicy
    rule_id: str


class JobContextState(BaseModel):
    """The constraint bundle produced for an active search."""

    model_config = ConfigDict(extra="forbid")

    context_id: str
    active_search_id: str
    catalog_snapshot_id: str
    constraints: list[ConstraintDefinition] = Field(default_factory=list)
    normalized_at: datetime
    normalization_warnings: list[str] = Field(default_factory=list)


class ConstraintCheck(BaseModel):
    """Outcome of evaluating one constraint against one job."""

    model_config = ConfigDict(extra="forbid")

    constraint_id: str
    field_name: str
    outcome: ConstraintOutcome
    observed_job_value: Any = None
    expected_candidate_value: Any = None
    explanation_code: str
    evidence_ids: list[str] = Field(default_factory=list)


class EligibilityResult(BaseModel):
    """Per-job eligibility decision with a full audit trail of checks."""

    model_config = ConfigDict(extra="forbid")

    eligibility_result_id: str
    job_id: str
    eligible: bool
    checks: list[ConstraintCheck] = Field(default_factory=list)
    hard_violation_count: int = 0
    unknown_hard_constraint_count: int = 0
    filtered_reason_codes: list[str] = Field(default_factory=list)
