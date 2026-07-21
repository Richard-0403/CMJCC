"""Agent handoff and evidence-log records for inspectability."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class AgentHandoff(BaseModel):
    """Records one contract-bound handoff between components."""

    model_config = ConfigDict(extra="forbid")

    handoff_id: str
    run_id: str
    from_component: str
    to_component: str
    contract_name: str
    input_schema_version: str
    output_schema_version: str | None = None
    attempted_at: datetime
    completed_at: datetime | None = None
    validation_passed: bool = False
    status: Literal["attempted", "completed", "failed", "recovered"]
    error_code: str | None = None


class EvidenceLogEntry(BaseModel):
    """A single decision-log event describing what a component did."""

    model_config = ConfigDict(extra="forbid")

    log_id: str
    run_id: str
    stage: str
    event_type: str
    actor: str
    source_ids: list[str] = Field(default_factory=list)
    input_object_ids: list[str] = Field(default_factory=list)
    output_object_ids: list[str] = Field(default_factory=list)
    rule_id: str | None = None
    previous_value: Any = None
    new_value: Any = None
    status: Literal["success", "failure", "skipped", "recovered"]
    error_code: str | None = None
    created_at: datetime
