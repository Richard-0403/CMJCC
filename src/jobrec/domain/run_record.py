"""RunRecord: the unified record threaded through every experiment run."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class RunRecord(BaseModel):
    """Reproducibility-focused record of a single end-to-end run."""

    model_config = ConfigDict(extra="forbid")

    run_id: str
    scenario_id: str | None = None
    session_id: str
    candidate_id: str
    experiment_variant: str
    feature_flags: dict[str, Any] = Field(default_factory=dict)
    workflow_states: list[str] = Field(default_factory=list)
    state_object_ids: dict[str, str] = Field(default_factory=dict)
    handoff_ids: list[str] = Field(default_factory=list)
    evidence_log_ids: list[str] = Field(default_factory=list)
    final_decision_id: str | None = None
    final_response_id: str | None = None
    started_at: datetime
    completed_at: datetime | None = None
    component_latency_ms: dict[str, float] = Field(default_factory=dict)
    total_latency_ms: float | None = None
    success: bool = False
    failure_code: str | None = None
    config_hash: str
    catalog_hash: str
    prompt_hash: str
    model_manifest: dict[str, Any] = Field(default_factory=dict)
    code_version: str
    db_version: str | None = None
    migration_version: int | None = None
