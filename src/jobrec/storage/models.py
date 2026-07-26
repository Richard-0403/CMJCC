"""SQLAlchemy ORM models.

Design notes (landing-plan section 18):
- State versions are immutable rows (composite PK candidate_id/version, etc.).
- Foreign keys point at specific *versions*, not just the candidate id.
- Large structured objects are stored as JSON; frequently-queried metric fields
  are columnised for querying.
- ``run_id`` threads through decisions, handoffs, evidence logs and run records.
- No secrets are ever stored.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column

from .db import Base


class SchemaVersion(Base):
    """Records the current schema/migration version.

    A minimal single-row table (pinned via ``id == 1``) capturing the applied
    migration version, when it was applied, and a short description. Managed by
    ``storage/migrations.ensure_schema_version`` rather than a full migration
    framework (see design R9.7).
    """

    __tablename__ = "schema_version"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, default=1)
    version: Mapped[int] = mapped_column(Integer, default=0)
    applied_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    description: Mapped[str | None] = mapped_column(String(256), nullable=True)


class Candidate(Base):
    __tablename__ = "candidates"
    candidate_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    latest_version: Mapped[int] = mapped_column(Integer, default=1)


class CandidateStateVersion(Base):
    __tablename__ = "candidate_state_versions"
    candidate_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    payload: Mapped[dict] = mapped_column(JSON)


class Session(Base):
    __tablename__ = "sessions"
    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    candidate_id: Mapped[str] = mapped_column(String(64), ForeignKey("candidates.candidate_id"))
    experiment_variant: Mapped[str] = mapped_column(String(32))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class DialogueStateVersion(Base):
    __tablename__ = "dialogue_state_versions"
    session_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, primary_key=True)
    candidate_id: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSON)


class EvidenceItemRow(Base):
    __tablename__ = "evidence_items"
    evidence_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    source: Mapped[str] = mapped_column(String(32))
    field_name: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSON)


class RecommendationDecisionRow(Base):
    __tablename__ = "recommendation_decisions"
    decision_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    experiment_variant: Mapped[str] = mapped_column(String(32), index=True)
    no_match: Mapped[bool] = mapped_column(Boolean, default=False)
    selected_count: Mapped[int] = mapped_column(Integer, default=0)
    payload: Mapped[dict] = mapped_column(JSON)


class AgentHandoffRow(Base):
    __tablename__ = "agent_handoffs"
    handoff_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    from_component: Mapped[str] = mapped_column(String(64))
    to_component: Mapped[str] = mapped_column(String(64))
    validation_passed: Mapped[bool] = mapped_column(Boolean)
    status: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict] = mapped_column(JSON)


class EvidenceLogRow(Base):
    __tablename__ = "evidence_logs"
    log_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    stage: Mapped[str] = mapped_column(String(64))
    event_type: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32))
    payload: Mapped[dict] = mapped_column(JSON)


class ResponseClaimRow(Base):
    __tablename__ = "response_claims"
    claim_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    claim_type: Mapped[str] = mapped_column(String(32))
    support_status: Mapped[str] = mapped_column(String(16))
    payload: Mapped[dict] = mapped_column(JSON)


class ResponseRow(Base):
    __tablename__ = "responses"
    response_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    response_type: Mapped[str] = mapped_column(String(32))
    message: Mapped[str] = mapped_column(Text)
    payload: Mapped[dict] = mapped_column(JSON)


class RunRecordRow(Base):
    __tablename__ = "run_records"
    run_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    scenario_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    session_id: Mapped[str] = mapped_column(String(64), index=True)
    candidate_id: Mapped[str] = mapped_column(String(64), index=True)
    experiment_variant: Mapped[str] = mapped_column(String(32), index=True)
    success: Mapped[bool] = mapped_column(Boolean, index=True)
    failure_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    total_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    config_hash: Mapped[str] = mapped_column(String(64), index=True)
    catalog_hash: Mapped[str] = mapped_column(String(64))
    prompt_hash: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSON)


class ModelCallRow(Base):
    __tablename__ = "model_call_records"
    call_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    run_id: Mapped[str] = mapped_column(String(128), index=True)
    purpose: Mapped[str] = mapped_column(String(64))
    provider: Mapped[str] = mapped_column(String(32))
    model: Mapped[str] = mapped_column(String(64))
    payload: Mapped[dict] = mapped_column(JSON)


class CatalogSnapshotRow(Base):
    __tablename__ = "catalog_snapshots"
    catalog_snapshot_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    catalog_hash: Mapped[str] = mapped_column(String(64))
    record_count: Mapped[int] = mapped_column(Integer)
    payload: Mapped[dict] = mapped_column(JSON)


class JobPostingRow(Base):
    __tablename__ = "job_postings"
    job_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    catalog_snapshot_id: Mapped[str] = mapped_column(String(64), index=True)
    title: Mapped[str] = mapped_column(String(256))
    role_family: Mapped[str | None] = mapped_column(String(64), nullable=True)
    payload: Mapped[dict] = mapped_column(JSON)
