"""Job posting model and active-search view."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class JobPosting(BaseModel):
    """A normalised job posting from the curated catalog."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    title: str
    company: str
    description: str

    normalized_title: str
    role_family: str | None = None
    industry: str | None = None
    employment_type: str | None = None

    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    responsibilities: list[str] = Field(default_factory=list)

    salary_min: float | None = None
    salary_max: float | None = None
    salary_currency: str | None = None
    salary_period: Literal["hour", "month", "year", "unknown"] = "unknown"
    salary_min_monthly_myr: float | None = None
    salary_max_monthly_myr: float | None = None

    country: str | None = None
    city: str | None = None
    region: str | None = None
    work_mode: Literal["onsite", "hybrid", "remote", "unspecified"] = "unspecified"

    min_years_experience: float | None = None
    max_years_experience: float | None = None
    experience_level: str | None = None

    required_work_authorization: list[str] = Field(default_factory=list)
    application_deadline: date | None = None
    is_active: bool = True

    source_uri: str | None = None
    source_snapshot_id: str
    ingested_at: datetime
    raw_payload_hash: str


class ActiveSearchState(BaseModel):
    """The merged 'this search' view produced by the CMJCC.

    It combines the candidate's long-term profile with current dialogue
    evidence, and classifies fields into hard/soft/unknown/clarification while
    keeping a field -> evidence id map for traceability.
    """

    model_config = ConfigDict(extra="forbid")

    active_search_id: str
    session_id: str
    candidate_id: str
    candidate_state_version: int
    dialogue_state_version: int

    target_roles: list[str] = Field(default_factory=list)
    skills_have: list[str] = Field(default_factory=list)
    preferred_locations: list[str] = Field(default_factory=list)
    salary_min: float | None = None
    salary_currency: str | None = None
    work_modes: list[str] = Field(default_factory=list)
    experience_level: str | None = None
    years_experience: float | None = None
    employment_types: list[str] = Field(default_factory=list)
    work_authorizations: list[str] = Field(default_factory=list)

    exclusions: dict[str, list[str]] = Field(default_factory=dict)
    hard_constraint_fields: list[str] = Field(default_factory=list)
    soft_preference_fields: list[str] = Field(default_factory=list)
    unknown_fields: list[str] = Field(default_factory=list)
    clarification_required_fields: list[str] = Field(default_factory=list)
    field_evidence_map: dict[str, list[str]] = Field(default_factory=dict)
    generated_at: datetime
