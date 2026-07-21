"""API request/response schemas."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class CreateCandidateRequest(BaseModel):
    candidate_id: str
    skills: list[str] = Field(default_factory=list)
    years_experience: float | None = None
    experience_level: str | None = None
    target_roles: list[str] = Field(default_factory=list)
    preferred_locations: list[str] = Field(default_factory=list)
    salary_min: float | None = None
    salary_currency: str | None = None
    work_modes: list[str] = Field(default_factory=list)
    education_level: str | None = None
    industries: list[str] = Field(default_factory=list)
    employment_types: list[str] = Field(default_factory=list)
    work_authorizations: list[str] = Field(default_factory=list)


class CreateSessionRequest(BaseModel):
    candidate_id: str
    experiment_variant: str = "full"


class TurnRequest(BaseModel):
    text: str
    request_id: str | None = None


class TurnResponse(BaseModel):
    run_id: str
    response_type: str
    message: str
    claims: list[dict[str, Any]] = Field(default_factory=list)
    recommendations: list[dict[str, Any]] = Field(default_factory=list)
    state_versions: dict[str, int] = Field(default_factory=dict)
    trace_summary: dict[str, Any] = Field(default_factory=dict)
