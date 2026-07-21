"""Ranking features, ranked jobs, recommendation decisions and response claims."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from .constraints import EligibilityResult


class RankingFeature(BaseModel):
    """One auditable ranking feature contribution."""

    model_config = ConfigDict(extra="forbid")

    name: str
    raw_value: float | int | str | None = None
    normalized_score: float = Field(ge=0.0, le=1.0)
    weight: float = Field(ge=0.0)
    weighted_contribution: float
    evidence_ids: list[str] = Field(default_factory=list)
    explanation_code: str


class RankedJob(BaseModel):
    """A scored, eligible job with its per-feature breakdown."""

    model_config = ConfigDict(extra="forbid")

    job_id: str
    rank: int
    total_score: float
    features: list[RankingFeature] = Field(default_factory=list)
    eligibility_result_id: str
    skill_gaps: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


class RecommendationDecision(BaseModel):
    """The complete, traceable decision for one turn."""

    model_config = ConfigDict(extra="forbid")

    decision_id: str
    session_id: str
    active_search_id: str
    context_id: str | None
    experiment_variant: str
    retrieved_job_ids: list[str] = Field(default_factory=list)
    eligibility_results: list[EligibilityResult] = Field(default_factory=list)
    ranked_jobs: list[RankedJob] = Field(default_factory=list)
    selected_job_ids: list[str] = Field(default_factory=list)
    no_match: bool = False
    no_match_reason_codes: list[str] = Field(default_factory=list)
    created_at: datetime
    scorer_version: str
    config_hash: str


class ResponseClaim(BaseModel):
    """A factual claim in the final response, bound to evidence ids."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    claim_type: Literal[
        "candidate_preference",
        "job_attribute",
        "constraint_result",
        "ranking_reason",
        "skill_gap",
        "no_match_reason",
    ]
    text: str
    evidence_ids: list[str] = Field(default_factory=list)
    support_status: Literal["supported", "unsupported", "unknown"] = "supported"


class Response(BaseModel):
    """The final response object returned to the caller."""

    model_config = ConfigDict(extra="forbid")

    response_id: str
    session_id: str
    response_type: str
    message: str
    claims: list[ResponseClaim] = Field(default_factory=list)
    created_at: datetime
