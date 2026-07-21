"""CandidateState: relatively stable / confirmed candidate information.

Temporary changes for a single search live in ``ActiveSearchState`` and must not
automatically overwrite the long-term values stored here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .evidence import PreferenceValue


class CandidateState(BaseModel):
    """Long-term, confirmed candidate profile and preferences."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: str
    version: int
    updated_at: datetime

    skills: list[PreferenceValue] = Field(default_factory=list)
    education_level: PreferenceValue | None = None
    years_experience: PreferenceValue | None = None
    experience_level: PreferenceValue | None = None
    target_roles: list[PreferenceValue] = Field(default_factory=list)
    preferred_locations: list[PreferenceValue] = Field(default_factory=list)
    salary_min: PreferenceValue | None = None
    salary_currency: PreferenceValue | None = None
    work_modes: list[PreferenceValue] = Field(default_factory=list)
    industries: list[PreferenceValue] = Field(default_factory=list)
    employment_types: list[PreferenceValue] = Field(default_factory=list)
    work_authorizations: list[PreferenceValue] = Field(default_factory=list)

    excluded_roles: list[PreferenceValue] = Field(default_factory=list)
    excluded_industries: list[PreferenceValue] = Field(default_factory=list)
    excluded_locations: list[PreferenceValue] = Field(default_factory=list)

    metadata: dict[str, Any] = Field(default_factory=dict)
