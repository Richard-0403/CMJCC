"""Retrieval contracts shared by all retriever implementations.

Retrieval only *recalls* a candidate pool (larger than top-k). It never makes
final eligibility decisions; hard-constraint filtering happens afterwards in the
Job Context Agent.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..domain.job import ActiveSearchState, JobPosting


@dataclass(frozen=True)
class QuerySpec:
    """A structured query derived from the active-search view."""

    roles: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    responsibilities: list[str] = field(default_factory=list)
    industries: list[str] = field(default_factory=list)
    excluded_terms: list[str] = field(default_factory=list)

    @classmethod
    def from_active_search(cls, active: ActiveSearchState) -> QuerySpec:
        excluded = []
        for vals in active.exclusions.values():
            excluded.extend(vals)
        return cls(
            roles=list(active.target_roles),
            skills=list(active.skills_have),
            responsibilities=[],
            industries=[],
            excluded_terms=excluded,
        )

    def positive_text(self) -> str:
        """The positive query text (excluded terms are NOT concatenated here)."""
        parts = self.roles + self.skills + self.responsibilities + self.industries
        return " ".join(parts)


@dataclass(frozen=True)
class RetrievedJob:
    """One recalled job with its (normalised) retrieval score and components."""

    job_id: str
    score: float
    components: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalOutcome:
    """Result of a retrieval call, including recall diagnostics."""

    retrieved: list[RetrievedJob]
    initial_pool_size: int
    expanded: bool = False
    expansion_reason: str | None = None


class Retriever(Protocol):
    """Protocol implemented by structured, tfidf and hybrid retrievers."""

    def retrieve(
        self, query: QuerySpec, jobs: list[JobPosting], pool_size: int
    ) -> RetrievalOutcome:
        ...
