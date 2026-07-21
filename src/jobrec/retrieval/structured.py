"""Structured role-family retrieval / prefilter.

Scores a job by how well its role family and skills match the query's structured
fields. Used both as a standalone retriever and as one component of the hybrid
retriever.
"""

from __future__ import annotations

from ..domain.job import JobPosting
from ..taxonomy import canonical_role, canonical_skill
from .base import QuerySpec, RetrievalOutcome, RetrievedJob


def structured_score(query: QuerySpec, job: JobPosting) -> float:
    """Return a [0,1] structured similarity between query and job."""
    role_score = 0.0
    if query.roles:
        wanted = {canonical_role(r) for r in query.roles}
        job_role = job.role_family or canonical_role(job.title)
        role_score = 1.0 if job_role in wanted else 0.0

    skill_score = 0.0
    if query.skills:
        wanted_skills = {canonical_skill(s) for s in query.skills}
        job_skills = {canonical_skill(s) for s in (job.required_skills + job.preferred_skills)}
        if wanted_skills:
            overlap = len(wanted_skills & job_skills) / len(wanted_skills)
            skill_score = overlap

    if query.roles and query.skills:
        return round(0.6 * role_score + 0.4 * skill_score, 6)
    if query.roles:
        return round(role_score, 6)
    return round(skill_score, 6)


class StructuredRetriever:
    """Recall by structured role/skill similarity."""

    name = "structured_retriever"

    def retrieve(
        self, query: QuerySpec, jobs: list[JobPosting], pool_size: int
    ) -> RetrievalOutcome:
        scored = [
            RetrievedJob(job_id=j.job_id, score=structured_score(query, j),
                         components={"structured": structured_score(query, j)})
            for j in jobs
        ]
        scored = [s for s in scored if s.score > 0.0]
        scored.sort(key=lambda s: (-s.score, s.job_id))
        return RetrievalOutcome(retrieved=scored[:pool_size], initial_pool_size=len(scored))
