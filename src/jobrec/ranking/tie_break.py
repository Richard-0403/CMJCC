"""Stable tie-break ordering for ranked jobs (landing-plan section 14.5)."""

from __future__ import annotations

from ..domain.recommendation import RankedJob


def _feature(job: RankedJob, name: str) -> float:
    for f in job.features:
        if f.name == name:
            return f.normalized_score
    return 0.0


def tie_break_key(job: RankedJob):
    """Sort key: higher total, then required-skill, role, salary; then job_id asc.

    Returns a tuple usable with ``sorted`` (negate 'higher-is-better' fields).
    """
    return (
        -round(job.total_score, 6),
        -round(_feature(job, "required_skill_match"), 6),
        -round(_feature(job, "role_match"), 6),
        -round(_feature(job, "salary_preference"), 6),
        job.job_id,
    )
