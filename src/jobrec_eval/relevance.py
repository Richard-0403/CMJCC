"""Automatic relevance oracle (transparent proxy for human relevance labels).

IMPORTANT: This is NOT human annotation. It is a deterministic reference grader
used because no human raters are available. It is fully documented and its
output is emitted as `relevance_labels.csv` so real raters can replace it; the
report flags this as a construct-validity threat.

Grade definition (0-3), per (scenario, job), applied to the WHOLE catalog so the
ideal DCG is computed over the full label universe (no pooling bias):

  1. Evaluate the job against the scenario's authoritative hard constraints
     (taken from the `full` variant's JobContextState). Any hard violation or an
     inactive/expired job => grade 0 (the guide's rule: a hard violation forces
     relevance 0 regardless of text similarity).
  2. Otherwise combine role fit and required-skill coverage:
        score = 0.5 * role_score + 0.5 * skill_coverage
     where role_score = 1.0 exact role-family match, 0.5 partial (title token
     overlap), 0.0 otherwise; skill_coverage = covered required skills / required.
  3. Map: grade 3 if score >= 0.75, 2 if >= 0.5, 1 if > 0, else 0. A role
     mismatch (role_score == 0) caps the grade at 0.
"""

from __future__ import annotations

import pandas as pd

from jobrec.agents.job_context_agent import JobContextAgent
from jobrec.config import AppConfig
from jobrec.domain.constraints import JobContextState
from jobrec.domain.job import JobPosting
from jobrec.taxonomy import canonical_role, canonical_skill

ORACLE_VERSION = "1.0.0"


def build_references(bundles) -> dict[str, dict]:
    """Per scenario, take the full-variant reference constraints + active search."""
    refs: dict[str, dict] = {}
    for b in bundles:
        if b.variant != "full":
            continue
        if b.scenario_id in refs:
            continue
        if b.job_context and b.active_search:
            refs[b.scenario_id] = {"job_context": b.job_context, "active_search": b.active_search}
    return refs


def _role_score(active: dict, job: JobPosting) -> float:
    wanted = {canonical_role(r) for r in active.get("target_roles", [])}
    if not wanted:
        return 0.0
    job_role = job.role_family or canonical_role(job.title)
    if job_role in wanted:
        return 1.0
    tokens = set(job.normalized_title.split())
    if any(set(r.split()) & tokens for r in wanted):
        return 0.5
    return 0.0


def _skill_coverage(active: dict, job: JobPosting) -> float:
    if not job.required_skills:
        return 1.0
    have = {canonical_skill(s) for s in active.get("skills_have", [])}
    covered = sum(1 for s in job.required_skills if canonical_skill(s) in have)
    return covered / len(job.required_skills)


def grade_catalog(
    catalog: list[JobPosting], references: dict[str, dict], config: AppConfig
) -> pd.DataFrame:
    """Grade every (scenario, job) pair. Returns a relevance_labels DataFrame."""
    agent = JobContextAgent(config)
    rows = []
    for scenario_id, ref in references.items():
        context = JobContextState.model_validate(ref["job_context"])
        active = ref["active_search"]
        for job in catalog:
            elig = agent.evaluate(job, context)
            role = _role_score(active, job)
            skill = _skill_coverage(active, job)
            if not elig.eligible or role == 0.0:
                grade = 0
            else:
                score = 0.5 * role + 0.5 * skill
                grade = 3 if score >= 0.75 else 2 if score >= 0.5 else 1 if score > 0 else 0
            rows.append({
                "scenario_id": scenario_id, "job_id": job.job_id, "rater_id": "auto_oracle",
                "relevance_grade": grade,
                "hard_violation_observed": (not elig.eligible),
                "role_fit": role, "skill_fit": round(skill, 3),
                "oracle_version": ORACLE_VERSION,
            })
    return pd.DataFrame(rows)


def grade_lookup(labels: pd.DataFrame) -> dict[tuple[str, str], int]:
    return {(r.scenario_id, r.job_id): int(r.relevance_grade) for r in labels.itertuples()}


def ideal_grades(labels: pd.DataFrame, scenario_id: str) -> list[int]:
    """Return all grades for a scenario, descending (for IDCG)."""
    sub = labels[labels["scenario_id"] == scenario_id]["relevance_grade"].tolist()
    return sorted((int(g) for g in sub), reverse=True)
