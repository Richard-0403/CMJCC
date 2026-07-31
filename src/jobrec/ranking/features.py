"""Feature computations for ranking.

Each feature is computed by a pure function that returns a FeatureResult with a
raw value, a normalised [0,1] score, an explanation code, an applicability flag
(so missing features can be renormalised), and the job fields it relied on (for
evidence binding).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.job import ActiveSearchState, JobPosting
from ..taxonomy import TRANSFERABLE_SKILLS, canonical_role, canonical_skill, level_rank


@dataclass(frozen=True)
class FeatureResult:
    raw_value: float | int | str | None
    normalized: float
    code: str
    applicable: bool = True
    job_fields: list[str] = field(default_factory=list)


def _skill_credit(candidate_skills: set[str], target_skill: str) -> float:
    """Full credit for exact/synonym match, partial for a transferable skill."""
    target = canonical_skill(target_skill)
    if target in candidate_skills:
        return 1.0
    for owned in candidate_skills:
        if target in TRANSFERABLE_SKILLS.get(owned, []):
            return 0.5
    return 0.0


def role_match(active: ActiveSearchState, job: JobPosting) -> FeatureResult:
    if not active.target_roles:
        return FeatureResult(None, 0.0, "role_not_specified", applicable=False)
    wanted = {canonical_role(r) for r in active.target_roles}
    job_role = job.role_family or canonical_role(job.title)
    if job_role in wanted:
        return FeatureResult(job_role, 1.0, "role_exact", job_fields=["role_family", "title"])
    # Partial: token overlap between job title and any wanted role.
    tokens = set(job.normalized_title.split())
    if any(set(r.split()) & tokens for r in wanted):
        return FeatureResult(job_role, 0.5, "role_partial", job_fields=["title"])
    return FeatureResult(job_role, 0.0, "role_mismatch", job_fields=["role_family", "title"])


def required_skill_match(active: ActiveSearchState, job: JobPosting) -> FeatureResult:
    if not job.required_skills:
        return FeatureResult(None, 0.0, "no_required_skills", applicable=False)
    have = {canonical_skill(s) for s in active.skills_have}
    credits = [_skill_credit(have, s) for s in job.required_skills]
    score = sum(credits) / len(credits)
    matched = sum(1 for c in credits if c >= 1.0)
    return FeatureResult(f"{matched}/{len(job.required_skills)}", round(score, 4),
                         "required_skill_coverage", job_fields=["required_skills"])


def preferred_skill_match(active: ActiveSearchState, job: JobPosting) -> FeatureResult:
    if not job.preferred_skills:
        return FeatureResult(None, 0.0, "no_preferred_skills", applicable=False)
    have = {canonical_skill(s) for s in active.skills_have}
    credits = [_skill_credit(have, s) for s in job.preferred_skills]
    score = sum(credits) / len(credits)
    matched = sum(1 for c in credits if c >= 1.0)
    return FeatureResult(f"{matched}/{len(job.preferred_skills)}", round(score, 4),
                         "preferred_skill_coverage", job_fields=["preferred_skills"])


def location_preference(active: ActiveSearchState, job: JobPosting) -> FeatureResult:
    if not active.preferred_locations:
        return FeatureResult(None, 0.0, "location_not_specified", applicable=False)
    wanted = {v.lower() for v in active.preferred_locations}
    observed = {v.lower() for v in [job.city, job.country, job.region] if v}
    if observed & wanted or ("malaysia" in wanted and (job.country or "").lower() == "malaysia"):
        return FeatureResult(job.city, 1.0, "location_match", job_fields=["city", "country"])
    if not observed:
        return FeatureResult(None, 0.0, "location_unknown", job_fields=["city"])
    return FeatureResult(job.city, 0.0, "location_mismatch", job_fields=["city", "country"])


def work_mode_preference(active: ActiveSearchState, job: JobPosting) -> FeatureResult:
    if not active.work_modes:
        return FeatureResult(None, 0.0, "work_mode_not_specified", applicable=False)
    wanted = {v.lower() for v in active.work_modes}
    if job.work_mode == "unspecified":
        return FeatureResult(job.work_mode, 0.5, "work_mode_unknown", job_fields=["work_mode"])
    if job.work_mode in wanted:
        return FeatureResult(job.work_mode, 1.0, "work_mode_match", job_fields=["work_mode"])
    return FeatureResult(job.work_mode, 0.0, "work_mode_mismatch", job_fields=["work_mode"])


def salary_preference(active: ActiveSearchState, job: JobPosting, salary_scale: float,
                      penalize_unknown: bool) -> FeatureResult:
    if active.salary_min is None:
        return FeatureResult(None, 0.0, "salary_not_specified", applicable=False)
    cmin = active.salary_min
    jmin = job.salary_min_monthly_myr
    jmax = job.salary_max_monthly_myr
    if jmin is None and jmax is None:
        if penalize_unknown:
            return FeatureResult(None, 0.5, "salary_unknown_penalized", job_fields=["salary_min"])
        return FeatureResult(None, 0.0, "salary_unknown", applicable=False)
    # ``job_fields`` names the field the comparison actually used, and it used to name the
    # RAW ``salary_min``. That field is in the posting's own currency and period, so a claim
    # citing it as evidence for "salary meets your stated minimum" showed a reader
    # ``job_posting:salary_min=1350`` against a stated 4000 -- while the comparison had
    # correctly used the normalised 4725 MYR. The conclusion was right and the evidence
    # could not support it, which is how 275 of these claims came to be adjudicated
    # unsupported by human raters. Cite the projection that was compared.
    if jmin is not None and jmin >= cmin:
        return FeatureResult(jmin, 1.0, "salary_meets_min",
                             job_fields=["salary_min_monthly_myr"])
    top = jmax if jmax is not None else jmin
    if top is not None:
        score = max(0.0, min((top - cmin) / max(salary_scale, 1.0), 1.0))
        return FeatureResult(top, round(score, 4), "salary_partial",
                             job_fields=["salary_min_monthly_myr",
                                         "salary_max_monthly_myr"])
    return FeatureResult(None, 0.0, "salary_unknown", applicable=False)


def experience_fit(active: ActiveSearchState, job: JobPosting, hard_experience: bool) -> FeatureResult:
    years = active.years_experience
    # Prefer explicit years-vs-range comparison.
    if years is not None and (job.min_years_experience is not None or job.max_years_experience is not None):
        lo = job.min_years_experience if job.min_years_experience is not None else 0.0
        hi = job.max_years_experience if job.max_years_experience is not None else float("inf")
        if lo <= years <= hi:
            return FeatureResult(years, 1.0, "experience_in_range", job_fields=["min_years_experience", "max_years_experience"])
        if years < lo:
            gap = lo - years
            if gap <= 1.0 and not hard_experience:
                return FeatureResult(years, 0.7, "slightly_below_min", job_fields=["min_years_experience"])
            if gap <= 2.0:
                return FeatureResult(years, 0.4, "below_min", job_fields=["min_years_experience"])
            return FeatureResult(years, 0.0, "far_below_min", job_fields=["min_years_experience"])
        # above range
        return FeatureResult(years, 0.7, "above_range", job_fields=["max_years_experience"])
    # Fall back to level comparison.
    if active.experience_level and job.experience_level:
        cr = level_rank(active.experience_level)
        jr = level_rank(job.experience_level)
        if cr is not None and jr is not None:
            if cr == jr:
                return FeatureResult(job.experience_level, 1.0, "level_exact", job_fields=["experience_level"])
            if abs(cr - jr) == 1:
                return FeatureResult(job.experience_level, 0.6, "level_adjacent", job_fields=["experience_level"])
            return FeatureResult(job.experience_level, 0.2, "level_far", job_fields=["experience_level"])
    return FeatureResult(None, 0.0, "experience_unknown", applicable=False)


def industry_preference(active: ActiveSearchState, job: JobPosting) -> FeatureResult:
    # Candidate industries are optional and often unset in this prototype.
    return FeatureResult(None, 0.0, "industry_not_specified", applicable=False)


def skill_gaps(active: ActiveSearchState, job: JobPosting) -> list[str]:
    """Required job skills the candidate does not (fully) possess."""
    have = {canonical_skill(s) for s in active.skills_have}
    gaps = []
    for s in job.required_skills:
        if _skill_credit(have, s) < 1.0:
            gaps.append(canonical_skill(s))
    return sorted(set(gaps))
