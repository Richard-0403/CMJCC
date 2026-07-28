"""Job Context and Constraint Agent.

Responsibilities:
- Turn the active-search view into a typed constraint bundle (JobContextState).
- Evaluate each job against those constraints, producing an EligibilityResult
  with a per-constraint audit trail.
- Distinguish hard / soft / unknown / not-applicable outcomes; hard failures
  filter a job out, unknown-hard outcomes are counted separately (never silently
  treated as pass).

The LLM is never involved here: every judgement is deterministic code.
"""

from __future__ import annotations

from datetime import date

from ..config import AppConfig
from ..domain.constraints import (
    ConstraintCheck,
    ConstraintDefinition,
    EligibilityResult,
    JobContextState,
)
from ..domain.enums import ConstraintOutcome, ConstraintStrength, UnknownPolicy
from ..domain.job import ActiveSearchState, JobPosting
from ..taxonomy import canonical_skill, level_rank
from ..utils.hashing import content_id
from ..utils.time import utcnow

SCORER_VERSION = "1.0.0"

# Which active-search fields can act as constraints, and their evaluation order
# (order only affects performance and log readability, not the final logic).
_CONSTRAINT_ORDER = [
    "not_expired",
    "work_authorizations",
    "preferred_locations",
    "work_modes",
    "salary_min",
    "experience",
    "required_skills",
    "employment_types",
    "exclusions",
]


class JobContextAgent:
    """Builds constraint bundles and evaluates job eligibility."""

    name = "job_context_agent"

    def __init__(self, config: AppConfig, reference_date: date | None = None) -> None:
        self.config = config
        self.reference_date = reference_date or date.fromisoformat(config.project.reference_date)

    # ------------------------------------------------------------------ build
    def build_context(
        self, active: ActiveSearchState, catalog_snapshot_id: str
    ) -> JobContextState:
        """Create a JobContextState (constraint bundle) from the active search."""
        constraints: list[ConstraintDefinition] = []
        warnings: list[str] = []
        hard = set(active.hard_constraint_fields)
        default_unknown = self.config.context.hard_constraint_unknown_default

        def ev(field: str) -> list[str]:
            return active.field_evidence_map.get(field, [])

        def strength(field: str) -> ConstraintStrength:
            return ConstraintStrength.HARD if field in hard else ConstraintStrength.SOFT

        def unknown_policy(field: str) -> UnknownPolicy:
            if field not in hard:
                return UnknownPolicy.PASS
            # Field-specific unknown policies for hard constraints.
            if field in {"salary_min"}:
                return default_unknown
            if field in {"work_modes"}:
                return UnknownPolicy.FAIL
            if field in {"work_authorizations"}:
                return UnknownPolicy.CLARIFY
            return default_unknown

        def mk(field: str, operator: str, expected) -> ConstraintDefinition:
            return ConstraintDefinition(
                constraint_id=content_id("con", active.active_search_id, field),
                field_name=field,
                operator=operator,
                expected_value=expected,
                strength=strength(field),
                weight=self.config.ranking.weights.get(_ranking_key(field), 0.0),
                evidence_ids=ev(field),
                unknown_policy=unknown_policy(field),
                rule_id=f"rule.{field}",
            )

        # Job validity / deadline is always a hard system constraint. This is the ONLY
        # place expiry policy is decided: an expired job fails, a job with no deadline
        # passes (``UnknownPolicy.PASS``). ``context.expired_job_policy`` looks like it
        # governs this but is inert -- see the note on ``ContextConfig``. Any claim about
        # expiry handling must cite this constraint, not that key.
        constraints.append(
            ConstraintDefinition(
                constraint_id=content_id("con", active.active_search_id, "not_expired"),
                field_name="not_expired",
                operator="not_expired",
                expected_value=self.reference_date.isoformat(),
                strength=ConstraintStrength.HARD,
                weight=0.0,
                evidence_ids=[],
                unknown_policy=UnknownPolicy.PASS,
                rule_id="rule.not_expired",
            )
        )

        if active.preferred_locations:
            constraints.append(mk("preferred_locations", "in", active.preferred_locations))
        if active.work_modes:
            constraints.append(mk("work_modes", "in", active.work_modes))
        if active.salary_min is not None:
            constraints.append(mk("salary_min", "gte", active.salary_min))
        if active.experience_level or active.years_experience is not None:
            constraints.append(
                mk("experience", "lte", {
                    "level": active.experience_level,
                    "years": active.years_experience,
                })
            )
        if active.work_authorizations:
            constraints.append(mk("work_authorizations", "contains_all", active.work_authorizations))
        if active.employment_types:
            constraints.append(mk("employment_types", "in", active.employment_types))

        # Required-skill hard constraint only when the search marks it hard.
        if "required_skills" in hard and active.skills_have:
            constraints.append(mk("required_skills", "contains_all", active.skills_have))

        # Exclusions are hard by definition.
        if active.exclusions:
            constraints.append(
                ConstraintDefinition(
                    constraint_id=content_id("con", active.active_search_id, "exclusions"),
                    field_name="exclusions",
                    operator="not_in",
                    expected_value=active.exclusions,
                    strength=ConstraintStrength.HARD,
                    weight=0.0,
                    evidence_ids=ev("exclusions"),
                    unknown_policy=UnknownPolicy.PASS,
                    rule_id="rule.exclusions",
                )
            )

        constraints.sort(key=lambda c: _CONSTRAINT_ORDER.index(c.field_name)
                         if c.field_name in _CONSTRAINT_ORDER else 99)

        return JobContextState(
            context_id=content_id("ctx", active.active_search_id, catalog_snapshot_id),
            active_search_id=active.active_search_id,
            catalog_snapshot_id=catalog_snapshot_id,
            constraints=constraints,
            normalized_at=utcnow(),
            normalization_warnings=warnings,
        )

    # --------------------------------------------------------------- evaluate
    def evaluate(self, job: JobPosting, context: JobContextState) -> EligibilityResult:
        """Evaluate a single job against the constraint bundle."""
        checks: list[ConstraintCheck] = []
        hard_violations = 0
        unknown_hard = 0
        reasons: list[str] = []

        for con in context.constraints:
            outcome, observed, code = self._evaluate_one(job, con)
            checks.append(
                ConstraintCheck(
                    constraint_id=con.constraint_id,
                    field_name=con.field_name,
                    outcome=outcome,
                    observed_job_value=observed,
                    expected_candidate_value=con.expected_value,
                    explanation_code=code,
                    evidence_ids=con.evidence_ids,
                )
            )
            if con.strength == ConstraintStrength.HARD:
                if outcome == ConstraintOutcome.FAIL:
                    hard_violations += 1
                    reasons.append(f"{con.field_name}:{code}")
                elif outcome == ConstraintOutcome.UNKNOWN:
                    # Apply the unknown policy for hard constraints.
                    if con.unknown_policy in (UnknownPolicy.FAIL,):
                        hard_violations += 1
                        reasons.append(f"{con.field_name}:unknown_fail")
                    else:
                        unknown_hard += 1
                        reasons.append(f"{con.field_name}:unknown_{con.unknown_policy}")

        eligible = hard_violations == 0
        return EligibilityResult(
            eligibility_result_id=content_id("elig", context.context_id, job.job_id),
            job_id=job.job_id,
            eligible=eligible,
            checks=checks,
            hard_violation_count=hard_violations,
            unknown_hard_constraint_count=unknown_hard,
            filtered_reason_codes=sorted(set(reasons)),
        )

    # ----------------------------------------------------------- per-field ops
    def _evaluate_one(
        self, job: JobPosting, con: ConstraintDefinition
    ) -> tuple[ConstraintOutcome, object, str]:
        field = con.field_name
        if field == "not_expired":
            return self._check_active(job)
        if field == "preferred_locations":
            return self._check_location(job, con.expected_value)
        if field == "work_modes":
            return self._check_work_mode(job, con.expected_value)
        if field == "salary_min":
            return self._check_salary(job, con.expected_value)
        if field == "experience":
            return self._check_experience(job, con.expected_value)
        if field == "work_authorizations":
            return self._check_authorization(job, con.expected_value)
        if field == "employment_types":
            return self._check_employment(job, con.expected_value)
        if field == "required_skills":
            return self._check_required_skills(job, con.expected_value)
        if field == "exclusions":
            return self._check_exclusions(job, con.expected_value)
        return ConstraintOutcome.NOT_APPLICABLE, None, "unhandled_field"

    def _check_active(self, job: JobPosting) -> tuple[ConstraintOutcome, object, str]:
        if not job.is_active:
            return ConstraintOutcome.FAIL, job.is_active, "job_inactive"
        if job.application_deadline is not None:
            if job.application_deadline < self.reference_date:
                return ConstraintOutcome.FAIL, str(job.application_deadline), "deadline_passed"
            return ConstraintOutcome.PASS, str(job.application_deadline), "deadline_ok"
        return ConstraintOutcome.PASS, None, "active_no_deadline"

    def _check_location(self, job, expected) -> tuple[ConstraintOutcome, object, str]:
        wanted = {str(x).lower() for x in expected}
        observed = [v for v in [job.city, job.country, job.region] if v]
        if not observed:
            return ConstraintOutcome.UNKNOWN, None, "location_unknown"
        obs_lower = {str(v).lower() for v in observed}
        # "Malaysia" as a preference matches any Malaysian city.
        if obs_lower & wanted:
            return ConstraintOutcome.PASS, observed, "location_match"
        if "malaysia" in wanted and (job.country or "").lower() == "malaysia":
            return ConstraintOutcome.PASS, observed, "location_country_match"
        return ConstraintOutcome.FAIL, observed, "location_mismatch"

    def _check_work_mode(self, job, expected) -> tuple[ConstraintOutcome, object, str]:
        wanted = {str(x).lower() for x in expected}
        if job.work_mode == "unspecified":
            return ConstraintOutcome.UNKNOWN, job.work_mode, "work_mode_unknown"
        if job.work_mode in wanted:
            return ConstraintOutcome.PASS, job.work_mode, "work_mode_match"
        return ConstraintOutcome.FAIL, job.work_mode, "work_mode_mismatch"

    def _check_salary(self, job, expected) -> tuple[ConstraintOutcome, object, str]:
        cmin = float(expected)
        jmax = job.salary_max_monthly_myr
        jmin = job.salary_min_monthly_myr
        if jmax is None and jmin is None:
            return ConstraintOutcome.UNKNOWN, None, "salary_unknown"
        top = jmax if jmax is not None else jmin
        bottom = jmin if jmin is not None else jmax
        if top is not None and top < cmin:
            return ConstraintOutcome.FAIL, {"min": jmin, "max": jmax}, "salary_below_min"
        if bottom is not None and bottom >= cmin:
            return ConstraintOutcome.PASS, {"min": jmin, "max": jmax}, "salary_meets_min"
        return ConstraintOutcome.PASS, {"min": jmin, "max": jmax}, "salary_range_crosses_min"

    def _check_experience(self, job, expected) -> tuple[ConstraintOutcome, object, str]:
        cand_years = expected.get("years")
        cand_level = expected.get("level")
        # If the job states a minimum years and the candidate is below it -> fail.
        if job.min_years_experience is not None and cand_years is not None:
            if cand_years < job.min_years_experience:
                return ConstraintOutcome.FAIL, job.min_years_experience, "below_min_years"
        # Level comparison: candidate below the job's required level -> fail.
        if job.experience_level and cand_level:
            jr = level_rank(job.experience_level)
            cr = level_rank(cand_level)
            if jr is not None and cr is not None and cr < jr:
                return ConstraintOutcome.FAIL, job.experience_level, "below_required_level"
            if jr is not None and cr is not None:
                return ConstraintOutcome.PASS, job.experience_level, "level_ok"
        if job.min_years_experience is None and not job.experience_level:
            return ConstraintOutcome.UNKNOWN, None, "experience_unknown"
        return ConstraintOutcome.PASS, job.experience_level, "experience_ok"

    def _check_authorization(self, job, expected) -> tuple[ConstraintOutcome, object, str]:
        req = {str(x).upper() for x in job.required_work_authorization}
        if not req:
            return ConstraintOutcome.PASS, [], "no_authorization_required"
        have = {str(x).upper() for x in expected}
        if req.issubset(have):
            return ConstraintOutcome.PASS, sorted(req), "authorization_ok"
        return ConstraintOutcome.FAIL, sorted(req), "authorization_missing"

    def _check_employment(self, job, expected) -> tuple[ConstraintOutcome, object, str]:
        if not job.employment_type:
            return ConstraintOutcome.UNKNOWN, None, "employment_unknown"
        wanted = {str(x).lower() for x in expected}
        if job.employment_type.lower() in wanted:
            return ConstraintOutcome.PASS, job.employment_type, "employment_match"
        return ConstraintOutcome.FAIL, job.employment_type, "employment_mismatch"

    def _check_required_skills(self, job, expected) -> tuple[ConstraintOutcome, object, str]:
        have = {canonical_skill(s) for s in expected}
        need = {canonical_skill(s) for s in job.required_skills}
        if not need:
            return ConstraintOutcome.PASS, [], "no_required_skills"
        missing = sorted(need - have)
        if missing:
            return ConstraintOutcome.FAIL, missing, "missing_required_skills"
        return ConstraintOutcome.PASS, sorted(need), "skills_ok"

    def _check_exclusions(self, job, expected) -> tuple[ConstraintOutcome, object, str]:
        roles = {str(x).lower() for x in expected.get("roles", [])}
        locs = {str(x).lower() for x in expected.get("locations", [])}
        inds = {str(x).lower() for x in expected.get("industries", [])}
        if job.role_family and job.role_family.lower() in roles:
            return ConstraintOutcome.FAIL, job.role_family, "excluded_role"
        if job.city and job.city.lower() in locs:
            return ConstraintOutcome.FAIL, job.city, "excluded_location"
        if job.industry and job.industry.lower() in inds:
            return ConstraintOutcome.FAIL, job.industry, "excluded_industry"
        return ConstraintOutcome.PASS, None, "no_exclusion_hit"


def _ranking_key(field: str) -> str:
    """Map a constraint field to its ranking-weight key (best effort)."""
    return {
        "preferred_locations": "location_preference",
        "work_modes": "work_mode_preference",
        "salary_min": "salary_preference",
        "experience": "experience_fit",
        "required_skills": "required_skill_match",
    }.get(field, field)



def diagnose_no_match(
    eligibility_results: list[EligibilityResult],
    context: JobContextState,
) -> dict:
    """Aggregate blocking constraints when every job was filtered out.

    Returns a structure listing which fields blocked how many jobs and which of
    them are *soft / unknown* relaxation candidates. Hard constraints stated by
    the user are never auto-relaxed; they are only reported.
    """
    field_counts: dict[str, int] = {}
    for res in eligibility_results:
        blocked_fields = {code.split(":", 1)[0] for code in res.filtered_reason_codes}
        for field in blocked_fields:
            field_counts[field] = field_counts.get(field, 0) + 1

    hard_fields = {c.field_name for c in context.constraints if c.strength.value == "hard"}
    blocking = [
        {"field": f, "filtered_jobs": n}
        for f, n in sorted(field_counts.items(), key=lambda kv: -kv[1])
    ]
    # Relaxation candidates: fields that blocked jobs but are NOT user hard
    # constraints (e.g. unknown-policy fields), which could be safely relaxed.
    relaxation = [
        {"field": f, "potential_matches": n, "requires_confirmation": True}
        for f, n in field_counts.items()
        if f not in hard_fields
    ]
    return {
        "no_match": True,
        "blocking_constraints": blocking,
        "relaxation_candidates": relaxation,
    }
