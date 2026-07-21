"""Ranking Agent: turns feature computations into auditable RankedJob objects.

Only eligible jobs are scored. Every feature exposes its raw value, normalised
score, weight, weighted contribution and the evidence ids (candidate-side and
job-side) that justify it. Missing (not-applicable) features are handled per the
configured ``missing_feature_policy`` and the choice is fixed for a run.
"""

from __future__ import annotations

from ..config import AppConfig
from ..domain.constraints import EligibilityResult
from ..domain.enums import ConfirmationStatus, EvidenceSource, PersistenceScope
from ..domain.job import ActiveSearchState, JobPosting
from ..domain.recommendation import RankedJob, RankingFeature
from ..evidence_store import EvidenceStore
from . import features as F
from .tie_break import tie_break_key

SCORER_VERSION = "1.0.0"

# feature name -> active-search field(s) providing candidate-side evidence.
_FEATURE_CANDIDATE_FIELDS = {
    "role_match": ["target_roles"],
    "required_skill_match": ["skills_have"],
    "preferred_skill_match": ["skills_have"],
    "location_preference": ["preferred_locations"],
    "work_mode_preference": ["work_modes"],
    "salary_preference": ["salary_min", "salary_currency"],
    "experience_fit": ["years_experience", "experience_level"],
    "industry_preference": ["industries"],
}


class RankingAgent:
    """Scores eligible jobs with an explainable, weighted linear model."""

    name = "ranking_agent"

    def __init__(self, store: EvidenceStore, config: AppConfig) -> None:
        self.store = store
        self.config = config
        self.weights = config.ranking.weights

    def rank(
        self,
        active: ActiveSearchState,
        jobs_by_id: dict[str, JobPosting],
        eligibility: list[EligibilityResult],
    ) -> list[RankedJob]:
        eligible = [e for e in eligibility if e.eligible]
        hard_experience = "experience" in active.hard_constraint_fields
        penalize_unknown_salary = self.config.context.hard_constraint_unknown_default.value == "penalize"

        ranked: list[RankedJob] = []
        for elig in eligible:
            job = jobs_by_id[elig.job_id]
            computed = {
                "role_match": F.role_match(active, job),
                "required_skill_match": F.required_skill_match(active, job),
                "preferred_skill_match": F.preferred_skill_match(active, job),
                "location_preference": F.location_preference(active, job),
                "work_mode_preference": F.work_mode_preference(active, job),
                "salary_preference": F.salary_preference(
                    active, job, self.config.ranking.salary_scale, penalize_unknown_salary),
                "experience_fit": F.experience_fit(active, job, hard_experience),
                "industry_preference": F.industry_preference(active, job),
            }

            # Renormalise weights over applicable features (fixed policy per run).
            applicable = {n: r for n, r in computed.items() if r.applicable}
            weight_sum = sum(self.weights.get(n, 0.0) for n in applicable) or 1.0

            ranking_features: list[RankingFeature] = []
            total = 0.0
            for name, res in computed.items():
                base_w = self.weights.get(name, 0.0)
                if self.config.ranking.missing_feature_policy == "renormalize":
                    eff_w = (base_w / weight_sum) if res.applicable else 0.0
                else:  # penalize: keep original weight, missing scores 0
                    eff_w = base_w
                contribution = round(eff_w * res.normalized, 6)
                if res.applicable:
                    total += contribution
                ranking_features.append(RankingFeature(
                    name=name,
                    raw_value=res.raw_value,
                    normalized_score=res.normalized,
                    weight=round(eff_w, 6),
                    weighted_contribution=contribution,
                    evidence_ids=self._feature_evidence(name, res, active, job),
                    explanation_code=res.code,
                ))

            ranked.append(RankedJob(
                job_id=job.job_id,
                rank=0,
                total_score=round(total, 6),
                features=ranking_features,
                eligibility_result_id=elig.eligibility_result_id,
                skill_gaps=F.skill_gaps(active, job),
                warnings=[],
            ))

        ranked.sort(key=tie_break_key)
        for i, rj in enumerate(ranked):
            rj.rank = i + 1
        return ranked

    # ------------------------------------------------------------- evidence
    def _feature_evidence(self, name, res: "F.FeatureResult", active, job) -> list[str]:
        ids: list[str] = []
        for field in _FEATURE_CANDIDATE_FIELDS.get(name, []):
            ids.extend(active.field_evidence_map.get(field, []))
        for job_field in res.job_fields:
            ids.append(self._job_evidence(job, job_field))
        # de-dupe while preserving order
        seen: set[str] = set()
        out = []
        for i in ids:
            if i and i not in seen:
                seen.add(i)
                out.append(i)
        return out

    def _job_evidence(self, job: JobPosting, field_name: str) -> str:
        value = getattr(job, field_name, None)
        item = self.store.register_field(
            EvidenceSource.JOB_POSTING, job.job_id, field_name, value,
            confidence=1.0, confirmation=ConfirmationStatus.CONFIRMED,
            scope=PersistenceScope.SESSION,
        )
        return item.evidence_id
