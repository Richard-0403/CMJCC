"""Explanation and Response Agent.

Generates the final response from the RecommendationDecision and the evidence
registry. Every factual claim is bound to evidence ids and passed through a
claim validator before presentation: unsupported claims are dropped (and
recorded), never shown. The agent must not introduce facts (companies, benefits,
skills, salaries) that are not present in the evidence package.
"""

from __future__ import annotations

from ..config import AppConfig
from ..domain.enums import ResponseType
from ..domain.job import ActiveSearchState, JobPosting
from ..domain.recommendation import (
    RecommendationDecision,
    Response,
    ResponseClaim,
)
from ..evidence_store import EvidenceStore
from ..utils.hashing import content_id
from ..utils.time import utcnow


class ClaimValidationError(Exception):
    """Raised only when a *required* claim cannot be grounded (defensive)."""


def validate_claims(
    claims: list[ResponseClaim], store: EvidenceStore
) -> tuple[list[ResponseClaim], list[ResponseClaim]]:
    """Split claims into (supported, dropped).

    A claim is supported only when it has at least one evidence id and every id
    resolves to a registered EvidenceItem.
    """
    supported: list[ResponseClaim] = []
    dropped: list[ResponseClaim] = []
    for claim in claims:
        if claim.evidence_ids and all(store.exists(e) for e in claim.evidence_ids):
            supported.append(claim.model_copy(update={"support_status": "supported"}))
        else:
            dropped.append(claim.model_copy(update={"support_status": "unsupported"}))
    return supported, dropped


class ExplanationAgent:
    """Builds grounded responses (recommendation / no-match / clarification)."""

    name = "explanation_agent"

    def __init__(self, store: EvidenceStore, config: AppConfig) -> None:
        self.store = store
        self.config = config

    def explain(
        self,
        decision: RecommendationDecision,
        active: ActiveSearchState,
        jobs_by_id: dict[str, JobPosting],
    ) -> tuple[Response, list[ResponseClaim]]:
        """Return (response, dropped_claims)."""
        if decision.no_match:
            return self._no_match(decision, active)
        return self._recommendation(decision, active, jobs_by_id)

    # ------------------------------------------------------- recommendation
    def _recommendation(self, decision, active, jobs_by_id):
        claims: list[ResponseClaim] = []
        lines: list[str] = []

        # 1) Need summary (candidate_preference claims).
        summary_bits = []
        if active.target_roles:
            summary_bits.append(" / ".join(active.target_roles))
        if active.preferred_locations:
            summary_bits.append("in " + " / ".join(active.preferred_locations))
        if active.salary_min:
            summary_bits.append(f"salary >= {active.salary_min:.0f} {active.salary_currency or ''}".strip())
        summary = "Based on your request" + (": " + ", ".join(summary_bits) if summary_bits else ".")
        lines.append(summary)
        for field in ["target_roles", "preferred_locations", "salary_min"]:
            ev = active.field_evidence_map.get(field, [])
            if ev:
                claims.append(self._claim("candidate_preference",
                    f"You indicated {field.replace('_', ' ')}: "
                    f"{getattr(active, field)}", ev))

        # 2) Top-k jobs.
        top = decision.ranked_jobs[: self.config.experiment.top_k]
        if not top:
            lines.append("No eligible jobs were found.")
        for rj in top:
            job = jobs_by_id[rj.job_id]
            lines.append(
                f"\n#{rj.rank} {job.title} @ {job.company} "
                f"(match {rj.total_score:.2f})"
            )
            # job attribute + ranking reason claims from the top features
            strong = [f for f in rj.features if f.normalized_score >= 0.6 and f.weight > 0][:4]
            for feat in strong:
                text = self._feature_text(feat.name, feat, job)
                if text:
                    lines.append(f"  - {text}")
                    claims.append(self._claim("ranking_reason", text, feat.evidence_ids, job.job_id))
            # skill gaps
            for gap in rj.skill_gaps[:2]:
                gap_ev = self._job_field_evidence(job, "required_skills")
                text = f"Gap: the role requires {gap}, which is not in your listed skills."
                lines.append(f"  - {text}")
                claims.append(self._claim("skill_gap", text, [gap_ev] if gap_ev else [], job.job_id))
            # unknown-field transparency
            if job.salary_min_monthly_myr is None and active.salary_min is not None:
                text = "Note: this posting does not state a salary."
                lines.append(f"  - {text}")

        # 3) Next-step nudge (non-factual, no claim needed).
        lines.append("\nYou can refine any preference (location, salary, work mode) to adjust these results.")

        response = Response(
            response_id=content_id("resp", decision.decision_id),
            session_id=decision.session_id,
            response_type=ResponseType.RECOMMENDATION.value,
            message="\n".join(lines),
            claims=[],
            created_at=utcnow(),
        )
        supported, dropped = validate_claims(claims, self.store)
        response = response.model_copy(update={"claims": supported})
        return response, dropped

    # ------------------------------------------------------------- no match
    def _no_match(self, decision, active):
        claims: list[ResponseClaim] = []
        lines = ["No jobs currently satisfy all of your hard requirements at once."]
        for code in decision.no_match_reason_codes:
            lines.append(f"  - Blocking condition: {code}")
        # ground the no-match reasons in the candidate's hard-constraint evidence
        for field in active.hard_constraint_fields:
            ev = active.field_evidence_map.get(field, [])
            if ev:
                claims.append(self._claim(
                    "no_match_reason",
                    f"Your hard requirement on {field.replace('_', ' ')} limits the results.",
                    ev,
                ))
        lines.append("You could relax a soft or unconfirmed preference to see more options.")
        response = Response(
            response_id=content_id("resp", decision.decision_id),
            session_id=decision.session_id,
            response_type=ResponseType.NO_MATCH.value,
            message="\n".join(lines),
            claims=[],
            created_at=utcnow(),
        )
        supported, dropped = validate_claims(claims, self.store)
        response = response.model_copy(update={"claims": supported})
        return response, dropped

    # --------------------------------------------------------------- claims
    def _claim(self, claim_type: str, text: str, evidence_ids: list[str], key_extra: str = "") -> ResponseClaim:
        return ResponseClaim(
            claim_id=content_id("claim", claim_type, text, key_extra),
            claim_type=claim_type,  # type: ignore[arg-type]
            text=text,
            evidence_ids=list(evidence_ids),
        )

    def _feature_text(self, name, feat, job: JobPosting) -> str | None:
        code = feat.explanation_code
        mapping = {
            "role_exact": f"Matches your target role ({job.role_family}).",
            "required_skill_coverage": f"Covers required skills ({feat.raw_value}).",
            "preferred_skill_coverage": f"Covers preferred skills ({feat.raw_value}).",
            "location_match": f"Located in {job.city}, matching your preference.",
            "work_mode_match": f"Work mode is {job.work_mode}, matching your preference.",
            "work_mode_unknown": None,
            "salary_meets_min": "Salary meets your stated minimum.",
            "salary_partial": "Salary range partially meets your minimum.",
            "experience_in_range": "Your experience fits the role's range.",
            "level_exact": f"Experience level matches ({job.experience_level}).",
        }
        return mapping.get(code)

    def _job_field_evidence(self, job: JobPosting, field_name: str) -> str | None:
        from ..domain.enums import ConfirmationStatus, EvidenceSource, PersistenceScope
        value = getattr(job, field_name, None)
        item = self.store.register_field(
            EvidenceSource.JOB_POSTING, job.job_id, field_name, value,
            confidence=1.0, confirmation=ConfirmationStatus.CONFIRMED,
            scope=PersistenceScope.SESSION,
        )
        return item.evidence_id
