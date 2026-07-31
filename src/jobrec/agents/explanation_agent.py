"""Explanation and Response Agent.

Generates the final response from the RecommendationDecision and the evidence
registry. Every factual claim is bound to evidence ids and passed through a
claim validator before presentation: unsupported claims are dropped (and
recorded), never shown. The agent must not introduce facts (companies, benefits,
skills, salaries) that are not present in the evidence package.
"""

from __future__ import annotations

from ..config import AppConfig
from ..domain.enums import ConfirmationStatus, EvidenceSource, ResponseType
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


#: The evidence a claim type needs before its proposition follows, as
#: ``claim_type -> (required source, required field predicate, why)``.
#:
#: The point of stating this per type is that the previous check could not distinguish
#: "cites evidence" from "cites the RIGHT evidence", and the difference is where every
#: known failure family lived.
_CANDIDATE_SOURCES = {EvidenceSource.PROFILE, EvidenceSource.DIALOGUE}
_JOB_SOURCES = {EvidenceSource.JOB_POSTING}

#: Claim types whose proposition is about the CANDIDATE and therefore cannot be established
#: from job-side evidence alone.
_NEEDS_CANDIDATE_EVIDENCE = {"candidate_preference", "skill_gap"}
#: Claim types whose proposition is about a JOB.
_NEEDS_JOB_EVIDENCE = {"job_attribute", "ranking_reason", "skill_gap"}
#: ``no_match_reason`` is a claim about the candidate's stated requirement, so it needs
#: candidate-side evidence. It deliberately does NOT assert that the requirement caused the
#: empty result: establishing causality needs a record of what each filtering stage removed,
#: which does not exist yet (P0-5). The wording was changed to match what the evidence can
#: carry rather than the claim being dropped -- dropping it would have left every no-match
#: response with no reasons at all, trading an unsupported explanation for none.
_NEEDS_CANDIDATE_EVIDENCE_ALSO = {"no_match_reason"}


def semantic_status(claim: ResponseClaim, store: EvidenceStore) -> str:
    """Does the resolved evidence entail ``claim``?

    Returns ``"supported"``, ``"unsupported"`` or ``"unknown"``. ``"unknown"`` means the
    checker has no rule for this claim type and declines to vouch for it, which is
    deliberately not the same as passing it.
    """
    items = [store.get(e) for e in claim.evidence_ids]
    resolved = [i for i in items if i is not None]
    if not resolved:
        return "unsupported"

    # Provisional or contradicted evidence cannot carry an assertive claim.
    if any(i.confirmation_status == ConfirmationStatus.UNCONFIRMED for i in resolved):
        return "unsupported"

    sources = {i.source for i in resolved}
    needs_candidate = _NEEDS_CANDIDATE_EVIDENCE | _NEEDS_CANDIDATE_EVIDENCE_ALSO
    if claim.claim_type in needs_candidate and not (sources & _CANDIDATE_SOURCES):
        return "unsupported"
    if claim.claim_type in _NEEDS_JOB_EVIDENCE and not (sources & _JOB_SOURCES):
        return "unsupported"
    if claim.claim_type in ("candidate_preference", "job_attribute", "constraint_result",
                            "ranking_reason", "skill_gap", "no_match_reason"):
        return "supported"
    return "unknown"


def validate_claims(
    claims: list[ResponseClaim], store: EvidenceStore
) -> tuple[list[ResponseClaim], list[ResponseClaim]]:
    """Split claims into (delivered, dropped), scoring trace AND semantics separately.

    ``trace_status`` is the old check: every cited id resolves. It is necessary and it was
    never sufficient. For several claim types the builder registers the evidence it is about
    to cite (see ``_job_field_evidence``), so resolution is true by construction and the
    check could not fail -- which is why it marked all 11197 claims of the official pair
    supported while human raters adjudicated 2349 of them unsupported.

    A claim is delivered only when both dimensions pass. Anything else is dropped and keeps
    the two verdicts, so a reader can tell a dangling reference from evidence that resolves
    but does not establish the point.
    """
    delivered: list[ResponseClaim] = []
    dropped: list[ResponseClaim] = []
    for claim in claims:
        trace = ("supported"
                 if claim.evidence_ids and all(store.exists(e) for e in claim.evidence_ids)
                 else "unsupported")
        semantic = semantic_status(claim, store) if trace == "supported" else "unsupported"
        overall = "supported" if trace == "supported" and semantic == "supported" else (
            "unknown" if semantic == "unknown" else "unsupported")
        scored = claim.model_copy(update={"trace_status": trace,
                                          "semantic_status": semantic,
                                          "support_status": overall})
        (delivered if overall == "supported" else dropped).append(scored)
    return delivered, dropped


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
            #
            # The wording used to be "the role requires X, which is not in your listed
            # skills" while the only evidence cited was the job's required_skills. That
            # evidence establishes the first half and is silent on the second: it cannot
            # show a skill is ABSENT from the candidate. Human raters adjudicated all 1883
            # of these unsupported, and they were right to -- the claim asserted something
            # its evidence could not reach.
            #
            # Two changes. The claim now cites the candidate's recorded skills alongside the
            # job's, so the comparison it describes is actually evidenced. And it asserts
            # only what that evidence supports: the skill is not RECORDED in the profile,
            # which is a statement about the record, not about the candidate's ability.
            for gap in rj.skill_gaps[:2]:
                gap_ev = [
                    e for e in (self._job_field_evidence(job, "required_skills"),
                                self._candidate_skills_evidence(active))
                    if e
                ]
                text = (f"Gap: the role requires {gap}, which is not recorded in your "
                        f"profile skills.")
                lines.append(f"  - {text}")
                claims.append(self._claim("skill_gap", text, gap_ev, job.job_id))
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
                # "limits the results" was a CAUSAL claim resting on evidence that only
                # showed the constraint existed. The empty result may have come from the
                # role scope, the location, an expired posting, or several constraints
                # together, and nothing here could tell them apart -- human raters
                # adjudicated all 156 of these unsupported. Restated as what the evidence
                # does show: this requirement was applied as a hard filter. The causal
                # form needs a per-stage record of what each filter removed, which is P0-5.
                claims.append(self._claim(
                    "no_match_reason",
                    f"Your stated requirement on {field.replace('_', ' ')} was applied as a "
                    f"hard filter.",
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
            claim_type=claim_type,
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

    def _candidate_skills_evidence(self, active) -> str | None:
        """The candidate's recorded skills, as evidence for a skill-gap comparison.

        Prefers the evidence the active search already carries for ``skills_have`` -- that
        is the candidate's own statement or profile entry, with its real provenance. Only
        when the search has none does it register the (possibly empty) recorded list, which
        is what makes "not recorded in your profile" evidenced rather than asserted.
        """
        existing = (active.field_evidence_map or {}).get("skills_have") or []
        if existing:
            return existing[0]
        from ..domain.enums import ConfirmationStatus, EvidenceSource, PersistenceScope
        item = self.store.register_field(
            EvidenceSource.PROFILE, active.candidate_id, "skills_have",
            list(active.skills_have or []),
            confidence=1.0, confirmation=ConfirmationStatus.CONFIRMED,
            scope=PersistenceScope.SESSION,
        )
        return item.evidence_id

    def _job_field_evidence(self, job: JobPosting, field_name: str) -> str | None:
        from ..domain.enums import ConfirmationStatus, EvidenceSource, PersistenceScope
        value = getattr(job, field_name, None)
        item = self.store.register_field(
            EvidenceSource.JOB_POSTING, job.job_id, field_name, value,
            confidence=1.0, confirmation=ConfirmationStatus.CONFIRMED,
            scope=PersistenceScope.SESSION,
        )
        return item.evidence_id
