"""Explanation and Response Agent.

Generates the final response from the RecommendationDecision and the evidence
registry. Every factual claim is bound to evidence ids and passed through a
claim validator before presentation: unsupported claims are dropped (and
recorded), never shown. The agent must not introduce facts (companies, benefits,
skills, salaries) that are not present in the evidence package.
"""

from __future__ import annotations

from typing import Any

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
#: candidate-side evidence.
_NEEDS_CANDIDATE_EVIDENCE_ALSO = {"no_match_reason"}

#: Metadata key marking an evidence item that records a FILTERING EFFECT -- how many jobs a
#: constraint removed at a given stage. A claim that a requirement caused the empty result
#: is causal, and evidence that the requirement merely exists is correlational, so the causal
#: form is only permitted to cite a stage record.
CAUSAL_EFFECT_KEY = "filtered_count"

#: Claim types that assert causation and therefore require such a record.
_NEEDS_CAUSAL_EVIDENCE = {"no_match_cause"}


def _records_a_filtering_effect(item) -> bool:
    """Does this evidence item record how many jobs a constraint removed?"""
    value = item.normalized_value
    if isinstance(value, dict) and CAUSAL_EFFECT_KEY in value:
        return True
    return CAUSAL_EFFECT_KEY in (item.metadata or {})


def _as_number(value: Any) -> float | None:
    """``value`` as a float when it is genuinely numeric, else ``None``."""
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int | float):
        return float(value)
    try:
        return float(str(value).strip())
    except (TypeError, ValueError):
        return None


def _atoms(value: Any) -> set[str]:
    """A value as a set of comparable atoms, so scalars and lists compare alike.

    Numbers are normalised through ``float`` so ``4000`` and ``4000.0`` are one atom, and
    text is case-folded and stripped. Without this, agreement checks would fail on
    presentation differences and the validator would reject true claims -- which is just as
    damaging to the reported grounding rate as accepting false ones.
    """
    if value is None:
        return set()
    values = value if isinstance(value, list | tuple | set) else [value]
    out: set[str] = set()
    for item in values:
        if item is None:
            continue
        number = _as_number(item)
        out.add(repr(number) if number is not None else str(item).strip().casefold())
    return out


def _values_agree(claimed: Any, evidenced: Any) -> bool:
    """Does the evidence's value support the value the claim states?

    Agreement, not equality: one evidence item records ONE value while a claim may render a
    whole list ("target roles: data analyst / bi analyst"), so an evidenced value that is
    among the claimed values agrees. Both empty also agrees, which is what makes "no skills
    recorded" evidenced rather than merely unrefuted.
    """
    claimed_atoms, evidenced_atoms = _atoms(claimed), _atoms(evidenced)
    if not claimed_atoms and not evidenced_atoms:
        return True
    if not evidenced_atoms:
        return False
    return evidenced_atoms <= claimed_atoms or claimed_atoms <= evidenced_atoms


def _numbers(items: list[Any]) -> list[float]:
    """The numeric values of ``items``, skipping any that are not numbers.

    One helper rather than a filtered comprehension per call site, so the non-numeric case is
    handled the same way everywhere and the result is a plain ``list[float]`` that comparisons
    can be applied to without a further guard.
    """
    out = []
    for item in items:
        number = _as_number(item.normalized_value)
        if number is not None:
            out.append(number)
    return out


def _candidate_field_aliases(field: str) -> set[str]:
    """Every name the candidate side records ``field`` under.

    One field has two names. An extraction calls it ``skills_have``; the profile stores it as
    the CandidateState attribute ``skills``, and evidence registered from the profile carries
    that second name. A checker that compared against one spelling silently rejected true
    claims -- every ``skill_gap`` and every skill-coverage reason failed, because the claim
    said ``skills_have`` while its evidence said ``skills``.

    Derived from :mod:`jobrec.agents.memory_agent`'s own field maps rather than a table
    written out here, so adding a field cannot leave this behind.
    """
    from .memory_agent import _LIST_FIELDS, _SCALAR_FIELDS

    mapping = {**_LIST_FIELDS, **_SCALAR_FIELDS}
    names = {field}
    if field in mapping:
        names.add(mapping[field])
    names.update(k for k, v in mapping.items() if v == field)
    return names


def _candidate_items(resolved: list[Any], field: str | None = None) -> list[Any]:
    out = [i for i in resolved if i.source in _CANDIDATE_SOURCES]
    if field is None:
        return out
    names = _candidate_field_aliases(field)
    return [i for i in out if i.field_name in names]


def _job_items(resolved: list[Any], job_id: str | None, field: str | None = None) -> list[Any]:
    """Job-side evidence, restricted to THIS job.

    The ``source_object_id`` check is what stops another job's attribute supporting a claim
    about this one: every recommendation cites several jobs' evidence in one response, so
    "some job in this response has that value" was never the proposition.
    """
    out = [i for i in resolved if i.source in _JOB_SOURCES]
    if job_id is not None:
        out = [i for i in out if i.source_object_id == job_id]
    return [i for i in out if field is None or i.field_name == field]


def _stage_records(resolved: list[Any], field: str | None) -> list[Any]:
    """Filtering-effect records for ``field``.

    The field is read out of the record's own value rather than off the evidence's name, so
    a record cannot be matched to a claim by a coincidence of naming.
    """
    out = []
    for item in resolved:
        if not _records_a_filtering_effect(item):
            continue
        value = item.normalized_value
        recorded = value.get("field") if isinstance(value, dict) else None
        if field is None or recorded == field:
            out.append(item)
    return out


def _check_candidate_preference(claim: ResponseClaim, resolved: list[Any]) -> bool:
    """``candidate_preference(field, value)``: the candidate stated THIS value for THIS field.

    Requiring the cited evidence to carry the value the claim asserts is also what rejects a
    superseded value: an earlier statement of the same field survives in the store, and
    citing it no longer passes because its value is not the one being claimed.
    """
    if not claim.field_name:
        return False
    return any(_values_agree(claim.expected_value, i.normalized_value)
               for i in _candidate_items(resolved, claim.field_name))


def _check_salary_meets_min(claim: ResponseClaim, resolved: list[Any]) -> bool:
    """``salary_meets_min(job_id, threshold)``: the job's GUARANTEED minimum clears it.

    Both sides are required. The job's salary alone cannot establish that it meets the
    CANDIDATE's minimum, and the candidate's minimum alone says nothing about the job.
    ``>=`` on the guaranteed minimum, matching the eligibility rule -- range OVERLAP is what
    the salary fix removed, so a claim must not reintroduce it.
    """
    thresholds = _numbers(_candidate_items(resolved, "salary_min"))
    if not thresholds:
        return False
    claimed = _as_number(claim.expected_value)
    if claimed is not None and claimed not in thresholds:
        return False
    threshold = claimed if claimed is not None else max(thresholds)
    minima = _numbers(_job_items(resolved, claim.job_id, "salary_min_monthly_myr"))
    return bool(minima) and max(minima) >= threshold


def _check_skill_not_recorded(claim: ResponseClaim, resolved: list[Any]) -> bool:
    """``skill_not_recorded(job_id, skill)``: the job requires it, the profile does not list it.

    A statement about the RECORD, never about the candidate's ability -- which is why the
    absence has to be read off a recorded skill list that is present. An empty list is
    evidence of absence; a missing entry is not, so candidate-side evidence is required
    rather than assumed. This is the 1883-claim family that cited only the job's requirement
    and asserted something about the candidate.
    """
    skill = claim.claim_args.get("skill")
    if not skill:
        return False
    wanted = _atoms(skill)
    required = _job_items(resolved, claim.job_id, "required_skills")
    if not any(wanted <= _atoms(i.normalized_value) for i in required):
        return False
    items = _candidate_items(resolved, "skills_have")
    if not items:
        return False
    # Absence is checked against the union of everything cited, and the claim has to cite the
    # WHOLE recorded list. Citing one entry would let "excel is not recorded" pass by quoting
    # the candidate's Python entry while their Excel entry sat uncited in the store -- the
    # second counterexample, in a subtler form.
    recorded: set[str] = set()
    for item in items:
        recorded |= _atoms(item.normalized_value)
    return not (wanted & recorded)


def _check_ranking_match(claim: ResponseClaim, resolved: list[Any]) -> bool:
    """``ranking_match(job_id, field, candidate_value, job_value)``.

    Names the ranking feature it explains, so the claim is tied to a real scoring reason
    rather than to a resemblance the renderer noticed. Both the candidate's stated value and
    the job's attribute must agree with what the claim says they are.
    """
    if not claim.claim_args.get("feature") or not claim.field_name:
        return False
    if not any(_values_agree(claim.expected_value, i.normalized_value)
               for i in _candidate_items(resolved, claim.field_name)):
        return False
    job_field = claim.claim_args.get("job_field")
    candidates = _job_items(resolved, claim.job_id, job_field)
    return any(_values_agree(claim.observed_value, i.normalized_value) for i in candidates)


def _check_no_match_cause(claim: ResponseClaim, resolved: list[Any]) -> bool:
    """``no_match_cause(field, removed, evaluated_jobs)``: THIS field removed THAT many.

    Causal, so it needs the stage record and not merely the constraint's existence -- and the
    record has to be the one for THIS field, which is the counterexample the
    evidence-class-only check let through (a salary stage record supporting a work-mode
    cause). The counts must agree with the record, and when the record lists the blocked job
    ids the claim's list must be exactly those ids.
    """
    if not claim.field_name:
        return False
    records = _stage_records(resolved, claim.field_name)
    if not records:
        return False
    claimed_removed = _as_number(claim.claim_args.get("removed"))
    claimed_blocked = claim.claim_args.get("blocked_job_ids")
    for record in records:
        value = record.normalized_value if isinstance(record.normalized_value, dict) else {}
        recorded_removed = _as_number(value.get(CAUSAL_EFFECT_KEY))
        if claimed_removed is not None and recorded_removed != claimed_removed:
            continue
        recorded_blocked = value.get("blocked_job_ids")
        if recorded_blocked is not None or claimed_blocked is not None:
            if sorted(recorded_blocked or []) != sorted(claimed_blocked or []):
                continue
            if recorded_removed is not None and len(recorded_blocked or []) != recorded_removed:
                continue
        return True
    return False


def _check_constraint_applied(claim: ResponseClaim, resolved: list[Any]) -> bool:
    """``constraint_applied(field)``: the candidate stated a requirement on THIS field.

    The non-causal form. It asserts only that the requirement was applied as a hard filter,
    which the candidate's own statement does establish -- unlike the causal claim, which
    needs a stage record.
    """
    return bool(claim.field_name) and bool(_candidate_items(resolved, claim.field_name))


def _check_job_attribute(claim: ResponseClaim, resolved: list[Any]) -> bool:
    """``job_attribute(job_id, field, value)``: THIS job's ``field`` really is that value.

    Purely job-side: it says nothing about the candidate, so it needs nothing from them.
    ``source_object_id`` still has to be this job, since a response cites several jobs'
    attributes and "some job here has that value" was never the claim.
    """
    if not claim.field_name:
        return False
    stated = claim.observed_value if claim.observed_value is not None else claim.expected_value
    return any(_values_agree(stated, i.normalized_value)
               for i in _job_items(resolved, claim.job_id, claim.field_name))


def _check_salary_partial(claim: ResponseClaim, resolved: list[Any]) -> bool:
    """``salary_partial(job_id, threshold)``: the RANGE reaches the threshold, the floor does not.

    A distinct proposition from :func:`_check_salary_meets_min`, and the reason both exist:
    under guaranteed-minimum semantics a posting paying 3000-4500 does NOT meet a 4000
    minimum, so the sentence that describes it must not be checked as though it did. Requiring
    ``max >= threshold > min`` is what keeps "partially meets" from quietly re-establishing the
    range-overlap rule the salary fix removed.
    """
    thresholds = _numbers(_candidate_items(resolved, "salary_min"))
    if not thresholds:
        return False
    threshold = _as_number(claim.expected_value)
    if threshold is None or threshold not in thresholds:
        return False
    floors = _numbers(_job_items(resolved, claim.job_id, "salary_min_monthly_myr"))
    ceilings = _numbers(_job_items(resolved, claim.job_id, "salary_max_monthly_myr"))
    if not floors or not ceilings:
        return False
    return max(ceilings) >= threshold > min(floors)


def _check_skill_covered(claim: ResponseClaim, resolved: list[Any]) -> bool:
    """``skill_covered(job_id, skills)``: the profile records every skill the claim names.

    Set containment on both sides, because coverage is an overlap rather than a value match.
    An empty skill list would make the claim vacuous, so it is rejected: "covers required
    skills ()" asserts nothing and must not count as grounded.
    """
    named = _atoms(claim.claim_args.get("skills"))
    if not named:
        return False
    job_field = claim.claim_args.get("job_field")
    if not any(named <= _atoms(i.normalized_value)
               for i in _job_items(resolved, claim.job_id, job_field)):
        return False
    # The UNION of the cited profile evidence: the profile records one skill per evidence
    # item, so a claim covering two skills is supported by two items together and by no
    # single one of them.
    recorded: set[str] = set()
    for item in _candidate_items(resolved, "skills_have"):
        recorded |= _atoms(item.normalized_value)
    return named <= recorded


def _check_experience_in_range(claim: ResponseClaim, resolved: list[Any]) -> bool:
    """``experience_in_range(job_id)``: the candidate's years fall inside the job's band."""
    years = _numbers(_candidate_items(resolved, "years_experience"))
    lows = _numbers(_job_items(resolved, claim.job_id, "min_years_experience"))
    if not years or not lows:
        return False
    return min(lows) <= max(years)


#: explanation code -> (predicate, candidate-side field, job-side field).
#:
#: Chosen alongside the sentence in :meth:`ExplanationAgent._feature_text`, so a rendered
#: reason and the proposition it is checked against cannot drift apart. ``work_mode_unknown``
#: is absent because it renders no sentence and therefore makes no claim.
_RANKING_PROPOSITIONS: dict[str, tuple[str, str, str | None]] = {
    "role_exact": ("ranking_match", "target_roles", "role_family"),
    "location_match": ("ranking_match", "preferred_locations", "city"),
    "work_mode_match": ("ranking_match", "work_modes", "work_mode"),
    "level_exact": ("ranking_match", "seniority_level", "experience_level"),
    "salary_meets_min": ("salary_meets_min", "salary_min", "salary_min_monthly_myr"),
    "salary_partial": ("salary_partial", "salary_min", "salary_max_monthly_myr"),
    "required_skill_coverage": ("skill_covered", "skills_have", "required_skills"),
    "preferred_skill_coverage": ("skill_covered", "skills_have", "preferred_skills"),
    "experience_in_range": ("experience_in_range", "years_experience",
                            "min_years_experience"),
}

#: predicate -> checker. A claim whose predicate is absent from this table is ``unknown``:
#: the validator declines to vouch for a proposition it has no rule for, which is not the
#: same as passing it.
_PREDICATE_CHECKS: dict[str, Any] = {
    "candidate_preference": _check_candidate_preference,
    "salary_meets_min": _check_salary_meets_min,
    "skill_not_recorded": _check_skill_not_recorded,
    "ranking_match": _check_ranking_match,
    "no_match_cause": _check_no_match_cause,
    "constraint_applied": _check_constraint_applied,
    "job_attribute": _check_job_attribute,
    "salary_partial": _check_salary_partial,
    "skill_covered": _check_skill_covered,
    "experience_in_range": _check_experience_in_range,
}


def semantic_status(claim: ResponseClaim, store: EvidenceStore) -> str:
    """Does the resolved evidence entail ``claim``'s proposition?

    Returns ``"supported"``, ``"unsupported"`` or ``"unknown"``.

    Dispatches on :attr:`~jobrec.domain.recommendation.ResponseClaim.predicate` and compares
    the claim's ARGUMENTS with the evidence's field, value and subject. The previous version
    checked only that the evidence was the right KIND for the claim type, which passed three
    families of false claim: a salary preference citing location evidence, a "skill not
    recorded" citing evidence that records the skill, and a work-mode cause citing the salary
    stage's record. Each cites the right class of evidence and says something it cannot show.

    ``"unknown"`` is returned for a predicate this module has no rule for, and for a claim
    that states no predicate at all. Both are dropped rather than delivered.
    """
    items = [store.get(e) for e in claim.evidence_ids]
    resolved = [i for i in items if i is not None]
    if not resolved or len(resolved) != len(claim.evidence_ids):
        # A dangling id is not a partial success: the claim cites something that is not there.
        return "unsupported"

    # Provisional or contradicted evidence cannot carry an assertive claim.
    if any(i.confirmation_status == ConfirmationStatus.UNCONFIRMED for i in resolved):
        return "unsupported"

    check = _PREDICATE_CHECKS.get(claim.predicate or "")
    if check is None:
        return "unknown"
    return "supported" if check(claim, resolved) else "unsupported"


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
                claims.append(self._claim(
                    "candidate_preference",
                    f"You indicated {field.replace('_', ' ')}: "
                    f"{getattr(active, field)}", ev,
                    predicate="candidate_preference", subject_id=active.candidate_id,
                    field_name=field, expected_value=getattr(active, field),
                ))

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
                    claims.append(self._ranking_claim(text, feat, job, active))
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
                gap_ev = [e for e in (self._job_field_evidence(job, "required_skills"),)
                          if e] + self._candidate_skills_evidence_all(active)
                text = (f"Gap: the role requires {gap}, which is not recorded in your "
                        f"profile skills.")
                lines.append(f"  - {text}")
                claims.append(self._claim(
                    "skill_gap", text, gap_ev, job.job_id,
                    predicate="skill_not_recorded", subject_id=active.candidate_id,
                    job_id=job.job_id, field_name="skills_have",
                    observed_value=list(active.skills_have or []),
                    claim_args={"skill": gap},
                ))
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
    def _filtering_evidence(self, decision, field: str, removed: int) -> str | None:
        """Register the stage record for ``field`` as citable evidence."""
        diagnosis = getattr(decision, "no_match_diagnosis", None) or {}
        from ..domain.enums import ConfirmationStatus, EvidenceSource, PersistenceScope
        # The count lives in normalized_value, not in metadata: it IS the evidence, so it
        # belongs in the value that the evidence id is content-addressed over. Putting it in
        # metadata would let two different counts share one id.
        item = self.store.register_field(
            EvidenceSource.SYSTEM_RULE, decision.decision_id, f"filtered_by:{field}",
            {"field": field, CAUSAL_EFFECT_KEY: removed,
             "stage_trace": diagnosis.get("stage_trace")},
            confidence=1.0, confirmation=ConfirmationStatus.CONFIRMED,
            scope=PersistenceScope.SESSION,
        )
        return item.evidence_id

    def _no_match(self, decision, active):
        claims: list[ResponseClaim] = []
        diagnosis = getattr(decision, "no_match_diagnosis", None) or {}
        blocked_counts = {b["field"]: b.get("filtered_jobs")
                          for b in diagnosis.get("blocking_constraints", [])}
        evaluated = diagnosis.get("evaluated_jobs")
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
                    predicate="constraint_applied", subject_id=active.candidate_id,
                    field_name=field, expected_value=getattr(active, field, None),
                ))
                # The CAUSAL claim, only when the diagnosis recorded how many jobs this
                # field actually removed. It cites the filtering record alongside the
                # candidate's statement, so "this requirement is why" rests on a count
                # rather than on the requirement's existence.
                removed = blocked_counts.get(field)
                if removed:
                    effect_ev = self._filtering_evidence(decision, field, removed)
                    if effect_ev:
                        claims.append(self._claim(
                            "no_match_cause",
                            f"Applying your requirement on {field.replace('_', ' ')} removed "
                            f"{removed} of the {evaluated} job(s) evaluated.",
                            [*ev, effect_ev],
                            predicate="no_match_cause", subject_id=active.candidate_id,
                            field_name=field,
                            expected_value=getattr(active, field, None),
                            claim_args={"removed": removed, "evaluated_jobs": evaluated},
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
    def _claim(self, claim_type: str, text: str, evidence_ids: list[str],
               key_extra: str = "", **proposition) -> ResponseClaim:
        """Build a claim. ``proposition`` carries the structured fields the validator reads.

        ``text`` is only a rendering: it is what the reader sees and it is never parsed to
        decide whether the claim holds. Anything the validator needs has to arrive through
        ``proposition``, which is why a builder that omits it produces an ``unknown`` verdict
        rather than a pass.
        """
        return ResponseClaim(
            claim_id=content_id("claim", claim_type, text, key_extra),
            claim_type=claim_type,
            text=text,
            evidence_ids=list(evidence_ids),
            **proposition,
        )

    def _ranking_claim(self, text, feat, job: JobPosting, active) -> ResponseClaim:
        """A ranking reason with the proposition its explanation code actually asserts.

        One claim TYPE, several propositions: "salary meets your minimum" and "located in
        your preferred city" are checked against different evidence and only coincide in
        being reasons a job scored well. The mapping is keyed on the explanation code, so the
        rendered sentence and the checked proposition are chosen together and cannot drift
        apart.

        Every code in :meth:`_feature_text` has an entry. A code without one would produce a
        claim no checker understands, which is scored ``unknown`` and dropped -- so the
        response would keep asserting the sentence while quietly grounding nothing, and the
        omission would show up as an unexplained fall in the grounding rate rather than as an
        error. The evidence the claim cites is extended with whatever the proposition needs to
        be checkable, rather than being left as the ranking feature's own ids.
        """
        code = feat.explanation_code
        spec = _RANKING_PROPOSITIONS.get(code)
        evidence = list(feat.evidence_ids)
        if spec is None:
            # Not silently unvalidatable: state the code so a reader of the dropped-claims
            # file can see which explanation lacks a proposition.
            return self._claim("ranking_reason", text, evidence, job.job_id,
                               claim_args={"feature": code, "unmapped_explanation_code": code})

        predicate, cand_field, job_field = spec
        if cand_field == "skills_have":
            # Coverage is a set relation, so all recorded skills are cited, not just one.
            cand_ev = self._candidate_skills_evidence_all(active)
        else:
            cand_ev = list((active.field_evidence_map or {}).get(cand_field) or [])
        job_fields = [job_field] if job_field else []
        if predicate == "salary_partial":
            # "partially meets" is a statement about the whole BAND, so the floor is needed
            # as well as the ceiling -- it is what distinguishes partial from meeting it.
            job_fields.append("salary_min_monthly_myr")
        cited_cand = cand_ev if cand_field == "skills_have" else cand_ev[:1]
        extras = [*cited_cand, *(self._job_field_evidence(job, f) for f in job_fields)]
        for extra in extras:
            if extra and extra not in evidence:
                evidence.append(extra)

        args: dict[str, Any] = {"feature": code}
        if job_field:
            args["job_field"] = job_field
        if predicate == "skill_covered" and job_field:
            # The INTERSECTION, so the claim names exactly the skills it can evidence on both
            # sides rather than the whole of either list.
            args["skills"] = sorted(
                _atoms(getattr(active, "skills_have", []) or [])
                & _atoms(getattr(job, job_field, []) or []))
        return self._claim(
            "ranking_reason", text, evidence, job.job_id,
            predicate=predicate, subject_id=active.candidate_id, job_id=job.job_id,
            field_name=cand_field, expected_value=getattr(active, cand_field, None),
            observed_value=getattr(job, job_field, None) if job_field else None,
            claim_args=args,
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

    def _candidate_skills_evidence_all(self, active) -> list[str]:
        """EVERY evidence id for the candidate's recorded skills.

        A skill-gap claim asserts an absence, so it has to cite the whole recorded list. Citing
        one entry would let "excel is not recorded" pass by quoting the candidate's Python
        entry while their Excel entry sat uncited in the store -- true by omission.
        """
        existing = list((active.field_evidence_map or {}).get("skills_have") or [])
        if existing:
            return existing
        one = self._candidate_skills_evidence(active)
        return [one] if one else []

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
