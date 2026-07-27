"""Unit tests for memory conflicts, ranking invariants and claim validation."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from jobrec.agents.candidate_understanding import CandidateUnderstandingAgent
from jobrec.agents.explanation_agent import validate_claims
from jobrec.agents.job_context_agent import JobContextAgent
from jobrec.agents.memory_agent import MemoryAgent
from jobrec.config import load_config
from jobrec.domain.dialogue import DialogueState
from jobrec.domain.enums import UnknownPolicy
from jobrec.domain.job import ActiveSearchState, JobPosting
from jobrec.domain.recommendation import ResponseClaim
from jobrec.evidence_store import EvidenceStore
from jobrec.orchestration.cmjcc import CMJCC, CMJCCInput
from jobrec.ranking.scoring import RankingAgent


def _cmjcc_run(profile, text, cfg):
    store = EvidenceStore()
    mem = MemoryAgent(store, cfg)
    cand = mem.create_candidate_state(profile)
    dlg = DialogueState(session_id="s", candidate_id=profile["candidate_id"], version=1, turns=[])
    ex = CandidateUnderstandingAgent().extract(text)
    dlg = mem.append_turn(dlg, "candidate", text)
    out = CMJCC(store, cfg).run(CMJCCInput(cand, dlg, ex, "snap", cfg, "run"))
    return cand, out, store


def test_temporary_override_does_not_pollute_long_term(config):
    cand, out, _ = _cmjcc_run(
        {"candidate_id": "c", "skills": ["Python"], "preferred_locations": ["Penang"]},
        "I want a data analyst role in Kuala Lumpur now.", config,
    )
    # long-term profile keeps Penang
    assert [p.value for p in cand.preferred_locations] == ["Penang"]
    # active search uses Kuala Lumpur
    assert out.active_search_state.preferred_locations == ["Kuala Lumpur"]


def test_factual_years_conflict_triggers_clarification(config):
    _, out, _ = _cmjcc_run(
        {"candidate_id": "c", "skills": ["Python"], "years_experience": 1, "target_roles": ["Data Analyst"]},
        "Actually I have 3 years experience.", config,
    )
    assert any(c.field_name == "years_experience" and c.resolution == "ask_clarification"
               for c in out.conflicts)
    assert out.active_search_state.years_experience == 1.0  # not silently overwritten


def test_work_mode_merges(config):
    _, out, _ = _cmjcc_run(
        {"candidate_id": "c", "skills": ["Python"], "work_modes": ["remote"], "target_roles": ["Data Analyst"]},
        "hybrid is also fine", config,
    )
    assert set(out.active_search_state.work_modes) == {"remote", "hybrid"}


def test_claim_validator_drops_unsupported(store):
    from jobrec.domain.enums import ConfirmationStatus, EvidenceSource, PersistenceScope

    item = store.register_field(
        source=EvidenceSource.PROFILE, source_object_id="c", field_name="skills",
        normalized_value="python", confidence=1.0,
        confirmation=ConfirmationStatus.CONFIRMED, scope=PersistenceScope.LONG_TERM,
    )
    supported = ResponseClaim(claim_id="ok", claim_type="candidate_preference",
                              text="knows python", evidence_ids=[item.evidence_id])
    bad = ResponseClaim(claim_id="bad", claim_type="job_attribute",
                        text="great culture", evidence_ids=["does-not-exist"])
    keep, drop = validate_claims([supported, bad], store)
    assert [c.claim_id for c in keep] == ["ok"]
    assert [c.claim_id for c in drop] == ["bad"]


def test_config_hash_stable_and_variant_sensitive():
    a = load_config("configs/experiment_full.yaml", base_dir="configs")
    b = load_config("configs/experiment_full.yaml", base_dir="configs")
    c = load_config("configs/experiment_no_context.yaml", base_dir="configs")
    assert a.config_hash() == b.config_hash()
    assert a.config_hash() != c.config_hash()

# ---------------------------------------------------------------------------
# Property-based test (Property 17)
# ---------------------------------------------------------------------------

# The agent rounds weights and contributions to 6 dp (the exporter does too), so
# the additive identity can only be off by representation noise, never by more
# than a fraction of the last rounded digit.
_TOTAL_EPS = 1e-6
_FEATURE_EPS = 1e-5

_ROLES = ["data analyst", "software engineer", "data engineer"]
_SKILLS = ["python", "sql", "excel", "power bi", "java"]
_LOCATIONS = ["Kuala Lumpur", "Penang", "Johor Bahru"]
_WORK_MODES = ["onsite", "hybrid", "remote"]
_LEVELS = ["entry", "junior", "mid", "senior"]
_HARD_FIELDS = ["salary_min", "preferred_locations", "work_modes", "experience", "required_skills"]


def _job(job_id: str, **overrides) -> JobPosting:
    title = overrides.pop("title", "Data Analyst")
    base = dict(
        job_id=job_id,
        title=title,
        company="ACME",
        description=f"{title} role",
        normalized_title=title.lower(),
        source_snapshot_id="snap",
        ingested_at=datetime(2025, 1, 1, tzinfo=UTC),
        raw_payload_hash=f"hash-{job_id}",
        is_active=True,
        application_deadline=None,
    )
    base.update(overrides)
    return JobPosting(**base)


@st.composite
def _arbitrary_jobs(draw) -> JobPosting:
    """A job whose attributes are unconstrained by the active search."""
    title = draw(st.sampled_from(_ROLES)).title()
    jmin = draw(st.one_of(st.none(), st.floats(min_value=1000.0, max_value=15000.0)))
    jmax = None if jmin is None else jmin + draw(st.floats(min_value=0.0, max_value=3000.0))
    ymin = draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=10.0)))
    return _job(
        "job-x",
        title=title,
        role_family=draw(st.one_of(st.none(), st.sampled_from(_ROLES))),
        required_skills=draw(st.lists(st.sampled_from(_SKILLS), max_size=3, unique=True)),
        preferred_skills=draw(st.lists(st.sampled_from(_SKILLS), max_size=2, unique=True)),
        city=draw(st.one_of(st.none(), st.sampled_from(_LOCATIONS))),
        country=draw(st.sampled_from([None, "Malaysia", "Singapore"])),
        work_mode=draw(st.sampled_from(["onsite", "hybrid", "remote", "unspecified"])),
        salary_min_monthly_myr=jmin,
        salary_max_monthly_myr=jmax,
        min_years_experience=ymin,
        max_years_experience=None if ymin is None else ymin + 5.0,
        experience_level=draw(st.one_of(st.none(), st.sampled_from(_LEVELS))),
    )


@st.composite
def _active_and_jobs(draw):
    """An active search plus a job set that always contains one compatible job.

    Seeding a compatible job keeps the property non-vacuous: hard-constraint
    filtering runs for real, but at least one job always reaches the ranker.
    """
    roles = draw(st.lists(st.sampled_from(_ROLES), max_size=2, unique=True))
    skills = draw(st.lists(st.sampled_from(_SKILLS), max_size=3, unique=True))
    locations = draw(st.lists(st.sampled_from(_LOCATIONS), max_size=2, unique=True))
    modes = draw(st.lists(st.sampled_from(_WORK_MODES), max_size=2, unique=True))
    salary_min = draw(st.one_of(st.none(), st.floats(min_value=1500.0, max_value=12000.0)))
    years = draw(st.one_of(st.none(), st.floats(min_value=0.0, max_value=20.0)))
    level = draw(st.one_of(st.none(), st.sampled_from(_LEVELS)))
    hard = draw(st.lists(st.sampled_from(_HARD_FIELDS), max_size=3, unique=True))

    active = ActiveSearchState(
        active_search_id="as-prop", session_id="s", candidate_id="c",
        candidate_state_version=1, dialogue_state_version=1,
        target_roles=roles, skills_have=skills, preferred_locations=locations,
        salary_min=salary_min, salary_currency="MYR" if salary_min else None,
        work_modes=modes, experience_level=level, years_experience=years,
        employment_types=[], work_authorizations=[], exclusions={},
        hard_constraint_fields=hard, soft_preference_fields=[], unknown_fields=[],
        clarification_required_fields=[], field_evidence_map={},
        generated_at=datetime(2025, 1, 1, tzinfo=UTC),
    )

    # A job that satisfies every hard constraint the active search can express.
    compatible = _job(
        "job-fit",
        title=(roles[0] if roles else "data analyst").title(),
        role_family=roles[0] if roles else "data analyst",
        required_skills=list(skills),
        preferred_skills=list(skills[:1]),
        city=locations[0] if locations else "Kuala Lumpur",
        country="Malaysia",
        work_mode=modes[0] if modes else "hybrid",
        salary_min_monthly_myr=(salary_min or 3000.0) + 500.0,
        salary_max_monthly_myr=(salary_min or 3000.0) + 1500.0,
        min_years_experience=0.0,
        max_years_experience=40.0,
        experience_level=level,
    )

    others = draw(st.lists(_arbitrary_jobs(), max_size=3))
    jobs = [compatible, *[j.model_copy(update={"job_id": f"job-{i}"}) for i, j in enumerate(others)]]
    return active, jobs


# Feature: cmjcc-experiment-readiness, Property 17: Ranking total_score equals the sum of
# feature contributions
@settings(max_examples=100, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    case=_active_and_jobs(),
    policy=st.sampled_from(["renormalize", "penalize"]),
    unknown_default=st.sampled_from([UnknownPolicy.FAIL, UnknownPolicy.PENALIZE]),
)
def test_property_total_score_equals_sum_of_feature_contributions(
    config, case, policy, unknown_default
) -> None:
    """Every RankedJob's total_score is exactly the sum of its feature contributions.

    Each contribution is itself ``normalized_score * weight``, so the whole score
    is inspectable from the persisted breakdown alone.

    **Validates: Requirements 25.1, 31.1**
    """
    active, jobs = case
    cfg = config.model_copy(deep=True)
    cfg.ranking.missing_feature_policy = policy
    cfg.context.hard_constraint_unknown_default = unknown_default

    context_agent = JobContextAgent(cfg)
    ctx = context_agent.build_context(active, "snap")
    eligibility = [context_agent.evaluate(job, ctx) for job in jobs]
    jobs_by_id = {job.job_id: job for job in jobs}

    ranked = RankingAgent(EvidenceStore(), cfg).rank(active, jobs_by_id, eligibility)

    # Only eligible jobs are scored, and every eligible job is (non-vacuity).
    assert len(ranked) == sum(1 for e in eligibility if e.eligible)
    assert ranked, "at least the seeded compatible job must be ranked"

    for rj in ranked:
        assert rj.features, "a scored job must expose its feature breakdown"
        contributions = [f.weighted_contribution for f in rj.features]
        assert rj.total_score == pytest.approx(sum(contributions), abs=_TOTAL_EPS)
        for f in rj.features:
            # The breakdown is explanatory: each row is normalized x weight.
            assert f.weighted_contribution == pytest.approx(
                f.normalized_score * f.weight, abs=_FEATURE_EPS
            )
