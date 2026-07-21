"""Property-style tests for cross-cutting invariants (landing-plan section 22.5)."""

from __future__ import annotations

from jobrec.app_service import AppService
from jobrec.config import load_config
from jobrec.domain.enums import ExperimentVariant


def _service(variant="full"):
    cfg = load_config("configs/experiment_full.yaml", base_dir="configs")
    cfg.experiment.variant = ExperimentVariant(variant)
    return AppService(cfg, "data/processed/jobs.jsonl")


def _recommend(variant="full", text="data analyst in Kuala Lumpur, at least RM4000, hybrid ok"):
    svc = _service(variant)
    svc.create_candidate({"candidate_id": "p", "skills": ["Python", "SQL"], "years_experience": 1,
                          "target_roles": ["Data Analyst"], "preferred_locations": ["Kuala Lumpur"]})
    sid = svc.create_session("p", variant)
    return svc, svc.process_turn(sid, text)


def test_full_never_selects_hard_violating_job():
    svc, res = _recommend("full")
    elig = {e.job_id: e for e in res.decision.eligibility_results}
    for jid in res.decision.selected_job_ids:
        assert elig[jid].eligible
        assert elig[jid].hard_violation_count == 0


def test_total_score_equals_sum_of_contributions():
    _, res = _recommend("full")
    for rj in res.decision.ranked_jobs:
        s = round(sum(f.weighted_contribution for f in rj.features), 6)
        assert abs(s - rj.total_score) < 1e-6


def test_every_claim_has_resolvable_evidence():
    svc, res = _recommend("full")
    # every supported claim resolves to at least one registered evidence id
    store = next(iter(svc._sessions.values()))[1]
    for claim in res.response.claims:
        assert claim.evidence_ids
        assert all(store.exists(e) for e in claim.evidence_ids)


def test_topk_not_exceeded():
    svc, res = _recommend("full")
    assert len(res.decision.selected_job_ids) <= svc.config.experiment.top_k


def test_candidate_version_only_increases():
    svc = _service("full")
    cs = svc.create_candidate({"candidate_id": "v", "skills": ["Python"]})
    assert cs.version == 1
    sid = svc.create_session("v", "full")
    svc.process_turn(sid, "data analyst in Kuala Lumpur at least RM4000")
    reloaded = svc.get_candidate("v")
    assert reloaded.version >= 1


def test_no_context_may_violate_hard_but_full_does_not():
    # full filters the impossible salary; no_context does not (ablation contrast).
    text = "data analyst in Kuala Lumpur, at least RM50000 per month"
    _, full = _recommend("full", text)
    _, nctx = _recommend("no_context", text)
    assert full.response.response_type == "no_match"
    assert nctx.response.response_type == "recommendation"
