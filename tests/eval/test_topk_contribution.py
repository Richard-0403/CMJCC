"""Tests for the R25 top-k score-breakdown table (`topk_contribution_table`).

Covers both halves of the requirement: the per-feature ranking breakdown the exporter
persists with every ranked job in ``recommendation_decision.json`` (R25.1) and the
per-feature contribution table the evaluation pipeline derives from it (R25.2).

The exhaustive ``total_score == sum(weighted_contribution)`` invariant is Property 17
and lives with the ranking unit tests; here the persisted contract and the aggregation
are what is under test.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobrec.app_service import AppService
from jobrec.config import load_config
from jobrec.evaluation.exporters import write_run_bundle
from jobrec_eval.loaders import RunBundle
from jobrec_eval.metrics_extra import topk_contribution_table

UTTERANCE = "I want a data analyst role in Kuala Lumpur, hybrid, at least RM4000."

#: The five breakdown fields R25.1 requires per feature.
BREAKDOWN_FIELDS = ("name", "normalized_score", "weight",
                    "weighted_contribution", "explanation_code")


def _feature(name: str, normalized: float, weight: float, code: str) -> dict:
    """One persisted ``RankingFeature`` dump."""
    return {
        "name": name,
        "raw_value": None,
        "normalized_score": normalized,
        "weight": weight,
        "weighted_contribution": round(normalized * weight, 6),
        "evidence_ids": [],
        "explanation_code": code,
    }


def _ranked_job(job_id: str, rank: int, features: list[dict]) -> dict:
    """One persisted ``RankedJob`` dump, scored from its own features."""
    return {
        "job_id": job_id,
        "rank": rank,
        "total_score": round(sum(f["weighted_contribution"] for f in features), 6),
        "features": features,
        "eligibility_result_id": f"elig-{job_id}",
        "skill_gaps": [],
        "warnings": [],
    }


def _bundle(variant: str, ranked_jobs: list[dict],
            selected: list[str] | None = None) -> RunBundle:
    return RunBundle(
        variant=variant, scenario_id="s1", run_index=0, path=Path("."),
        run_record={"run_id": f"{variant}-s1"},
        decision={
            "ranked_jobs": ranked_jobs,
            "selected_job_ids": (selected if selected is not None
                                 else [rj["job_id"] for rj in ranked_jobs]),
            "no_match": not ranked_jobs,
        },
        response=None, claims=[], handoffs=[], evidence_log=[],
        latency={}, active_search=None, job_context=None,
    )


def _row(df, scope: str, feature: str, rank=None):
    sub = df[(df["scope"] == scope) & (df["feature"] == feature)]
    if rank is not None:
        sub = sub[sub["rank"] == rank]
    assert len(sub) == 1, f"expected one {scope} row for {feature} (rank={rank})"
    return sub.iloc[0]


def test_contribution_shares_and_drivers_are_aggregated_over_the_topk():
    """Each feature's share of the top-k score and its driver count (R25.2)."""
    top = _ranked_job("j1", 1, [
        _feature("role_match", 1.0, 0.4, "role_exact"),
        _feature("location_preference", 1.0, 0.1, "location_match"),
        _feature("industry_preference", 0.0, 0.0, "industry_not_specified"),
    ])
    second = _ranked_job("j2", 2, [
        _feature("role_match", 0.5, 0.4, "role_partial"),
        _feature("location_preference", 1.0, 0.1, "location_match"),
        _feature("industry_preference", 0.0, 0.0, "industry_not_specified"),
    ])
    df = topk_contribution_table([_bundle("full", [top, second])])

    role = _row(df, "variant", "role_match")
    assert role["jobs"] == 2
    assert role["mean_total_score"] == pytest.approx((0.5 + 0.3) / 2)
    assert role["mean_normalized_score"] == pytest.approx(0.75)
    assert role["mean_weight"] == pytest.approx(0.4)
    assert role["mean_contribution"] == pytest.approx((0.4 + 0.2) / 2)
    # role_match carries 0.6 of the 0.8 total score handed out over the top-k.
    assert role["contribution_share"] == pytest.approx(0.6 / 0.8)
    # ...and is the dominant reason for both recommendations.
    assert role["top_driver_jobs"] == 2
    assert role["top_driver_share"] == pytest.approx(1.0)
    assert role["inactive_jobs"] == 0
    # Ties in code frequency resolve by code name, so the picked code is stable.
    assert role["dominant_explanation_code"] == "role_exact"

    location = _row(df, "variant", "location_preference")
    assert location["contribution_share"] == pytest.approx(0.2 / 0.8)
    assert location["top_driver_jobs"] == 0
    assert location["dominant_explanation_code"] == "location_match"

    # An unstated preference is visible as an inert feature rather than missing.
    industry = _row(df, "variant", "industry_preference")
    assert industry["jobs"] == industry["inactive_jobs"] == 2
    assert industry["contribution_share"] == pytest.approx(0.0)
    assert industry["dominant_explanation_code"] == "industry_not_specified"


def test_per_rank_scope_shows_why_the_second_job_ranked_lower():
    """The ``variant_rank`` scope separates rank 1 from rank 2 (R25.2)."""
    top = _ranked_job("j1", 1, [_feature("role_match", 1.0, 0.4, "role_exact")])
    second = _ranked_job("j2", 2, [_feature("role_match", 0.5, 0.4, "role_partial")])
    df = topk_contribution_table([_bundle("full", [top, second])])

    first = _row(df, "variant_rank", "role_match", rank=1)
    lower = _row(df, "variant_rank", "role_match", rank=2)
    assert first["jobs"] == lower["jobs"] == 1
    assert first["mean_contribution"] == pytest.approx(0.4)
    assert lower["mean_contribution"] == pytest.approx(0.2)
    assert first["dominant_explanation_code"] == "role_exact"
    assert lower["dominant_explanation_code"] == "role_partial"
    # The two scopes are tagged, so aggregate rows are never double-counted.
    assert set(df["scope"]) == {"variant", "variant_rank"}
    assert _row(df, "variant", "role_match")["jobs"] == 2


def test_top_k_truncates_and_variants_are_kept_apart():
    """``top_k`` limits the rows a run contributes; variants tally separately."""
    jobs = [_ranked_job(f"j{i}", i, [_feature("role_match", 1.0, 0.4, "role_exact")])
            for i in range(1, 4)]
    bundles = [_bundle("full", jobs), _bundle("no_memory", jobs[:1])]

    df = topk_contribution_table(bundles, top_k=2)
    # The third-ranked job is dropped, so only ranks 1 and 2 are tallied.
    assert set(df[df["scope"] == "variant_rank"]["rank"]) == {1, 2}

    by_variant = df[df["scope"] == "variant"].set_index("variant")
    assert by_variant.loc["full", "jobs"] == 2
    assert by_variant.loc["no_memory", "jobs"] == 1


def test_runs_without_recommendations_read_as_none_not_zero():
    """No-match runs contribute no rows and an empty table still names its columns."""
    no_match = topk_contribution_table([_bundle("full", [], selected=[])])
    assert no_match.empty
    assert "contribution_share" in no_match.columns
    assert "dominant_explanation_code" in no_match.columns

    empty = topk_contribution_table([])
    assert empty.empty
    assert list(empty.columns) == list(no_match.columns)


def test_persisted_decision_carries_the_feature_breakdown(tmp_path):
    """The exporter persists the per-feature breakdown the table needs (R25.1).

    Runs a real deterministic turn, writes the run bundle, reads
    ``recommendation_decision.json`` back and feeds it straight into the table, so
    the persistence half of R25 is verified end to end.
    """
    cfg = load_config("configs/experiment_full.yaml", base_dir="configs")
    svc = AppService(cfg, "data/processed/jobs.jsonl")
    svc.create_candidate({"candidate_id": "c-topk", "skills": ["Python", "SQL"],
                          "years_experience": 3, "preferred_locations": ["Kuala Lumpur"],
                          "work_modes": ["hybrid"]})
    session_id = svc.create_session("c-topk", "full")
    result = svc.process_turn(session_id, UTTERANCE)

    out = write_run_bundle(result, tmp_path / "run", cfg)
    decision = json.loads((out / "recommendation_decision.json").read_text())

    assert decision["selected_job_ids"], "deterministic run recommended nothing"
    ranked_by_id = {rj["job_id"]: rj for rj in decision["ranked_jobs"]}
    for job_id in decision["selected_job_ids"]:
        features = ranked_by_id[job_id]["features"]
        assert features, f"no persisted breakdown for {job_id}"
        for feature in features:
            assert all(field in feature for field in BREAKDOWN_FIELDS)
            assert 0.0 <= feature["normalized_score"] <= 1.0
            assert feature["weight"] >= 0.0

    # The persisted artifact feeds the metric directly.
    bundle = RunBundle(
        variant="full", scenario_id="s1", run_index=0, path=out,
        run_record={"run_id": "real"}, decision=decision, response=None,
        claims=[], handoffs=[], evidence_log=[], latency={},
        active_search=None, job_context=None,
    )
    df = topk_contribution_table([bundle], top_k=cfg.experiment.top_k)

    top_job = ranked_by_id[decision["selected_job_ids"][0]]
    persisted_names = {f["name"] for f in top_job["features"]}
    variant_rows = df[df["scope"] == "variant"]
    assert set(variant_rows["feature"]) == persisted_names
    assert set(variant_rows["jobs"]) == {len(decision["selected_job_ids"])}

    rank_one = df[(df["scope"] == "variant_rank") & (df["rank"] == 1)]
    assert rank_one["mean_total_score"].unique().tolist() == [
        pytest.approx(top_job["total_score"])]
    for name, contribution in ((f["name"], f["weighted_contribution"])
                               for f in top_job["features"]):
        row = rank_one[rank_one["feature"] == name].iloc[0]
        assert row["mean_contribution"] == pytest.approx(contribution)

    # Exactly one feature is credited as the reason the top job ranked first.
    assert rank_one["top_driver_jobs"].sum() == 1
