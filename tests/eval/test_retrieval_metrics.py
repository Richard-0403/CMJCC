"""Tests for the R14 retrieval-layer evaluation (`retrieval_metrics`).

Covers both halves of the requirement: the retrieval artifact the exporter persists
into ``retrieval_results.json`` (initial pool + scores, pool size, full-catalog
fallback, retrieval latency) and the per-variant aggregation of Recall@pool /
relevant-job coverage against the relevance oracle, with retrieval errors reported
separately from ranking errors (R14.1, R14.2).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from jobrec.app_service import AppService
from jobrec.config import load_config
from jobrec.evaluation.exporters import write_run_bundle
from jobrec_eval.loaders import RunBundle
from jobrec_eval.metrics_extra import retrieval_metrics

UTTERANCE = "I want a data analyst role in Kuala Lumpur, at least RM4000."


def _retrieval(pool: list[str], *, scores: list[float] | None = None,
               initial_pool_size: int | None = None, expanded: bool = False,
               latency: float | None = 5.0, executed: bool = True) -> dict:
    """A persisted ``retrieval_results.json`` payload for one run."""
    scores = scores if scores is not None else [1.0] * len(pool)
    return {
        "executed": executed,
        "retrieved_job_ids": list(pool),
        "initial_pool": [{"job_id": jid, "score": s, "components": {}}
                         for jid, s in zip(pool, scores, strict=True)],
        "initial_pool_size": initial_pool_size if initial_pool_size is not None else len(pool),
        "pool_job_ids": list(pool),
        "pool_size": len(pool),
        "requested_pool_size": 50,
        "expanded": expanded,
        "expansion_reason": "empty_recall_fallback" if expanded else None,
        "full_catalog_fallback_count": 1 if expanded else 0,
        "retrieval_latency_ms": latency,
    }


def _bundle(variant: str, scenario_id: str, retrieval: dict, *,
            selected: list[str] | None = None,
            handoffs: list[dict] | None = None) -> RunBundle:
    b = RunBundle(
        variant=variant, scenario_id=scenario_id, run_index=0, path=Path("."),
        run_record={"run_id": f"{variant}-{scenario_id}"},
        decision={"selected_job_ids": selected or []},
        response=None, claims=[], handoffs=handoffs or [], evidence_log=[],
        latency={}, active_search=None, job_context=None,
    )
    b.retrieval = retrieval
    return b


def _labels(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"scenario_id": s, "job_id": j, "relevance_grade": g} for s, j, g in rows])


def _row(df: pd.DataFrame, variant: str) -> pd.Series:
    return df.set_index("variant").loc[variant]


def test_pool_sizes_scores_fallbacks_and_latency_are_aggregated_per_variant():
    """Retrieval diagnostics are summarized per variant, latency kept separate (R14.1)."""
    bundles = [
        _bundle("full", "s1", _retrieval(["j1", "j2"], scores=[0.9, 0.5],
                                         initial_pool_size=7, latency=4.0)),
        # An empty recall that fell back to the whole catalog.
        _bundle("full", "s2", _retrieval(["j1", "j2", "j3", "j4"], scores=[0.0] * 4,
                                         initial_pool_size=0, expanded=True, latency=8.0)),
    ]

    row = _row(retrieval_metrics(bundles), "full")

    assert row["runs"] == row["retrieval_runs"] == 2
    assert row["mean_initial_pool_size"] == pytest.approx(3.5)
    assert row["mean_pool_size"] == pytest.approx(3.0)
    assert row["mean_retrieval_score"] == pytest.approx((0.9 + 0.5) / 6)
    assert row["full_catalog_fallbacks"] == 1
    assert row["fallback_rate"] == pytest.approx(0.5)
    assert row["median_retrieval_latency_ms"] == pytest.approx(6.0)
    assert row["mean_retrieval_latency_ms"] == pytest.approx(6.0)


def test_recall_at_pool_and_error_attribution_separate_retrieval_from_ranking():
    """Recall@pool, coverage, and layer-attributed errors (R14.1, R14.2).

    The oracle marks ``j1``/``j2`` relevant for ``s1``. A variant that pools both and
    recommends one is clean; a variant that pools both but recommends neither is a
    RANKING error; a variant that pools neither is a RETRIEVAL error; and a failed
    retrieval handoff is a retrieval error regardless of recall.
    """
    labels = _labels([("s1", "j1", 3), ("s1", "j2", 2), ("s1", "j9", 0)])
    bundles = [
        _bundle("full", "s1", _retrieval(["j1", "j2", "j9"]), selected=["j1"]),
        _bundle("ranks_badly", "s1", _retrieval(["j1", "j2"]), selected=["j9"]),
        _bundle("recalls_badly", "s1", _retrieval(["j9"]), selected=["j9"]),
        _bundle("handoff_fails", "s1", _retrieval(["j1"]), selected=["j1"], handoffs=[
            {"contract_name": "RetrievalOutcome", "validation_passed": False},
        ]),
    ]

    df = retrieval_metrics(bundles, labels, relevance_threshold=2)

    clean = _row(df, "full")
    assert clean["scored_runs"] == 1
    assert clean["recall_at_pool"] == pytest.approx(1.0)
    assert clean["relevant_job_coverage"] == pytest.approx(1.0)
    assert clean["retrieval_errors"] == 0
    assert clean["ranking_errors"] == 0

    ranking = _row(df, "ranks_badly")
    assert ranking["recall_at_pool"] == pytest.approx(1.0)
    assert ranking["retrieval_errors"] == 0
    assert ranking["ranking_errors"] == 1
    assert ranking["ranking_error_rate"] == pytest.approx(1.0)

    retrieval = _row(df, "recalls_badly")
    assert retrieval["recall_at_pool"] == pytest.approx(0.0)
    assert retrieval["relevant_job_coverage"] == pytest.approx(0.0)
    assert retrieval["retrieval_errors"] == 1
    # A retrieval failure is never also charged to ranking.
    assert retrieval["ranking_errors"] == 0

    broken_handoff = _row(df, "handoff_fails")
    assert broken_handoff["retrieval_errors"] == 1
    assert broken_handoff["ranking_errors"] == 0


def test_missing_data_reads_as_none_rather_than_a_misleading_zero():
    """No labels and no executed retrieval both read N/A (module convention)."""
    without_labels = _row(retrieval_metrics([_bundle("full", "s1", _retrieval(["j1"]))]), "full")
    assert without_labels["scored_runs"] == 0
    assert without_labels["recall_at_pool"] is None
    assert without_labels["relevant_job_coverage"] is None

    # A clarification short-circuit never reaches retrieval.
    not_executed = _row(retrieval_metrics([
        _bundle("full", "s1", _retrieval([], executed=False, latency=None)),
    ]), "full")
    assert not_executed["runs"] == 1
    assert not_executed["retrieval_runs"] == 0
    assert not_executed["mean_pool_size"] is None
    assert not_executed["fallback_rate"] is None
    assert not_executed["retrieval_error_rate"] is None
    assert not_executed["ranking_error_rate"] is None

    empty = retrieval_metrics([])
    assert empty.empty
    assert "recall_at_pool" in empty.columns
    assert "retrieval_errors" in empty.columns


def test_persisted_retrieval_results_carry_the_required_fields(tmp_path):
    """The exporter writes a self-describing retrieval artifact the metric can read.

    Runs a real deterministic turn, writes the run bundle, and reads
    ``retrieval_results.json`` back, so the persisted contract behind R14.1 is
    verified end to end.
    """
    cfg = load_config("configs/experiment_full.yaml", base_dir="configs")
    svc = AppService(cfg, "data/processed/jobs.jsonl")
    svc.create_candidate({"candidate_id": "c-retrieval", "skills": ["Python", "SQL"],
                          "years_experience": 1, "preferred_locations": ["Kuala Lumpur"],
                          "work_modes": ["hybrid"]})
    session_id = svc.create_session("c-retrieval", "full")
    result = svc.process_turn(session_id, UTTERANCE)

    out = write_run_bundle(result, tmp_path / "run", cfg)
    data = json.loads((out / "retrieval_results.json").read_text())

    assert data["executed"] is True
    assert data["retrieved_job_ids"], "deterministic retrieval recalled nothing"
    assert [r["job_id"] for r in data["initial_pool"]] == data["retrieved_job_ids"]
    assert all(r["score"] is not None for r in data["initial_pool"])
    assert data["initial_pool_size"] >= data["pool_size"] == len(data["pool_job_ids"])
    assert data["pool_size"] <= data["requested_pool_size"]
    assert data["full_catalog_fallback_count"] == 0
    assert data["retrieval_latency_ms"] is not None

    # The persisted artifact feeds the metric directly: label one pooled job relevant.
    bundle = _bundle("full", "s1", data, selected=[data["pool_job_ids"][0]])
    labels = _labels([("s1", data["pool_job_ids"][0], 3)])
    row = _row(retrieval_metrics([bundle], labels), "full")
    assert row["mean_pool_size"] == pytest.approx(float(data["pool_size"]))
    assert row["recall_at_pool"] == pytest.approx(1.0)
    assert row["retrieval_errors"] == 0
    assert row["ranking_errors"] == 0
