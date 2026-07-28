"""Human-adjudicated relevance labels as a first-class metric source (checklist item 10).

The checklist requires that adjudicated human labels RECOMPUTE NDCG@5 / Precision@5 / mean
graded relevance and that the automatic oracle be COMPARED against them. Nothing did that
before: the human file only fed the kappa numbers, disagreements were resolved by
``round((rater_1 + rater_2) / 2)`` with no audit trail, and ``analysis_plan.yaml`` hardcoded
``relevance_source: automatic_oracle``.

What is asserted here:

- **Adjudication** (:mod:`jobrec_eval.annotation`): an ``adjudicated`` column is preferred
  over the averaging fallback, the returned dicts say which path produced the gold, and a
  rater disagreement with no adjudicated verdict is counted as unadjudicated instead of
  being quietly averaged.
- **The label table**: an adjudicated file loads into the exact shape the oracle table has,
  so :class:`~jobrec_eval.metrics.MetricsComputer` consumes either one unchanged.
- **The pipeline**: ``--relevance-source human`` genuinely changes the three ranking
  metrics, fails loudly with no labels, writes the oracle-vs-human comparison table, and
  records the real source plus the label file's provenance in the analysis plan.
- **The report**: the relevance-source wording is correct in BOTH modes -- in particular no
  "no human raters were used" claim survives under human mode.

Every human label used here is an obviously-synthetic fixture (see
:data:`_SYNTHETIC_GRADE_CYCLE` and the ``notes`` column it writes): no human judgement is
fabricated. Pipeline runs write only into ``tmp_path``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import yaml

from jobrec_eval.annotation import (
    ADJUDICATION_COLUMN,
    ADJUDICATION_ROUNDED_MEAN,
    HUMAN_RATER_ID,
    RELEVANCE_LABEL_COLUMNS,
    MissingAdjudicatedLabelsError,
    claim_agreement,
    load_adjudicated_relevance_labels,
    relevance_agreement,
)
from jobrec_eval.cli import (
    RELEVANCE_COMPARISON_COLUMNS,
    RELEVANCE_METRICS,
    RELEVANCE_SOURCE_HUMAN,
    RELEVANCE_SOURCE_ORACLE,
    relevance_source_comparison,
    run_pipeline,
    select_relevance_labels,
)
from jobrec_eval.relevance import grade_lookup, ideal_grades
from jobrec_eval.report import generate_markdown

CONFIG = "configs/experiment_full.yaml"
SCENARIOS = "evaluation/data/scenarios_subset.jsonl"
CATALOG = "data/processed/jobs.jsonl"
VARIANTS = ["full", "no_memory"]

#: Grades the synthetic fixture assigns to the returned jobs of every scenario, cycling by
#: rank. Deliberately unlike anything the oracle produces (which grades returned jobs high),
#: so a metric computed from these labels cannot coincide with the oracle's by accident.
_SYNTHETIC_GRADE_CYCLE = (0, 3, 1, 2)

#: Stamped into the fixture's ``notes`` column so no reader can mistake it for real data.
_SYNTHETIC_NOTE = "SYNTHETIC FIXTURE - not a human judgement"


# --------------------------------------------------------------------- fixtures
def _oracle_labels(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame([{"scenario_id": s, "job_id": j, "rater_id": "auto_oracle",
                          "relevance_grade": g} for s, j, g in rows])


def _human_file(tmp_path: Path, rows: list[dict], name: str = "relevance_labels_human.csv"
                ) -> Path:
    path = tmp_path / name
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ------------------------------------------- adjudicated column vs averaging fallback
def test_adjudicated_column_is_preferred_over_the_rounded_mean_fallback(tmp_path):
    """The gold is the ``adjudicated`` value, not ``round((rater_1 + rater_2) / 2)``.

    Both raters are deliberately wrong about the adjudicated verdict and the oracle agrees
    with their rounded mean, so the averaging path would report perfect oracle-vs-human
    agreement while the adjudicated gold does not.
    """
    path = _human_file(tmp_path, [
        {"scenario_id": "s1", "job_id": "j1", "rater_1": 1, "rater_2": 2,
         "adjudicated": 0, "notes": _SYNTHETIC_NOTE},
        {"scenario_id": "s1", "job_id": "j2", "rater_1": 2, "rater_2": 3,
         "adjudicated": 0, "notes": _SYNTHETIC_NOTE},
    ])
    # round((1+2)/2) == 2 and round((2+3)/2) == 2 -> the fallback would match the oracle.
    oracle = _oracle_labels([("s1", "j1", 2), ("s1", "j2", 2)])

    out = relevance_agreement(path, oracle)

    assert out["adjudication_source"] == ADJUDICATION_COLUMN
    assert out["n_adjudicated"] == 2
    assert out["unadjudicated_disagreements"] == 0
    assert out["n_gold_items"] == 2
    # Gold is 0 against an oracle of 2: agreement is NOT the perfect 1.0 the mean would give.
    assert out["oracle_vs_human_weighted_kappa"] != pytest.approx(1.0)
    # The label table built from the same file carries the adjudicated grades, not the mean.
    labels = load_adjudicated_relevance_labels(path)
    assert sorted(labels.labels["relevance_grade"]) == [0, 0]


def test_file_without_an_adjudicated_column_reports_the_fallback_it_used(tmp_path):
    """A legacy file still yields agreement numbers, but labelled as the heuristic."""
    path = _human_file(tmp_path, [
        {"scenario_id": "s1", "job_id": "j1", "rater_1": 1, "rater_2": 2},
        {"scenario_id": "s1", "job_id": "j2", "rater_1": 3, "rater_2": 3},
    ])

    out = relevance_agreement(path, _oracle_labels([("s1", "j1", 2), ("s1", "j2", 3)]))

    assert out["adjudication_source"] == ADJUDICATION_ROUNDED_MEAN
    assert out["n_adjudicated"] == 0
    assert out["n_rater_concordant"] == 1
    # The averaged disagreement is still counted as unadjudicated, so the report can flag it.
    assert out["unadjudicated_disagreements"] == 1
    # A heuristic never produces the label table behind the published ranking metrics.
    assert load_adjudicated_relevance_labels(path) is None


def test_unadjudicated_disagreements_are_counted_not_averaged(tmp_path):
    """With the column present, a blank verdict on a disagreement is excluded and counted."""
    path = _human_file(tmp_path, [
        {"scenario_id": "s1", "job_id": "j1", "rater_1": 3, "rater_2": 3,
         "adjudicated": "", "notes": _SYNTHETIC_NOTE},
        {"scenario_id": "s1", "job_id": "j2", "rater_1": 0, "rater_2": 3,
         "adjudicated": "", "notes": _SYNTHETIC_NOTE},
        {"scenario_id": "s1", "job_id": "j3", "rater_1": 0, "rater_2": 2,
         "adjudicated": 1, "notes": _SYNTHETIC_NOTE},
    ])

    out = relevance_agreement(path, _oracle_labels(
        [("s1", "j1", 3), ("s1", "j2", 2), ("s1", "j3", 2)]))

    assert out["adjudication_source"] == ADJUDICATION_COLUMN
    assert out["n_items"] == 3
    assert out["n_adjudicated"] == 1        # j3
    assert out["n_rater_concordant"] == 1   # j1, the raters agreed
    assert out["unadjudicated_disagreements"] == 1  # j2, excluded from the gold
    assert out["n_gold_items"] == 2

    # The dropped row never reaches the label table, and no averaged 2 appears for j2.
    labels = load_adjudicated_relevance_labels(path).labels
    assert set(zip(labels["job_id"], labels["relevance_grade"], strict=True)) == {
        ("j1", 3), ("j3", 1)}
    assert labels["job_id"].tolist() == ["j1", "j3"]


def test_claim_agreement_prefers_the_adjudicated_column_and_counts_the_rest(tmp_path):
    """Same adjudication rule for claims, including validator-vs-human agreement."""
    path = _human_file(tmp_path, [
        {"run_id": "r1", "claim_id": "c1", "rater_1": 1, "rater_2": 1, "validator": 1,
         "adjudicated": "", "notes": _SYNTHETIC_NOTE},
        {"run_id": "r1", "claim_id": "c2", "rater_1": 0, "rater_2": 1, "validator": 1,
         "adjudicated": 0, "notes": _SYNTHETIC_NOTE},
        {"run_id": "r1", "claim_id": "c3", "rater_1": 0, "rater_2": 1, "validator": 0,
         "adjudicated": "", "notes": _SYNTHETIC_NOTE},
    ], name="claim_annotations_human.csv")

    out = claim_agreement(path)

    assert out["adjudication_source"] == ADJUDICATION_COLUMN
    assert out["n_items"] == 3
    assert out["n_adjudicated"] == 1        # c2, an explicit verdict
    assert out["n_rater_concordant"] == 1   # c1, the raters agreed
    assert out["unadjudicated_disagreements"] == 1  # c3, excluded from the gold
    assert out["n_gold_items"] == 2
    assert "validator_vs_human_kappa" in out


def test_claim_agreement_without_the_column_reports_the_fallback(tmp_path):
    path = _human_file(tmp_path, [
        {"run_id": "r1", "claim_id": "c1", "rater_1": 1, "rater_2": 0, "validator": 1},
        {"run_id": "r1", "claim_id": "c2", "rater_1": 1, "rater_2": 1, "validator": 1},
    ], name="claim_annotations_human.csv")

    out = claim_agreement(path)

    assert out["adjudication_source"] == ADJUDICATION_ROUNDED_MEAN
    assert out["unadjudicated_disagreements"] == 1
    assert out["n_gold_items"] == 2


# ------------------------------------------------- the label table is a drop-in table
def test_adjudicated_file_loads_into_the_oracle_label_shape(tmp_path):
    """Same columns and dtypes as the oracle table, so nothing downstream special-cases it."""
    path = _human_file(tmp_path, [
        {"scenario_id": "s1", "job_id": "j1", "rater_1": 2, "rater_2": 3,
         "adjudicated": 3, "notes": _SYNTHETIC_NOTE},
        {"scenario_id": "s1", "job_id": "j2", "rater_1": 0, "rater_2": 1,
         "adjudicated": 1, "notes": _SYNTHETIC_NOTE},
        {"scenario_id": "s2", "job_id": "j1", "rater_1": 0, "rater_2": 0,
         "adjudicated": "", "notes": _SYNTHETIC_NOTE},
    ])

    loaded = load_adjudicated_relevance_labels(path)

    assert list(loaded.labels.columns) == RELEVANCE_LABEL_COLUMNS
    assert set(loaded.labels["rater_id"]) == {HUMAN_RATER_ID}
    assert loaded.labels["relevance_grade"].dtype.kind == "i"
    assert loaded.labels["scenario_id"].map(type).eq(str).all()
    # The functions MetricsComputer consumes read it without any adaptation.
    assert grade_lookup(loaded.labels) == {("s1", "j1"): 3, ("s1", "j2"): 1, ("s2", "j1"): 0}
    assert ideal_grades(loaded.labels, "s1") == [3, 1]


def test_loaded_labels_carry_the_file_provenance(tmp_path):
    """Path, content hash and per-row tallies, so a reader can tell WHICH labels were used."""
    path = _human_file(tmp_path, [
        {"scenario_id": "s1", "job_id": "j1", "rater_1": 2, "rater_2": 2,
         "adjudicated": "", "notes": _SYNTHETIC_NOTE},
        {"scenario_id": "s1", "job_id": "j2", "rater_1": 0, "rater_2": 3,
         "adjudicated": 2, "notes": _SYNTHETIC_NOTE},
        {"scenario_id": "s1", "job_id": "j3", "rater_1": 0, "rater_2": 3,
         "adjudicated": "", "notes": _SYNTHETIC_NOTE},
    ])

    prov = load_adjudicated_relevance_labels(path).provenance

    assert prov["path"] == str(path)
    assert prov["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert prov["rows_in_file"] == 3
    assert prov["graded_pairs"] == 2
    assert prov["adjudicated_pairs"] == 1
    assert prov["rater_concordant_pairs"] == 1
    assert prov["unadjudicated_disagreements_dropped"] == 1
    assert prov["adjudication_source"] == ADJUDICATION_COLUMN


def test_missing_file_and_unusable_file_read_as_no_labels(tmp_path):
    assert load_adjudicated_relevance_labels(tmp_path / "nope.csv") is None
    # Column present but nothing adjudicated and every row disputed -> no gold at all.
    empty = _human_file(tmp_path, [
        {"scenario_id": "s1", "job_id": "j1", "rater_1": 0, "rater_2": 3,
         "adjudicated": "", "notes": _SYNTHETIC_NOTE}])
    assert load_adjudicated_relevance_labels(empty) is None


@pytest.mark.parametrize(("rows", "message"), [
    ([{"scenario_id": "s1", "job_id": "j1", "rater_1": 3, "rater_2": 3, "adjudicated": 7}],
     "grades must be 0-3"),
    ([{"scenario_id": "s1", "job_id": "j1", "rater_1": 3, "rater_2": 3, "adjudicated": 3},
      {"scenario_id": "s1", "job_id": "j1", "rater_1": 1, "rater_2": 1, "adjudicated": 1}],
     "must be labelled once"),
])
def test_unusable_label_values_raise_rather_than_being_coerced(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        load_adjudicated_relevance_labels(_human_file(tmp_path, rows))


# ------------------------------------------------------- source selection & comparison
def test_human_source_without_labels_fails_instead_of_falling_back():
    oracle = _oracle_labels([("s1", "j1", 3)])

    with pytest.raises(MissingAdjudicatedLabelsError, match="relevance_labels_human.csv"):
        select_relevance_labels("human", oracle, None)

    labels, recorded = select_relevance_labels("oracle", oracle, None)
    assert recorded == RELEVANCE_SOURCE_ORACLE
    assert labels is oracle


def _variant_summary_stub(values: dict[str, dict[str, float]]) -> pd.DataFrame:
    return pd.DataFrame([
        {"variant": variant,
         **{f"{m}_mean": metrics[m] for m in RELEVANCE_METRICS},
         **{f"{m}_n": 4 for m in RELEVANCE_METRICS}}
        for variant, metrics in values.items()])


def test_comparison_table_shape_and_delta_direction():
    """One row per variant x ranking metric; delta is human − oracle."""
    oracle = _variant_summary_stub({
        "full": {"ndcg_at_5": 0.8, "precision_at_5": 0.6, "mean_graded_relevance": 2.4}})
    human = _variant_summary_stub({
        "full": {"ndcg_at_5": 0.5, "precision_at_5": 0.2, "mean_graded_relevance": 1.2}})

    table = relevance_source_comparison(oracle, human)

    assert list(table.columns) == RELEVANCE_COMPARISON_COLUMNS
    assert list(table["metric"]) == RELEVANCE_METRICS
    assert list(table["variant"]) == ["full"] * len(RELEVANCE_METRICS)
    row = table[table["metric"] == "ndcg_at_5"].iloc[0]
    assert row["oracle"] == pytest.approx(0.8)
    assert row["human"] == pytest.approx(0.5)
    assert row["delta"] == pytest.approx(0.5 - 0.8)
    assert row["n_oracle"] == row["n_human"] == 4


def test_comparison_table_keeps_its_shape_without_human_labels():
    """The human cells are empty, never imputed from the oracle."""
    oracle = _variant_summary_stub({
        "full": {"ndcg_at_5": 0.8, "precision_at_5": 0.6, "mean_graded_relevance": 2.4}})

    table = relevance_source_comparison(oracle, None)

    assert list(table.columns) == RELEVANCE_COMPARISON_COLUMNS
    assert len(table) == len(RELEVANCE_METRICS)
    assert table["oracle"].notna().all()
    assert table["human"].isna().all()
    assert table["delta"].isna().all()
    assert table["n_human"].isna().all()


# ------------------------------------------------------------- report wording, both modes
_METRIC_KEYS = ["ndcg_at_5", "precision_at_5", "hcsr", "task_success", "grounding",
                "handoff_success", "turn_count", "total_latency_ms"]


def _report_data(**overrides) -> dict:
    data = {
        "experiment": {
            "experiment_id": "exp-relsource", "reference_date": "2026-01-01",
            "catalog_snapshot_id": "catalog-2026-01", "catalog_hash": "abc123def456789",
            "variants": ["full", "no_memory"], "scenario_count": 2, "repeat_count": 1,
            "run_count": 4, "bootstrap_iterations": 200, "bootstrap_seed": 2026,
            "eval_version": "1.0.0",
        },
        "oracle_version": "1.0.0",
        "scenario_type_counts": {"multi_turn": 2},
        "n_memory_dependent": 2, "n_context_dependent": 1,
        "variant_summary": [{"variant": v, **{f"{k}_mean": 0.5 for k in _METRIC_KEYS}}
                            for v in ["full", "no_memory"]],
        "scenario_variant": [
            {"scenario_id": f"s{i}", "scenario_type": "multi_turn", "variant": v,
             "ndcg_at_5": 0.7, "hcsr": 0.8, "task_success": 1.0, "grounding": 0.9}
            for i in range(2) for v in ["full", "no_memory"]],
        "memory_contribution": [], "context_contribution": [],
        "overall_comparisons": [],
        "error_summary": "Runs: 4; system failures: 0; task-unsuccessful runs: 0.",
    }
    data.update(overrides)
    return data


def _human_source_block() -> dict:
    return {
        "selected": RELEVANCE_SOURCE_HUMAN, "flag": "human", "oracle_version": "1.0.0",
        "human_labels_available": True,
        "human_labels": {
            "path": "tmp/relevance_labels_human.csv", "sha256": "0123456789abcdef",
            "graded_pairs": 8, "scenarios": 2, "adjudicated_pairs": 6,
            "rater_concordant_pairs": 2, "unadjudicated_disagreements_dropped": 1,
            "adjudication_source": ADJUDICATION_COLUMN,
        },
        "retrieval_labels": RELEVANCE_SOURCE_ORACLE,
    }


def test_report_under_oracle_mode_keeps_the_no_human_raters_statement():
    md = generate_markdown(_report_data())
    squeezed = " ".join(md.replace(">", " ").split())

    assert "Relevance is scored by a deterministic automatic oracle, not human raters." in squeezed
    assert "No human raters were used in this run." in squeezed
    assert ("Relevance source for NDCG@5 / Precision@5 / Mean Graded Relevance: "
            "**automatic oracle** (version 1.0.0), not human raters.") in squeezed
    assert "relevance uses an automatic oracle, not human judgement" in squeezed


def test_report_under_human_mode_never_claims_no_human_raters():
    """The header, §4 and §12 must all be true when human labels produced the numbers."""
    md = generate_markdown(_report_data(
        relevance_source=_human_source_block(),
        relevance_agreement={
            "n_items": 9, "raw_agreement_raters": 0.5, "weighted_kappa_raters": 0.6,
            "oracle_vs_human_weighted_kappa": 0.4,
            "adjudication_source": ADJUDICATION_COLUMN, "n_adjudicated": 6,
            "n_rater_concordant": 2, "unadjudicated_disagreements": 1, "n_gold_items": 8,
        }))
    squeezed = " ".join(md.replace(">", " ").split())

    # No false claim survives anywhere in the document.
    assert "No human raters were used" not in squeezed
    assert "not human raters" not in squeezed
    assert "relevance uses an automatic oracle, not human judgement" not in squeezed
    assert "human-annotated relevance and a real LLM backend are the natural next steps" \
        not in squeezed

    # Header, §1/§5/§6/§7 source line, §4 and §12 all name the human source.
    assert ("Relevance is scored by human relevance labels from two raters after "
            "adjudication, not by the automatic oracle.") in squeezed
    assert ("Relevance source for NDCG@5 / Precision@5 / Mean Graded Relevance: "
            "**adjudicated human labels**") in squeezed
    assert "tmp/relevance_labels_human.csv" in squeezed
    assert "sha256 `0123456789ab`" in squeezed
    assert "Human relevance labels were used in this run:" in squeezed
    assert "relevance uses human labels from two raters after adjudication" in squeezed
    # The adjudication trail is stated, not the heuristic.
    assert "6 adjudicated row(s) plus 2 row(s) the two raters already agreed on" in squeezed
    assert "**1 disagreement(s) remain unadjudicated**" in squeezed


def test_report_flags_a_rounded_mean_file_as_not_adjudicated():
    """§4 must not present the averaging heuristic as a completed adjudication."""
    md = generate_markdown(_report_data(relevance_agreement={
        "n_items": 4, "raw_agreement_raters": 0.5, "weighted_kappa_raters": 0.3,
        "oracle_vs_human_weighted_kappa": 0.2,
        "adjudication_source": ADJUDICATION_ROUNDED_MEAN, "n_adjudicated": 0,
        "n_rater_concordant": 2, "unadjudicated_disagreements": 2, "n_gold_items": 4,
    }))
    squeezed = " ".join(md.split())

    assert "**Adjudication is incomplete for this file**" in squeezed
    assert f"adjudication_source={ADJUDICATION_ROUNDED_MEAN}" in squeezed
    assert "resolved by the legacy rounded-mean heuristic" in squeezed


def test_report_oracle_mode_says_human_labels_exist_when_they_do():
    """Oracle numbers with human labels on disk: neither claim may be dropped."""
    block = {**_human_source_block(), "selected": RELEVANCE_SOURCE_ORACLE, "flag": "oracle"}
    squeezed = " ".join(generate_markdown(
        _report_data(relevance_source=block)).replace(">", " ").split())

    assert "**automatic oracle** (version 1.0.0), not human raters." in squeezed
    assert "Adjudicated human labels are available for this run" in squeezed
    assert "the reported ranking metrics are the oracle's" in squeezed
    assert "No human raters were used" not in squeezed


def test_report_renders_the_comparison_table_and_the_retrieval_decision():
    rows = [{"variant": "full", "metric": m, "oracle": 0.8, "human": 0.5, "delta": -0.3,
             "n_oracle": 4, "n_human": 4} for m in RELEVANCE_METRICS]
    md = generate_markdown(_report_data(relevance_source=_human_source_block(),
                                        relevance_source_comparison=rows))

    assert ("### 5.5 Relevance source: automatic oracle vs adjudicated human labels") in md
    assert "| variant | metric | oracle | human | Δ (human − oracle) | n(oracle) | n(human) |" in md
    assert "| full | ndcg_at_5 | 0.800 | 0.500 | -0.300 | 4 | 4 |" in md
    squeezed = " ".join(md.split())
    assert "**automatic oracle in both modes**" in squeezed
    assert "judged pool" in squeezed
    assert "metrics/relevance_source_comparison.csv" in squeezed


# ------------------------------------------------------------------ pipeline wiring
@pytest.fixture(scope="module")
def oracle_run(tmp_path_factory) -> SimpleNamespace:
    """A deterministic pipeline run whose scenario directory holds NO human labels."""
    root = tmp_path_factory.mktemp("relevance-source")
    data_dir = root / "data"
    data_dir.mkdir()
    scenarios = data_dir / Path(SCENARIOS).name
    scenarios.write_text(Path(SCENARIOS).read_text(encoding="utf-8"), encoding="utf-8")
    result = run_pipeline(CONFIG, str(scenarios), CATALOG, str(root / "oracle"), 1, None,
                          200, 2026, variants=VARIANTS)
    out = Path(result["out_dir"])
    return SimpleNamespace(root=root, scenarios=scenarios, out=out,
                           runs_dir=root / "oracle" / "_runs" / out.name)


@pytest.fixture(scope="module")
def human_labels_path(oracle_run) -> Path:
    """Synthetic adjudicated labels for the returned pairs, deliberately unlike the oracle.

    Both rater columns disagree with each other AND with ``adjudicated``, so a metric that
    matched the rounded mean instead of the adjudicated verdict would be visible.
    """
    recs = pd.read_csv(oracle_run.out / "normalized" / "recommendations.csv")
    rows = []
    for scenario_id, group in recs.sort_values("rank").groupby("scenario_id"):
        for index, job_id in enumerate(dict.fromkeys(group["job_id"])):
            grade = _SYNTHETIC_GRADE_CYCLE[index % len(_SYNTHETIC_GRADE_CYCLE)]
            rows.append({
                "scenario_id": scenario_id, "job_id": job_id,
                "rater_1": (grade + 1) % 4, "rater_2": (grade + 2) % 4,
                "adjudicated": grade, "notes": _SYNTHETIC_NOTE,
            })
    path = oracle_run.scenarios.parent / "relevance_labels_human.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def human_run(oracle_run, human_labels_path) -> SimpleNamespace:
    """The same runs re-analysed with ``--relevance-source human`` (bundles reused)."""
    result = run_pipeline(CONFIG, str(oracle_run.scenarios), CATALOG,
                          str(oracle_run.root / "human"), 1, str(oracle_run.runs_dir),
                          200, 2026, variants=VARIANTS, relevance_source="human")
    return SimpleNamespace(out=Path(result["out_dir"]), labels=human_labels_path)


def _variant_metric(out: Path, variant: str, metric: str) -> float:
    summary = pd.read_csv(out / "metrics" / "variant_summary.csv").set_index("variant")
    return summary.loc[variant, f"{metric}_mean"]


def test_pipeline_requesting_human_labels_without_any_fails_loudly(oracle_run, tmp_path):
    """No silent fallback: the run stops instead of publishing oracle numbers as human."""
    scenarios = tmp_path / Path(SCENARIOS).name
    scenarios.write_text(oracle_run.scenarios.read_text(encoding="utf-8"), encoding="utf-8")

    with pytest.raises(MissingAdjudicatedLabelsError, match="Refusing to fall back"):
        run_pipeline(CONFIG, str(scenarios), CATALOG, str(tmp_path / "out"), 1,
                     str(oracle_run.runs_dir), 200, 2026, variants=VARIANTS,
                     relevance_source="human")


def test_human_source_changes_every_ranking_metric(oracle_run, human_run):
    """NDCG@5 / P@5 / MGR are recomputed from the adjudicated labels, not the oracle."""
    for metric in RELEVANCE_METRICS:
        oracle_value = _variant_metric(oracle_run.out, "full", metric)
        human_value = _variant_metric(human_run.out, "full", metric)
        assert pd.notna(oracle_value) and pd.notna(human_value), metric
        assert human_value != pytest.approx(oracle_value), metric


def test_label_independent_metrics_are_untouched_by_the_source(oracle_run, human_run):
    """Only grade-derived metrics move: HCSR, grounding and task success are unchanged."""
    for metric in ("hcsr", "grounding", "task_success", "handoff_success"):
        assert (_variant_metric(human_run.out, "full", metric)
                == pytest.approx(_variant_metric(oracle_run.out, "full", metric)))


def test_comparison_csv_has_the_expected_shape_in_both_modes(oracle_run, human_run):
    """Same columns and row count either way; the human cells are empty only when unmeasured."""
    oracle_table = pd.read_csv(oracle_run.out / "metrics" / "relevance_source_comparison.csv")
    human_table = pd.read_csv(human_run.out / "metrics" / "relevance_source_comparison.csv")

    for table in (oracle_table, human_table):
        assert list(table.columns) == RELEVANCE_COMPARISON_COLUMNS
        assert set(table["variant"]) == set(VARIANTS)
        assert set(table["metric"]) == set(RELEVANCE_METRICS)
        assert len(table) == len(VARIANTS) * len(RELEVANCE_METRICS)
        assert table["oracle"].notna().all()

    # Run 1 had no human labels beside its scenario file: nothing is imputed.
    assert oracle_table["human"].isna().all()
    assert oracle_table["delta"].isna().all()

    # Run 2 carries both columns, and each side equals the summary computed from that source.
    assert human_table["human"].notna().all()
    for row in human_table[human_table["variant"] == "full"].itertuples():
        assert row.oracle == pytest.approx(_variant_metric(oracle_run.out, "full", row.metric))
        assert row.human == pytest.approx(_variant_metric(human_run.out, "full", row.metric))
        assert row.delta == pytest.approx(row.human - row.oracle)


def test_analysis_plan_records_the_real_source_and_label_provenance(oracle_run, human_run):
    oracle_plan = yaml.safe_load(
        (oracle_run.out / "manifests" / "analysis_plan.yaml").read_text())
    human_plan = yaml.safe_load(
        (human_run.out / "manifests" / "analysis_plan.yaml").read_text())

    assert oracle_plan["relevance_source"] == RELEVANCE_SOURCE_ORACLE
    assert oracle_plan["human_relevance_labels"] is None

    assert human_plan["relevance_source"] == RELEVANCE_SOURCE_HUMAN
    assert human_plan["relevance_source_flag"] == "human"
    assert human_plan["relevance_metrics_from_source"] == RELEVANCE_METRICS
    # Retrieval recall deliberately stays on the full-catalog oracle universe.
    assert human_plan["retrieval_recall_relevance_source"] == RELEVANCE_SOURCE_ORACLE

    prov = human_plan["human_relevance_labels"]
    assert prov["path"] == str(human_run.labels)
    assert prov["sha256"] == hashlib.sha256(human_run.labels.read_bytes()).hexdigest()
    assert prov["graded_pairs"] > 0
    assert prov["adjudication_source"] == ADJUDICATION_COLUMN


def test_human_run_report_and_artifacts_state_the_human_source(human_run):
    md = (human_run.out / "report" / "analysis_report.md").read_text(encoding="utf-8")
    squeezed = " ".join(md.replace(">", " ").split())

    assert "No human raters were used" not in squeezed
    assert ("Relevance is scored by human relevance labels from two raters after "
            "adjudication, not by the automatic oracle.") in squeezed
    assert "### 5.5 Relevance source: automatic oracle vs adjudicated human labels" in md

    data = json.loads((human_run.out / "report" / "analysis_report_data.json").read_text(
        encoding="utf-8"))
    assert data["relevance_source"]["selected"] == RELEVANCE_SOURCE_HUMAN
    assert data["relevance_source"]["retrieval_labels"] == RELEVANCE_SOURCE_ORACLE

    # The consumed human table is persisted beside the oracle one, and both are checksummed.
    human_table = pd.read_csv(human_run.out / "normalized" / "relevance_labels_human.csv")
    assert list(human_table.columns) == RELEVANCE_LABEL_COLUMNS
    checksums = json.loads((human_run.out / "checksums.json").read_text())
    files = checksums.get("files", checksums)
    for artifact in ("normalized/relevance_labels_human.csv",
                     "metrics/relevance_source_comparison.csv"):
        assert artifact in files, artifact


def test_oracle_run_report_keeps_the_oracle_disclaimer(oracle_run):
    md = (oracle_run.out / "report" / "analysis_report.md").read_text(encoding="utf-8")
    squeezed = " ".join(md.replace(">", " ").split())

    assert "Relevance is scored by a deterministic automatic oracle, not human raters." in squeezed
    assert "No human raters were used in this run." in squeezed
    assert not (oracle_run.out / "normalized" / "relevance_labels_human.csv").exists()
