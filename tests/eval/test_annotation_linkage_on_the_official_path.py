"""The linkage guard must be on the pipeline, not merely available to it.

``claim_agreement`` and ``relevance_coverage`` can both refuse bad annotations, but a guard
that the official analysis does not call protects nothing. These tests run the real pipeline
and assert the artifacts it produces, so the wiring cannot be removed without a failure.

What is checked:

* ``normalized/claim_occurrences.csv`` exists and carries every occurrence field, including
  dropped claims with an explicit ``delivery_status``.
* ``manifests/annotation_coverage.json`` records the claim linkage and the relevance coverage,
  with the delta pairs listed rather than folded into a metric as zeros.
* A returned pair with no human label appears as a delta, never as a scored zero.
* Stale human claim labels make the analysis FAIL rather than publish a kappa.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from jobrec_eval.annotation_linkage import (
    DELIVERY_DELIVERED,
    OCCURRENCE_FIELDS,
    StaleAnnotationError,
)
from jobrec_eval.cli import run_pipeline

CATALOG = "data/processed/jobs.jsonl"
CONFIG = "configs/experiment_full.yaml"


def _tiny_scenarios(root: Path) -> Path:
    """Two scenarios, one of which returns jobs, so there is something to cover."""
    rows = [json.loads(line) for line
            in Path("evaluation/data/scenarios.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    keep = [r for r in rows if r["scenario_id"] in ("SC-A-01", "SC-E-02")]
    assert len(keep) == 2
    root.mkdir(parents=True, exist_ok=True)
    path = root / "scenarios.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in keep:
            fh.write(json.dumps(row, default=str) + "\n")
    return path


@pytest.fixture(scope="module")
def analysed(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("linkage-pipeline")
    scenarios = _tiny_scenarios(root / "inputs")
    result = run_pipeline(CONFIG, str(scenarios), CATALOG, str(root / "out"),
                          repeats=1, experiment_dir=None, bootstrap_iters=10,
                          bootstrap_seed=1, variants=["full"])
    return Path(result["output_dir"] if isinstance(result, dict) and "output_dir" in result
                else next((root / "out").glob("exp-*")))


def test_the_pipeline_writes_the_claim_occurrence_table(analysed: Path):
    path = analysed / "normalized" / "claim_occurrences.csv"
    assert path.exists(), "the official path did not record claim occurrences"

    frame = pd.read_csv(path)
    for column in OCCURRENCE_FIELDS:
        assert column in frame.columns, column
    assert not frame.empty
    # Every occurrence names which experiment and run it came from.
    assert frame["experiment_id"].nunique() == 1
    assert frame["run_id"].notna().all()
    assert frame["annotation_signature"].notna().all()
    assert set(frame["delivery_status"]) <= {DELIVERY_DELIVERED, "dropped"}
    assert DELIVERY_DELIVERED in set(frame["delivery_status"])


def test_signatures_separate_propositions_within_one_real_experiment(analysed: Path):
    """Distinct propositions must not collapse onto one signature in real data."""
    frame = pd.read_csv(analysed / "normalized" / "claim_occurrences.csv")
    per_signature = frame.groupby("annotation_signature")["claim_type"].nunique()
    assert (per_signature == 1).all(), (
        "a signature spans several claim types, so it is merging propositions")


def test_the_pipeline_writes_the_coverage_report(analysed: Path):
    path = analysed / "manifests" / "annotation_coverage.json"
    assert path.exists(), "the official path did not record annotation coverage"

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["experiment_id"]
    coverage = payload["relevance_coverage"]
    for key in ("returned_pairs", "reused_overlapping_labels",
                "delta_pairs_requiring_annotation", "obsolete_extra_labels", "coverage"):
        assert key in coverage, key


def test_unlabelled_returned_pairs_are_a_delta_and_not_scored_zero(analysed: Path):
    """No human labels exist for this tmp experiment, so every returned pair is a delta."""
    payload = json.loads(
        (analysed / "manifests" / "annotation_coverage.json").read_text(encoding="utf-8"))
    coverage = payload["relevance_coverage"]

    assert payload["human_relevance_labels_present"] is False
    assert coverage["reused_overlapping_labels"] == 0
    assert coverage["delta_pairs_requiring_annotation"] == coverage["returned_pairs"]
    # Coverage is 0.0 with returned pairs present -- reported, not silently treated as
    # "all irrelevant".
    assert coverage["coverage"] == 0.0
    assert coverage["delta_sample"], "the pairs needing annotation are not listed"


def test_stale_claim_labels_fail_the_analysis(tmp_path):
    """The guard has to bite on the official path, not just in isolation."""
    scenarios = _tiny_scenarios(tmp_path / "inputs")
    # Human claim labels for an experiment that never produced these signatures.
    pd.DataFrame([{"experiment_id": "exp-somethingelse",
                   "annotation_signature": "sig-nevergenerated",
                   "claim_id": "c1", "validator": 1, "rater_1": 1, "rater_2": 0}]
                 ).to_csv(scenarios.parent / "claim_annotations_human.csv", index=False)

    with pytest.raises(StaleAnnotationError, match="not a measurement"):
        run_pipeline(CONFIG, str(scenarios), CATALOG, str(tmp_path / "out"),
                     repeats=1, experiment_dir=None, bootstrap_iters=10,
                     bootstrap_seed=1, variants=["full"])
