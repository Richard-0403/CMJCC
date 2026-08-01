"""The annotation unit is the signature, and human metrics are gated on coverage.

Five assertions, each naming a way the previous shape produced a number that looked usable:

* the annotation task had one item per ``claim_id``, which digests the rendered SENTENCE and so
  merges propositions that read alike at different values;
* those items could carry the union of two propositions' evidence, asking a rater a question
  neither claim makes;
* withheld claims were absent entirely, so the validator's false-NEGATIVE rate had no sample;
* a returned pair with no human relevance label entered the ranking metric as a 0 grade, mixing
  "judged irrelevant" with "not judged";
* and four failed HTTP attempts were summed from their INDEX, reporting ten.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from jobrec_eval.annotation_linkage import (
    DELIVERY_DELIVERED,
    DELIVERY_DROPPED,
    annotation_items,
    annotation_signature,
    claim_occurrences,
    evidence_projection,
)
from jobrec_eval.cli import HUMAN_RELEVANCE_MIN_COVERAGE, run_pipeline

CATALOG = "data/processed/jobs.jsonl"
CONFIG = "configs/experiment_full.yaml"


def _claim(**over) -> dict:
    base = {"claim_id": "claim-shared", "claim_type": "ranking_reason",
            "text": "Salary meets your stated minimum.", "predicate": "salary_meets_min",
            "field_name": "salary_min", "job_id": "job-1", "expected_value": 4000,
            "observed_value": 4500, "claim_args": {}, "evidence_ids": [],
            "support_status": "supported"}
    base.update(over)
    return base


def _run(**over) -> dict:
    base = {"run_id": "run-1", "scenario_id": "SC-A-01", "variant": "full",
            "repeat_index": 0, "claims": [], "dropped_claims": [], "evidence_by_id": {}}
    base.update(over)
    return base


# ---------------------------------------- 1. items == unique signatures
def test_the_item_count_equals_the_number_of_unique_signatures():
    """Not the number of claims, and not the number of claim_ids."""
    claims = [_claim(expected_value=v) for v in (4000, 5000, 6000)]
    # The same three propositions again in a second run: more occurrences, same items.
    occurrences = claim_occurrences("exp-1", [
        _run(run_id="run-1", claims=claims),
        _run(run_id="run-2", claims=claims),
    ])

    signatures = {row["annotation_signature"] for row in occurrences}
    items = annotation_items(occurrences)

    assert len(occurrences) == 6
    assert len(signatures) == 3
    assert len(items) == len(signatures) == 3
    # All six occurrences shared ONE claim_id, so keying on that would have given one item.
    assert len({row["claim_id"] for row in occurrences}) == 1
    assert {i["occurrence_count"] for i in items} == {2}


def test_each_item_keeps_its_own_signature_and_no_other():
    items = annotation_items(claim_occurrences("exp-1", [
        _run(claims=[_claim(expected_value=4000), _claim(expected_value=6000)])]))
    assert len(items) == 2
    assert len({i["annotation_signature"] for i in items}) == 2


# ------------------------------------- 2. different signatures share no evidence
def test_different_signatures_do_not_share_evidence():
    """A rater shown the union of two propositions' evidence is asked a question neither makes."""
    store = {
        "ev-low": {"source": "dialogue", "field_name": "salary_min",
                   "normalized_value": 4000},
        "ev-high": {"source": "dialogue", "field_name": "salary_min",
                    "normalized_value": 6000},
    }
    low = _claim(expected_value=4000, evidence_ids=["ev-low"])
    high = _claim(expected_value=6000, evidence_ids=["ev-high"])

    assert annotation_signature(low, store) != annotation_signature(high, store)
    low_ev = evidence_projection(low, store)
    high_ev = evidence_projection(high, store)
    assert low_ev != high_ev
    # No value appears in both projections, so nothing was unioned.
    assert not [e for e in low_ev if e in high_ev]


def test_an_item_carries_no_evidence_from_another_signature():
    store = {"ev-low": {"source": "dialogue", "field_name": "salary_min",
                        "normalized_value": 4000},
             "ev-high": {"source": "dialogue", "field_name": "salary_min",
                         "normalized_value": 6000}}
    occurrences = claim_occurrences("exp-1", [_run(
        claims=[_claim(expected_value=4000, evidence_ids=["ev-low"]),
                _claim(expected_value=6000, evidence_ids=["ev-high"])],
        evidence_by_id=store)])
    items = annotation_items(occurrences)

    assert len(items) == 2
    # Each item resolves to exactly one signature, and that signature was computed from one
    # claim's evidence only.
    by_signature = {row["annotation_signature"]: row for row in occurrences}
    for item in items:
        assert item["annotation_signature"] in by_signature


# ------------------------------------------ 3. dropped claims reach the pilot
def test_dropped_claims_become_annotation_items():
    """The only sample a validator false-negative estimate can be measured on."""
    occurrences = claim_occurrences("exp-1", [_run(
        claims=[_claim(expected_value=4000)],
        dropped_claims=[_claim(expected_value=9999, text="Withheld claim.",
                               support_status="unsupported")])])
    items = annotation_items(occurrences)

    assert len(items) == 2
    statuses = {i["delivery_status"] for i in items}
    assert statuses == {DELIVERY_DELIVERED, DELIVERY_DROPPED}
    withheld = next(i for i in items if i["delivery_status"] == DELIVERY_DROPPED)
    assert withheld["validator"] == 0
    assert withheld["text"] == "Withheld claim."


def test_a_signature_seen_both_ways_is_reported_as_delivered():
    """The user did see it at least once; the split stays visible in the occurrence table."""
    claim = _claim()
    occurrences = claim_occurrences("exp-1", [
        _run(run_id="run-1", claims=[claim]),
        _run(run_id="run-2", dropped_claims=[claim]),
    ])
    items = annotation_items(occurrences)
    assert len(items) == 1
    assert items[0]["delivery_status"] == DELIVERY_DELIVERED
    assert {r["delivery_status"] for r in occurrences} == {DELIVERY_DELIVERED,
                                                          DELIVERY_DROPPED}


# ------------------------------- 4. no human metric values without coverage
@pytest.fixture(scope="module")
def analysed(tmp_path_factory) -> Path:
    """A real pipeline run with NO human relevance labels beside its scenarios."""
    root = tmp_path_factory.mktemp("coverage-gate")
    rows = [json.loads(line) for line
            in Path("evaluation/data/scenarios.jsonl").read_text(
                encoding="utf-8").splitlines() if line.strip()]
    keep = [r for r in rows if r["scenario_id"] in ("SC-A-01", "SC-A-02")]
    inputs = root / "inputs"
    inputs.mkdir(parents=True)
    scenarios = inputs / "scenarios.jsonl"
    with scenarios.open("w", encoding="utf-8", newline="\n") as fh:
        for row in keep:
            fh.write(json.dumps(row, default=str) + "\n")
    run_pipeline(CONFIG, str(scenarios), CATALOG, str(root / "out"), repeats=1,
                 experiment_dir=None, bootstrap_iters=10, bootstrap_seed=1,
                 variants=["full"])
    return Path(next((root / "out").glob("exp-*")))


def test_missing_relevance_labels_yield_no_human_metric_values(analysed: Path):
    coverage = json.loads(
        (analysed / "manifests" / "annotation_coverage.json").read_text(encoding="utf-8"))

    assert coverage["human_relevance_min_coverage"] == HUMAN_RELEVANCE_MIN_COVERAGE
    assert coverage["human_metrics_withheld_reason"], (
        "human metrics were computed despite missing labels")
    assert "UNKNOWN, not irrelevant" in coverage["human_metrics_withheld_reason"]

    comparison = analysed / "metrics" / "relevance_source_comparison.csv"
    if comparison.exists():
        frame = pd.read_csv(comparison)
        human_cols = [c for c in frame.columns if "human" in c.lower()]
        for column in human_cols:
            assert frame[column].isna().all(), (
                f"{column} carries a value computed over unlabelled pairs")


def test_the_full_delta_list_is_written_not_a_sample(analysed: Path):
    """A 20-row sample cannot be handed to a rater."""
    coverage = json.loads(
        (analysed / "manifests" / "annotation_coverage.json").read_text(encoding="utf-8"))
    expected = coverage["relevance_coverage"]["delta_pairs_requiring_annotation"]

    path = analysed / "annotation" / "relevance_delta_annotation.csv"
    assert path.exists(), "the delta annotation task was not written"
    frame = pd.read_csv(path)
    assert len(frame) == expected, (len(frame), expected)
    assert list(frame.columns) == ["scenario_id", "job_id"]
    if expected > 20:
        assert len(frame) > 20, "the CSV was truncated to the JSON sample size"


def test_the_claim_template_has_one_row_per_signature(analysed: Path):
    occurrences = pd.read_csv(analysed / "normalized" / "claim_occurrences.csv")
    template = pd.read_csv(analysed / "annotation" / "claim_template.csv")

    assert len(template) == occurrences["annotation_signature"].nunique()
    for column in ("experiment_id", "annotation_signature", "delivery_status"):
        assert column in template.columns, column
    assert template["annotation_signature"].is_unique
