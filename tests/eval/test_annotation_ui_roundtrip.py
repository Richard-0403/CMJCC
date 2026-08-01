"""End-to-end round trip on a REAL experiment: items -> store -> export -> agreement.

This is the test that proves the annotation tool closes the loop the thesis needs (checklist
items 10/11): items are built from real run bundles, two raters' answers and one adjudication
are recorded through the store API, the CSVs are exported, and the exported files are then read
back by the code that will actually consume them --
:func:`jobrec_eval.annotation.load_adjudicated_relevance_labels`,
:func:`~jobrec_eval.annotation.relevance_agreement` and
:func:`~jobrec_eval.annotation.claim_agreement`. If any column, spelling or adjudication rule
drifted, one of those three would report ``None``, raise, or fall back to the legacy
rounded-mean path; all three are asserted to take the ``adjudicated_column`` path instead.

THE LABELS HERE ARE SYNTHETIC AND SAID SO: rater ids are ``SYNTHETIC-RATER-*`` and every grade
comes from a fixed arithmetic pattern over the item index. They exist to exercise the plumbing
and must never be read as collected human judgements. Real labels arrive only through the UI.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from jobrec_eval.annotation import (
    ADJUDICATION_COLUMN,
    HUMAN_RATER_ID,
    RELEVANCE_LABEL_COLUMNS,
    claim_agreement,
    load_adjudicated_relevance_labels,
    relevance_agreement,
)
from jobrec_eval.annotation_linkage import DEFAULT_SAMPLING_SEED, DROPPED_STRATA_FIELDS
from jobrec_eval.annotation_ui import console
from jobrec_eval.annotation_ui.assignment import assign_two_raters
from jobrec_eval.annotation_ui.export import (
    CLAIMS_CSV_FILENAME,
    RELEVANCE_CSV_FILENAME,
    export_annotations,
)
from jobrec_eval.annotation_ui.loader import build_items
from jobrec_eval.annotation_ui.store import (
    KIND_CLAIM,
    KIND_RELEVANCE,
    META_ANNOTATION_UNIVERSE,
    META_EXPERIMENT_ID,
    META_SAMPLING_SEED,
    open_store,
)

RATER_POOL = ["SYNTHETIC-RATER-01", "SYNTHETIC-RATER-02", "SYNTHETIC-RATER-03"]
ADJUDICATOR = "SYNTHETIC-ADJUDICATOR"
SEED = 2026

#: Items deliberately left half-labelled, to prove incomplete items are excluded and counted.
LEFT_INCOMPLETE = 2


def _synthetic_relevance_labels(index: int) -> tuple[int, int]:
    """Obviously-synthetic 0-3 grade pair; every third item disagrees by one grade."""
    first = index % 4
    return (first, first if index % 3 else (first + 1) % 4)


def _synthetic_claim_labels(index: int) -> tuple[int, int]:
    """Obviously-synthetic {0,1} pair; every fourth item disagrees."""
    first = 1 if index % 5 else 0
    return (first, first if index % 4 else 1 - first)


@pytest.fixture(scope="module")
def annotated(annotation_experiment, tmp_path_factory):
    """A real-item store filled with synthetic labels, one adjudication pass applied."""
    directory = tmp_path_factory.mktemp("annotation-roundtrip")
    oracle = _synthetic_oracle(annotation_experiment)
    built = build_items(annotation_experiment.experiment_dir,
                        annotation_experiment.scenarios_path,
                        annotation_experiment.catalog_path, oracle_labels=oracle)
    store = open_store(directory / "annotation")
    store.register_raters(RATER_POOL)
    store.add_items(built.all_items)
    store.save_assignment_plan(assign_two_raters(store.item_keys(), RATER_POOL, SEED))
    store.set_meta({META_EXPERIMENT_ID: annotation_experiment.experiment_dir.name})

    records = list(store.iter_export_records())
    relevance = [r for r in records if r.kind == KIND_RELEVANCE]
    claims = [r for r in records if r.kind == KIND_CLAIM]

    for index, record in enumerate(relevance):
        first, second = _synthetic_relevance_labels(index)
        store.upsert_annotation(record.item_key, record.slot_raters[1], first,
                                notes="synthetic", duration_ms=8000)
        if index >= len(relevance) - LEFT_INCOMPLETE:
            continue  # left half-labelled on purpose
        store.upsert_annotation(record.item_key, record.slot_raters[2], second,
                                duration_ms=12000)
    for index, record in enumerate(claims):
        first, second = _synthetic_claim_labels(index)
        store.upsert_annotation(record.item_key, record.slot_raters[1], first, duration_ms=5000)
        store.upsert_annotation(record.item_key, record.slot_raters[2], second, duration_ms=6000)

    # Adjudicate every OTHER disagreement, so both the adjudicated and the unadjudicated
    # export cases are exercised on real items.
    open_cases = store.disagreements(unadjudicated_only=True)
    assert open_cases, "the synthetic label pattern produced no disagreement to adjudicate"
    for position, case in enumerate(open_cases):
        if position % 2:
            continue
        store.record_adjudication(case.item_key, ADJUDICATOR, case.slot_1_label,
                                  reason="synthetic adjudication")
    try:
        yield store, built, oracle, directory
    finally:
        store.close()


def _synthetic_oracle(annotation_experiment) -> pd.DataFrame:
    """A synthetic automatic-oracle table over the returned pairs, for the comparison side."""
    from jobrec_eval.loaders import normalize
    pairs = normalize(annotation_experiment.bundles)["recommendations"][
        ["scenario_id", "job_id"]].drop_duplicates().reset_index(drop=True)
    return pd.DataFrame({
        "scenario_id": pairs["scenario_id"].astype(str),
        "job_id": pairs["job_id"].astype(str),
        "rater_id": "SYNTHETIC-ORACLE",
        "relevance_grade": [(index * 2) % 4 for index in range(len(pairs))],
    })


@pytest.fixture(scope="module")
def exported(annotated, tmp_path_factory):
    store, built, oracle, _ = annotated
    out = tmp_path_factory.mktemp("annotation-export")
    result = export_annotations(store, out, release_dir=out / "human_annotations")
    return result, store, built, oracle


def test_export_writes_both_csvs_over_real_items(exported):
    result, store, built, _ = exported
    relevance = pd.read_csv(result.relevance_path)
    claims = pd.read_csv(result.claims_path)

    assert result.relevance_path.name == RELEVANCE_CSV_FILENAME
    assert result.claims_path.name == CLAIMS_CSV_FILENAME
    # Every complete relevance item, minus the ones deliberately left half-labelled.
    assert len(relevance) == built.stats.relevance_items - LEFT_INCOMPLETE
    assert result.incomplete_count(KIND_RELEVANCE) == LEFT_INCOMPLETE
    assert result.manifest["counts"]["incomplete_relevance_items"] == LEFT_INCOMPLETE
    # Claims expand back to one row per (run_id, claim_id) occurrence.
    assert len(claims) == built.stats.claim_occurrences
    # Keyed on the SIGNATURE, not on claim_id: several propositions share one claim_id, so
    # nunique() over claim_id undercounts the judgements (98 against 146 on this fixture).
    assert claims["annotation_signature"].nunique() == built.stats.claim_items
    assert claims["claim_id"].nunique() < claims["annotation_signature"].nunique()
    assert len(claims) > claims["annotation_signature"].nunique()
    assert not claims.duplicated(subset=["run_id", "claim_id", "annotation_signature"]).any()
    # The linkage columns a returned file needs to be tied back to this batch.
    for column in ("experiment_id", "annotation_signature", "delivery_status"):
        assert column in claims.columns, column
    assert (claims["experiment_id"] != "").all()
    assert set(claims["delivery_status"]) <= {"delivered", "dropped"}
    assert not relevance.duplicated(subset=["scenario_id", "job_id"]).any()


def test_the_exported_relevance_file_loads_as_an_adjudicated_label_table(exported):
    """The pipeline's ``--relevance-source human`` path reads this file."""
    result, store, _, _ = exported
    loaded = load_adjudicated_relevance_labels(result.relevance_path)

    assert loaded is not None
    assert list(loaded.labels.columns) == RELEVANCE_LABEL_COLUMNS
    assert set(loaded.labels["rater_id"]) == {HUMAN_RATER_ID}
    assert loaded.labels["relevance_grade"].between(0, 3).all()
    assert not loaded.labels.duplicated(subset=["scenario_id", "job_id"]).any()

    provenance = loaded.provenance
    assert provenance["adjudication_source"] == ADJUDICATION_COLUMN
    # Concordant + adjudicated rows become gold; unadjudicated disagreements are dropped and
    # counted, never averaged.
    assert provenance["graded_pairs"] == (provenance["rater_concordant_pairs"]
                                          + provenance["adjudicated_pairs"])
    assert provenance["unadjudicated_disagreements_dropped"] > 0
    assert provenance["adjudicated_pairs"] > 0


def test_relevance_agreement_parses_the_export_and_reports_the_adjudicated_source(exported):
    result, _, _, oracle = exported
    agreement = relevance_agreement(result.relevance_path, oracle)

    assert agreement is not None
    assert agreement["adjudication_source"] == ADJUDICATION_COLUMN
    assert agreement["n_items"] > 0
    assert agreement["weighted_kappa_raters"] is not None
    assert agreement["oracle_vs_human_weighted_kappa"] is not None
    assert agreement["unadjudicated_disagreements"] > 0
    assert agreement["n_adjudicated"] > 0


def test_claim_agreement_parses_the_export_and_reports_the_adjudicated_source(exported):
    result, _, _, _ = exported
    agreement = claim_agreement(result.claims_path)

    assert agreement is not None
    assert agreement["adjudication_source"] == ADJUDICATION_COLUMN
    assert agreement["n_items"] > 0
    assert agreement["cohens_kappa"] is not None
    # The validator column survived the round trip, so validator-vs-human is computable.
    assert "validator_vs_human_kappa" in agreement


def test_the_release_dump_and_manifest_describe_the_real_pass(exported):
    result, store, built, _ = exported
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    dump_lines = result.dump_path.read_text(encoding="utf-8").strip().splitlines()

    assert result.dump_path.parent.name == "human_annotations"
    assert manifest["rater_pool"] == RATER_POOL
    assert manifest["assignment_seed"] == str(SEED)
    assert manifest["counts"]["relevance_items"] == built.stats.relevance_items
    assert manifest["counts"]["claim_items"] == built.stats.claim_items
    assert manifest["counts"]["claim_rows_exported"] == built.stats.claim_occurrences
    assert manifest["annotation_effort"]["total_duration_ms"] > 0
    # One header plus one record per item.
    assert len(dump_lines) == 1 + len(built.all_items)
    assert store.assignment_counts() == manifest["assignment_counts"]
    assert max(manifest["assignment_counts"].values()) - min(
        manifest["assignment_counts"].values()) <= 1


def test_the_console_builds_and_exports_from_the_real_experiment(annotation_experiment,
                                                                tmp_path, capsys):
    """The CLI entry point covers build -> status -> export without any web layer."""
    annotation_dir = tmp_path / "annotation"
    assert console.main([
        "build", "--experiment-dir", str(annotation_experiment.experiment_dir),
        "--scenarios", annotation_experiment.scenarios_path,
        "--catalog", annotation_experiment.catalog_path,
        "--annotation-dir", str(annotation_dir),
        "--raters", ",".join(RATER_POOL), "--seed", str(SEED),
    ]) == 0
    build_output = json.loads(capsys.readouterr().out)
    assert build_output["stats"]["claim_items"] > 0
    assert build_output["max_load_imbalance"] <= 1
    assert build_output["raters"] == RATER_POOL

    # The pre-registered frame is written BEFORE any label exists, and the store records where
    # it is and which seed drew it. Coverage is measured against this file, so a frame nobody
    # can locate or reproduce would make the denominator unverifiable.
    universe_path = annotation_dir / console.UNIVERSE_FILENAME
    assert universe_path.is_file()
    universe = json.loads(universe_path.read_text(encoding="utf-8"))
    assert universe["sampling_seed"] == DEFAULT_SAMPLING_SEED
    assert universe["strata_fields"] == list(DROPPED_STRATA_FIELDS)
    assert universe["counts"]["delivered_signatures"] == build_output["stats"]["claim_items"] - (
        build_output["stats"]["dropped_claim_items"])
    assert len(universe["delivered"]) == universe["counts"]["delivered_signatures"]
    with open_store(annotation_dir, create=False) as store:
        meta = store.meta()
    assert meta[META_ANNOTATION_UNIVERSE] == console.UNIVERSE_FILENAME
    assert meta[META_SAMPLING_SEED] == str(DEFAULT_SAMPLING_SEED)
    # Rebuilding from the same bundles reproduces the frame exactly.
    assert console.main([
        "build", "--experiment-dir", str(annotation_experiment.experiment_dir),
        "--scenarios", annotation_experiment.scenarios_path,
        "--catalog", annotation_experiment.catalog_path,
        "--annotation-dir", str(annotation_dir),
        "--raters", ",".join(RATER_POOL), "--seed", str(SEED),
    ]) == 0
    rebuilt = json.loads(capsys.readouterr().out)
    assert rebuilt["stats"] == build_output["stats"]
    again = json.loads(universe_path.read_text(encoding="utf-8"))
    assert {k: v for k, v in again.items() if k != "created_at"} == {
        k: v for k, v in universe.items() if k != "created_at"}

    assert console.main(["status", "--annotation-dir", str(annotation_dir)]) == 0
    status = json.loads(capsys.readouterr().out)
    assert status["items"][KIND_RELEVANCE] == build_output["stats"]["relevance_items"]
    assert status["both_slots_complete"] == 0

    # Nothing labelled yet: the export writes header-only CSVs and counts every item as
    # incomplete rather than emitting blank label rows.
    assert console.main([
        "export", "--annotation-dir", str(annotation_dir), "--out-dir", str(tmp_path / "csv"),
    ]) == 0
    export_output = json.loads(capsys.readouterr().out)
    assert export_output["row_counts"][RELEVANCE_CSV_FILENAME] == 0
    assert export_output["skipped_incomplete"][KIND_RELEVANCE] == status["items"][KIND_RELEVANCE]
    assert export_output["skipped_incomplete"][KIND_CLAIM] == status["items"][KIND_CLAIM]
    assert pd.read_csv(tmp_path / "csv" / RELEVANCE_CSV_FILENAME).empty
