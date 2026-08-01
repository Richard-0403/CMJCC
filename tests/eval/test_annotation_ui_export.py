"""Export behaviour: slot columns, the four adjudication cases, validation and the archive.

The exported CSVs are a contract with :mod:`jobrec_eval.annotation`, so these tests pin the
exact cell values for all four states an item can be in:

============================  ==============================================================
case                          ``adjudicated``
============================  ==============================================================
raters agree                  empty (concordant raters are their own gold downstream)
disagreement, adjudicated     the recorded verdict
disagreement, unadjudicated   empty (reported as unadjudicated, never averaged)
one label missing             row absent, counted as incomplete
============================  ==============================================================

They also pin that validation happens BEFORE any file is written: a duplicated pair or a grade
outside 0-3 makes :func:`jobrec_eval.annotation.load_adjudicated_relevance_labels` raise at
analysis time, so emitting such a file would just move the failure later.

Every rater id and label here is synthetic (``SYNTHETIC-*``).
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from jobrec_eval.annotation import ADJUDICATED_COLUMN, load_adjudicated_relevance_labels
from jobrec_eval.annotation_ui.assignment import assign_two_raters
from jobrec_eval.annotation_ui.export import (
    CLAIM_COLUMNS,
    CLAIMS_CSV_FILENAME,
    DUMP_JSONL_FILENAME,
    MANIFEST_FILENAME,
    RELEVANCE_COLUMNS,
    RELEVANCE_CSV_FILENAME,
    ExportValidationError,
    export_annotations,
    validate_claim_rows,
    validate_relevance_rows,
)
from jobrec_eval.annotation_ui.store import (
    KIND_CLAIM,
    KIND_RELEVANCE,
    META_EXPERIMENT_ID,
    AnnotationItem,
    ClaimOccurrence,
    open_store,
)

RATER_A = "SYNTHETIC-RATER-A"
RATER_B = "SYNTHETIC-RATER-B"
ADJUDICATOR = "SYNTHETIC-ADJUDICATOR"
SEED = 7

AGREE = "rel::SYN-SC-01::SYN-job-01"
ADJUDICATED = "rel::SYN-SC-02::SYN-job-02"
OPEN_DISAGREEMENT = "rel::SYN-SC-03::SYN-job-03"
INCOMPLETE = "rel::SYN-SC-04::SYN-job-04"
CLAIM_SIGNATURE = "sig-SYN0000000000001"
CLAIM = f"clm::{CLAIM_SIGNATURE}"


def _relevance_item(item_key: str, index: int) -> AnnotationItem:
    return AnnotationItem(
        item_key=item_key, kind=KIND_RELEVANCE,
        payload={"scenario": {"scenario_id": f"SYN-SC-{index:02d}"},
                 "job": {"job_id": f"SYN-job-{index:02d}"}},
        analysis={"oracle_grade": 2},
        scenario_id=f"SYN-SC-{index:02d}", job_id=f"SYN-job-{index:02d}")


@pytest.fixture()
def store(tmp_path):
    """A synthetic store covering all four export cases plus a two-occurrence claim."""
    with open_store(tmp_path / "annotation") as store:
        store.register_raters([RATER_A, RATER_B])
        store.add_items([
            _relevance_item(AGREE, 1), _relevance_item(ADJUDICATED, 2),
            _relevance_item(OPEN_DISAGREEMENT, 3), _relevance_item(INCOMPLETE, 4),
            AnnotationItem(
                item_key=CLAIM, kind=KIND_CLAIM,
                # Schema v2: a claim item must name the proposition it stands for, and its
                # occurrences carry the batch, the repeat and whether the user saw them.
                annotation_signature=CLAIM_SIGNATURE,
                payload={"claim_text": "Synthetic claim.", "evidence": []},
                analysis={"occurrence_count": 2}, claim_id="SYN-claim-01",
                occurrences=(
                    ClaimOccurrence(run_id="SYN-run-1", claim_id="SYN-claim-01",
                                    variant="full", validator_label=1,
                                    support_status="supported",
                                    experiment_id="SYN-exp-1", repeat_index=0,
                                    annotation_signature=CLAIM_SIGNATURE,
                                    delivery_status="delivered"),
                    ClaimOccurrence(run_id="SYN-run-2", claim_id="SYN-claim-01",
                                    variant="no_memory", validator_label=0,
                                    support_status="unsupported",
                                    experiment_id="SYN-exp-1", repeat_index=0,
                                    annotation_signature=CLAIM_SIGNATURE,
                                    delivery_status="dropped"))),
        ])
        store.save_assignment_plan(
            assign_two_raters(store.item_keys(), [RATER_A, RATER_B], SEED))
        store.set_meta({META_EXPERIMENT_ID: "exp-SYNTHETIC"})

        label(store, AGREE, 2, 2)
        label(store, ADJUDICATED, 3, 1)
        label(store, OPEN_DISAGREEMENT, 0, 2)
        label(store, INCOMPLETE, 1, None)
        label(store, CLAIM, 1, 1)
        store.record_adjudication(ADJUDICATED, ADJUDICATOR, 2, reason="synthetic adjudication")
        yield store


def label(store, item_key: str, slot_1: int | None, slot_2: int | None) -> None:
    """Label an item by SLOT, so the test does not care which rater holds which slot."""
    record = next(r for r in store.iter_export_records() if r.item_key == item_key)
    for slot, value in ((1, slot_1), (2, slot_2)):
        if value is not None:
            store.upsert_annotation(item_key, record.slot_raters[slot], value,
                                    notes=f"synthetic note slot {slot}", duration_ms=1000 * slot)


def test_relevance_csv_has_the_contract_columns_and_slot_based_raters(store, tmp_path):
    result = export_annotations(store, tmp_path / "out")
    frame = pd.read_csv(result.relevance_path)

    assert result.relevance_path.name == RELEVANCE_CSV_FILENAME
    assert list(frame.columns) == RELEVANCE_COLUMNS
    # rater_1/rater_2 are SLOTS: the values follow the assignment slot, not a rater identity,
    # so the columns stay stable across items and pool sizes.
    by_scenario = frame.set_index("scenario_id")
    assert by_scenario.loc["SYN-SC-01", ["rater_1", "rater_2"]].tolist() == [2, 2]
    assert by_scenario.loc["SYN-SC-02", ["rater_1", "rater_2"]].tolist() == [3, 1]
    assert set(frame["scenario_id"]) == {"SYN-SC-01", "SYN-SC-02", "SYN-SC-03"}


def test_the_four_adjudication_cases(store, tmp_path):
    """Agree / adjudicated / unadjudicated / incomplete, each in its exact cell form."""
    result = export_annotations(store, tmp_path / "out")
    frame = pd.read_csv(result.relevance_path).set_index("scenario_id")
    adjudicated = frame[ADJUDICATED_COLUMN]

    # 1. raters agree -> EMPTY, because the consuming side uses their shared label as gold.
    assert pd.isna(adjudicated.loc["SYN-SC-01"])
    # 2. adjudicated disagreement -> the recorded verdict, never a derived one.
    assert int(adjudicated.loc["SYN-SC-02"]) == 2
    # 3. unadjudicated disagreement -> EMPTY, so it is reported as unadjudicated, not averaged.
    assert pd.isna(adjudicated.loc["SYN-SC-03"])
    assert int(frame.loc["SYN-SC-03", "rater_1"]) == 0
    assert int(frame.loc["SYN-SC-03", "rater_2"]) == 2
    # 4. incomplete -> no row at all, and counted.
    assert "SYN-SC-04" not in frame.index
    assert result.incomplete_count(KIND_RELEVANCE) == 1
    assert result.incomplete[0].item_key == INCOMPLETE
    assert result.incomplete[0].labelled_slots == (1,)
    assert result.incomplete[0].missing_raters
    assert result.manifest["counts"]["incomplete_relevance_items"] == 1
    assert result.manifest["counts"]["unadjudicated_disagreements"] == 1
    assert result.manifest["counts"]["adjudicated_disagreements"] == 1


def test_claim_rows_expand_per_occurrence_with_a_per_run_validator(store, tmp_path):
    """One judgement, one row per ``(run_id, claim_id)``, validator taken per run."""
    result = export_annotations(store, tmp_path / "out")
    frame = pd.read_csv(result.claims_path)

    assert result.claims_path.name == CLAIMS_CSV_FILENAME
    assert list(frame.columns) == CLAIM_COLUMNS
    assert len(frame) == 2
    assert set(frame["run_id"]) == {"SYN-run-1", "SYN-run-2"}
    assert set(frame["claim_id"]) == {"SYN-claim-01"}
    # The human labels replicate; the validator does not -- it ran per run.
    assert frame["rater_1"].tolist() == [1, 1]
    assert frame["rater_2"].tolist() == [1, 1]
    assert frame.set_index("run_id").loc["SYN-run-1", "validator"] == 1
    assert frame.set_index("run_id").loc["SYN-run-2", "validator"] == 0
    assert frame[ADJUDICATED_COLUMN].isna().all()


def test_notes_are_attributed_by_slot(store, tmp_path):
    result = export_annotations(store, tmp_path / "out")
    frame = pd.read_csv(result.relevance_path).set_index("scenario_id")

    assert "rater_1: synthetic note slot 1" in frame.loc["SYN-SC-01", "notes"]
    assert "rater_2: synthetic note slot 2" in frame.loc["SYN-SC-01", "notes"]
    assert "adjudication: synthetic adjudication" in frame.loc["SYN-SC-02", "notes"]


def test_manifest_records_seed_pool_counts_and_a_hash_per_csv(store, tmp_path):
    result = export_annotations(store, tmp_path / "out")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.manifest_path.name == MANIFEST_FILENAME
    assert manifest["experiment_id"] == "exp-SYNTHETIC"
    assert manifest["assignment_seed"] == str(SEED)
    assert manifest["rater_pool"] == [RATER_A, RATER_B]
    assert manifest["counts"]["relevance_items"] == 4
    assert manifest["counts"]["relevance_items_exported"] == 3
    assert manifest["counts"]["claim_rows_exported"] == 2
    for name in (RELEVANCE_CSV_FILENAME, CLAIMS_CSV_FILENAME):
        assert len(manifest["files"][name]["sha256"]) == 64
        assert manifest["files"][name]["sha256"] == result.hashes[name]
    assert manifest["annotation_effort"]["annotations"] == 9
    assert manifest["annotation_effort"]["total_duration_ms"] > 0


def test_the_jsonl_dump_is_append_only(store, tmp_path):
    """A second export adds a generation instead of erasing what was already shipped."""
    release = tmp_path / "final_release" / "human_annotations"
    first = export_annotations(store, tmp_path / "out", release_dir=release)
    lines_after_first = first.dump_path.read_text(encoding="utf-8").strip().splitlines()

    second = export_annotations(store, tmp_path / "out", release_dir=release)
    lines_after_second = second.dump_path.read_text(encoding="utf-8").strip().splitlines()

    assert first.dump_path.name == DUMP_JSONL_FILENAME
    assert first.dump_path.parent == release
    assert len(lines_after_second) == 2 * len(lines_after_first)
    header = json.loads(lines_after_first[0])
    assert header["record_type"] == "export_header"
    assert header["raters"] == [RATER_A, RATER_B]
    items = [json.loads(line) for line in lines_after_first if '"item"' in line]
    assert {i["item_key"] for i in items} == {AGREE, ADJUDICATED, OPEN_DISAGREEMENT,
                                             INCOMPLETE, CLAIM}
    # The archive carries the analysis side too: that is what makes the pass reproducible.
    assert any(i["analysis"].get("oracle_grade") is not None for i in items)
    assert any(i["adjudication"] for i in items)


def test_validation_refuses_an_out_of_range_grade_and_a_duplicate_pair():
    """Both failures are exactly what the consuming loader raises on."""
    with pytest.raises(ExportValidationError, match="outside"):
        validate_relevance_rows([{"scenario_id": "SYN-SC-01", "job_id": "SYN-job-01",
                                  "rater_1": 4, "rater_2": 2, ADJUDICATED_COLUMN: ""}])
    with pytest.raises(ExportValidationError, match="outside"):
        validate_relevance_rows([{"scenario_id": "SYN-SC-01", "job_id": "SYN-job-01",
                                  "rater_1": 1, "rater_2": 2, ADJUDICATED_COLUMN: 7}])
    with pytest.raises(ExportValidationError, match="once"):
        validate_relevance_rows([
            {"scenario_id": "SYN-SC-01", "job_id": "SYN-job-01", "rater_1": 1, "rater_2": 1,
             ADJUDICATED_COLUMN: ""},
            {"scenario_id": "SYN-SC-01", "job_id": "SYN-job-01", "rater_1": 2, "rater_2": 2,
             ADJUDICATED_COLUMN: ""}])

    def claim(**over):
        base = {"experiment_id": "SYN-exp-1", "run_id": "SYN-run-1",
                "claim_id": "SYN-claim-01", "annotation_signature": CLAIM_SIGNATURE,
                "delivery_status": "delivered", "rater_1": 1, "rater_2": 1, "validator": 1,
                ADJUDICATED_COLUMN: ""}
        base.update(over)
        return base

    with pytest.raises(ExportValidationError, match="outside"):
        validate_claim_rows([claim(rater_1=2)])
    with pytest.raises(ExportValidationError, match="once"):
        validate_claim_rows([claim(), claim(rater_1=0, rater_2=0)])
    # A row that cannot be tied to a batch, a proposition or a delivery state is refused: this
    # is the shape every pre-migration file has, and reporting it against these runs is the
    # defect the columns exist to prevent.
    for missing in ("experiment_id", "annotation_signature", "delivery_status"):
        with pytest.raises(ExportValidationError, match="needs"):
            validate_claim_rows([claim(**{missing: ""})])
    # Two propositions under one claim_id in one run are NOT a duplicate; the old
    # (run_id, claim_id) key refused to write any file at all when that happened.
    validate_claim_rows([claim(), claim(annotation_signature="sig-SYN0000000000002",
                                       rater_1=0, rater_2=0)])


def test_a_file_the_consuming_loader_would_reject_is_never_written(tmp_path):
    """Why the guard exists: such a file makes the analysis-time loader raise."""
    bad = tmp_path / RELEVANCE_CSV_FILENAME
    pd.DataFrame([{"scenario_id": "SYN-SC-01", "job_id": "SYN-job-01", "rater_1": 4,
                   "rater_2": 4, ADJUDICATED_COLUMN: ""}]).to_csv(bad, index=False)
    with pytest.raises(ValueError, match="grades must be"):
        load_adjudicated_relevance_labels(bad)


def test_exported_filenames_match_what_the_pipeline_looks_for():
    """The CLI reads these exact names beside ``--scenarios``; drift would silently skip them."""
    from jobrec_eval.cli import HUMAN_CLAIMS_FILENAME, HUMAN_RELEVANCE_FILENAME

    assert RELEVANCE_CSV_FILENAME == HUMAN_RELEVANCE_FILENAME
    assert CLAIMS_CSV_FILENAME == HUMAN_CLAIMS_FILENAME
