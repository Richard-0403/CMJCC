"""The blank workload a rater actually receives, and the path it comes back on.

A workload file is the one artifact that leaves the repo and goes to a person, so two
properties are asserted rather than assumed:

* it carries no blinded field. It is rendered from ``AnnotationStore.queue``, whose
  ``RaterItem`` has no field for the analysis side at all, so a leak would mean the payload
  itself was contaminated -- but the file is checked again because an unblinded workload
  silently converts an independent human judgement into agreement with the system under test.
* a blank label is SKIPPED on import, never written as 0. "Nobody judged this" and "judged
  irrelevant / unsupported" must not become the same value, which is the same defect the
  relevance coverage gate exists for one layer up.

One file per rater, each in that rater's own seeded shuffle order: two raters must not receive
the items in the same sequence, or fatigue and drift line up between them and inflate kappa.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from scripts.build_annotation_workload import LABEL_COLUMN, export, import_labels

from jobrec_eval.annotation_ui.assignment import assign_two_raters
from jobrec_eval.annotation_ui.loader import build_items
from jobrec_eval.annotation_ui.store import BLINDED_FIELD_NAMES, open_store

RATERS = ("SYNTHETIC-R1", "SYNTHETIC-R2")


@pytest.fixture(scope="module")
def workload(annotation_experiment, tmp_path_factory):
    """A real store built from the real bundles, exported to blank workload files."""
    root = tmp_path_factory.mktemp("workload")
    store_dir, out_dir = root / "store", root / "files"
    built = build_items(annotation_experiment.experiment_dir,
                        annotation_experiment.scenarios_path,
                        annotation_experiment.catalog_path)
    with open_store(store_dir) as store:
        store.register_raters(RATERS)
        store.add_items(built.all_items)
        store.save_assignment_plan(
            assign_two_raters(store.item_keys(), RATERS, seed=2026))
    export(store_dir, out_dir, None, write=True)
    return store_dir, out_dir


def _read(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_each_rater_gets_their_own_file_in_their_own_order(workload):
    _store_dir, out = workload
    a = _read(out / f"claim_{RATERS[0]}.csv")
    b = _read(out / f"claim_{RATERS[1]}.csv")

    assert a and b
    keys_a = [r["item_key"] for r in a]
    keys_b = [r["item_key"] for r in b]
    assert set(keys_a) == set(keys_b), "the two raters must judge the SAME items"
    assert keys_a != keys_b, (
        "both raters got the same order; a shared ordering lets fatigue line up and inflate "
        "agreement")
    # No file carries the other rater's answer, or any answer at all.
    for rows in (a, b):
        assert all(not r[LABEL_COLUMN] for r in rows)
        assert "rater_1" not in rows[0] and "rater_2" not in rows[0]


def test_no_workload_file_carries_a_blinded_field(workload):
    _store_dir, out = workload
    for path in sorted(out.glob("*.csv")):
        blob = json.dumps(_read(path), ensure_ascii=False).lower()
        for blinded in BLINDED_FIELD_NAMES:
            assert f'"{blinded}"' not in blob, f"{path.name} leaks {blinded}"


def test_a_claim_row_carries_what_the_judgement_needs(workload):
    _store_dir, out = workload
    rows = _read(out / f"claim_{RATERS[0]}.csv")
    row = next(r for r in rows if r["evidence"])

    for column in ("claim_text", "predicate", "field", "expected_value", "observed_value",
                   "delivery_status", "evidence", "has_unresolvable_evidence"):
        assert column in row, column
    assert row["claim_text"]
    # The evidence is rendered readably rather than as an opaque id list: a rater judging
    # "does this support the sentence" has to see what the citation SAYS.
    assert "field=" in row["evidence"] and "value=" in row["evidence"]


def test_a_blank_label_is_skipped_and_never_imported_as_zero(workload, tmp_path):
    store_dir, out = workload
    import shutil

    scratch = tmp_path / "scratch"
    shutil.copytree(store_dir, scratch)
    files = tmp_path / "files"
    shutil.copytree(out, files)

    # Fill exactly one row; leave every other blank.
    path = files / f"claim_{RATERS[0]}.csv"
    rows = _read(path)
    columns = list(rows[0])
    rows[0][LABEL_COLUMN] = "1"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    import_labels(scratch, files, write=True)

    with open_store(scratch, create=False) as store:
        saved = store.annotation(rows[0]["item_key"], RATERS[0])
        assert saved is not None and saved.label == 1
        total = store._db.execute("SELECT COUNT(*) n FROM annotations").fetchone()["n"]
    assert total == 1, f"{total} labels written from one filled row; blanks became values"


def test_an_out_of_range_or_non_numeric_label_is_refused(workload, tmp_path):
    store_dir, out = workload
    import shutil

    scratch = tmp_path / "scratch2"
    shutil.copytree(store_dir, scratch)
    files = tmp_path / "files2"
    shutil.copytree(out, files)

    path = files / f"claim_{RATERS[0]}.csv"
    rows = _read(path)
    columns = list(rows[0])
    rows[0][LABEL_COLUMN] = "7"      # claim labels are {0, 1}
    rows[1][LABEL_COLUMN] = "maybe"
    rows[2][LABEL_COLUMN] = "0"      # the only valid one
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)

    status = import_labels(scratch, files, write=True)
    assert status == 1, "invalid rows must be reported through the exit code"

    with open_store(scratch, create=False) as store:
        total = store._db.execute("SELECT COUNT(*) n FROM annotations").fetchone()["n"]
        assert total == 1, "an invalid label reached the store"
        assert store.annotation(rows[2]["item_key"], RATERS[0]).label == 0


def test_the_relevance_workload_can_be_restricted_to_the_coverage_delta(workload, tmp_path):
    """Relevance labels survive a code change, so only the unlabelled pairs need a rater."""
    store_dir, _out = workload
    rows = _read(store_dir.parent / "files" / f"relevance_{RATERS[0]}.csv")
    assert rows, "the unrestricted export produced no relevance rows"

    delta = tmp_path / "delta.csv"
    keep = [(rows[0]["scenario_id"], rows[0]["job_id"])]
    with delta.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["scenario_id", "job_id"])
        writer.writerows(keep)

    out = tmp_path / "restricted"
    export(store_dir, out, delta, write=True)

    restricted = _read(out / f"relevance_{RATERS[0]}.csv")
    assert len(restricted) == 1 < len(rows)
    assert (restricted[0]["scenario_id"], restricted[0]["job_id"]) == keep[0]
    # The claim workload is untouched by the relevance restriction.
    assert len(_read(out / f"claim_{RATERS[0]}.csv")) == len(
        _read(store_dir.parent / "files" / f"claim_{RATERS[0]}.csv"))
