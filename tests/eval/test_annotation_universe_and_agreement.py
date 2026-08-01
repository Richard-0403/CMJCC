"""The annotation frame is pre-registered, and only labels inside it reach kappa.

Four defects are pinned here, each one a way a number came out looking usable:

* **coverage against whatever got annotated.** With the denominator taken from the labels
  themselves, coverage reads 100% for any sample. The frame is fixed FIRST -- every delivered
  signature, plus a seeded stratified draw of the withheld ones -- written to a manifest, and
  coverage is measured against that.
* **kappa over rows instead of judgements.** One human judgement covers every occurrence of a
  proposition and the CSV replicates it across them, so kappa over rows reported n=3578 for 694
  judgements on the 210-run pilot.
* **kappa over labels that describe other runs.** An obsolete signature, or one from another
  experiment, changed the reported agreement for this one.
* **withheld claims counted as missing annotation.** A dropped claim that the frame
  deliberately left out is out of scope, not unlabelled work.

Every rater id here is prefixed ``SYNTHETIC-`` and every label is invented by the test.
"""

from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from jobrec_eval.annotation import claim_agreement
from jobrec_eval.annotation_linkage import (
    DELIVERY_DELIVERED,
    DELIVERY_DROPPED,
    DROPPED_STRATA_FIELDS,
    EXCLUDED_NO_SIGNATURE,
    EXCLUDED_OBSOLETE,
    EXCLUDED_OTHER_EXPERIMENT,
    EXCLUDED_OUT_OF_UNIVERSE,
    build_annotation_universe,
    claim_occurrences,
    filter_labels_to_universe,
    link_claim_labels,
)
from jobrec_eval.annotation_ui.assignment import assign_two_raters
from jobrec_eval.annotation_ui.export import (
    CLAIM_COLUMNS,
    CLAIMS_CSV_FILENAME,
    export_annotations,
    validate_claim_rows,
)
from jobrec_eval.annotation_ui.loader import build_items
from jobrec_eval.annotation_ui.store import KIND_CLAIM, open_store

EXPERIMENT = "exp-synthetic-universe"
PILOT_ROOT = Path("artifacts/pilot_deterministic")

#: Rater ids used by this module. Prefixed so no fixture label can be mistaken for a human one.
RATERS = ("SYNTHETIC-R1", "SYNTHETIC-R2")


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


@pytest.fixture
def occurrences() -> list[dict]:
    """Three delivered propositions and twelve withheld ones across two strata."""
    delivered = [_claim(expected_value=v) for v in (4000, 5000, 6000)]
    withheld = [
        _claim(expected_value=10_000 + i, text="Withheld salary claim.",
               support_status="unsupported")
        for i in range(7)
    ] + [
        _claim(expected_value=20_000 + i, claim_type="constraint_note",
               predicate="work_mode_matches", field_name="work_mode",
               text="Withheld work-mode claim.", support_status="contradicted")
        for i in range(5)
    ]
    return claim_occurrences(EXPERIMENT, [
        _run(run_id="run-1", claims=delivered, dropped_claims=withheld),
        _run(run_id="run-2", repeat_index=1, claims=delivered),
    ])


# --------------------------------------------------- the pre-registered frame
def test_delivered_signatures_are_taken_whole_and_withheld_ones_are_sampled(occurrences):
    """Coverage over a subset of the delivered explanations would not be coverage at all."""
    universe = build_annotation_universe(EXPERIMENT, occurrences, dropped_per_stratum=2)

    delivered = {r["annotation_signature"] for r in occurrences
                 if r["delivery_status"] == DELIVERY_DELIVERED}
    dropped = {r["annotation_signature"] for r in occurrences
               if r["delivery_status"] == DELIVERY_DROPPED}

    assert set(universe.delivered) == delivered
    assert len(dropped) == 12
    # Two strata (support_status differs), two drawn from each.
    assert len(universe.dropped_sampled) == 4
    assert set(universe.dropped_sampled) < dropped
    assert universe.size == len(delivered) + 4


def test_the_draw_is_reproducible_and_seed_dependent(occurrences):
    """A frame nobody can reproduce is not a pre-registration."""
    first = build_annotation_universe(EXPERIMENT, occurrences, seed=7, dropped_per_stratum=2)
    again = build_annotation_universe(EXPERIMENT, occurrences, seed=7, dropped_per_stratum=2)
    assert first.signatures == again.signatures
    assert first.dropped_sampled == again.dropped_sampled

    other = build_annotation_universe(EXPERIMENT, occurrences, seed=8, dropped_per_stratum=2)
    # Same size, and the seed is what selected the members.
    assert len(other.dropped_sampled) == len(first.dropped_sampled)
    assert other.seed == 8


def test_a_stratum_appearing_does_not_reshuffle_the_others(occurrences):
    """Per-stratum seeding, so adding a claim type cannot change another type's draw."""
    salary_only = [r for r in occurrences
                   if r["delivery_status"] == DELIVERY_DELIVERED
                   or r["support_status"] == "unsupported"]
    narrow = build_annotation_universe(EXPERIMENT, salary_only, dropped_per_stratum=2)
    wide = build_annotation_universe(EXPERIMENT, occurrences, dropped_per_stratum=2)

    salary_signatures = {r["annotation_signature"] for r in occurrences
                         if r["support_status"] == "unsupported"}
    assert (set(narrow.dropped_sampled) & salary_signatures
            == set(wide.dropped_sampled) & salary_signatures)


def test_the_manifest_records_the_seed_the_rules_and_the_frame(occurrences):
    manifest = build_annotation_universe(
        EXPERIMENT, occurrences, seed=99, dropped_per_stratum=3).manifest()

    assert manifest["sampling_seed"] == 99
    assert manifest["strata_fields"] == list(DROPPED_STRATA_FIELDS)
    assert manifest["dropped_per_stratum"] == 3
    assert "per stratum" in manifest["sampling_rule"]
    assert manifest["counts"]["delivered_signatures"] == 3
    assert manifest["counts"]["dropped_population_signatures"] == 12
    assert manifest["counts"]["dropped_sampled_signatures"] == 6
    # Per-stratum populations, so an exhausted stratum is distinguishable from a subsampled one.
    assert [s["population"] for s in manifest["strata"]] == [5, 7]
    assert all(set(s["stratum"]) == set(DROPPED_STRATA_FIELDS) for s in manifest["strata"])
    # The frame itself, so coverage can be RECOMPUTED rather than merely believed.
    assert len(manifest["delivered"]) == 3
    assert len(manifest["dropped_sampled"]) == 6
    json.dumps(manifest)  # must survive the manifest writer


def test_a_signature_seen_both_delivered_and_withheld_is_not_also_sampled():
    """The user saw it, so it is in the frame whole; it must not be double-counted."""
    claim = _claim()
    both = claim_occurrences(EXPERIMENT, [
        _run(run_id="run-1", claims=[claim]),
        _run(run_id="run-2", dropped_claims=[claim]),
    ])
    universe = build_annotation_universe(EXPERIMENT, both)

    assert len(universe.delivered) == 1
    assert universe.dropped_sampled == ()
    assert universe.dropped_population == ()
    assert universe.size == 1


# ------------------------------------------ the frame is the coverage denominator
def test_an_unsampled_withheld_claim_is_out_of_scope_not_missing_annotation(occurrences):
    universe = build_annotation_universe(EXPERIMENT, occurrences, dropped_per_stratum=2)
    labels = [{"experiment_id": EXPERIMENT, "annotation_signature": sig}
              for sig in sorted(universe.signatures)]

    report = link_claim_labels(EXPERIMENT, occurrences, labels, universe=universe)

    assert report.universe_signatures == universe.size
    assert report.coverage == 1.0, "a fully annotated FRAME did not read as full coverage"
    assert report.missing_signatures == 0
    # The eight withheld propositions the frame left out are reported as out of scope.
    assert report.out_of_universe_signatures == 8
    assert report.as_dict()["coverage_denominator"] == "pre_registered_universe"


def test_without_a_frame_the_denominator_is_every_signature_produced(occurrences):
    """The stricter default: no frame registered means nothing is out of scope."""
    labels = [{"experiment_id": EXPERIMENT, "annotation_signature": r["annotation_signature"]}
              for r in occurrences if r["delivery_status"] == DELIVERY_DELIVERED]
    report = link_claim_labels(EXPERIMENT, occurrences, labels)

    assert report.universe_signatures is None
    assert report.current_signatures == 15
    assert report.coverage == pytest.approx(3 / 15)
    assert report.as_dict()["coverage_denominator"] == "signatures_produced"


# ------------------------------------------------- which label rows reach kappa
def test_each_refused_label_row_is_counted_under_its_own_reason(occurrences):
    universe = build_annotation_universe(EXPERIMENT, occurrences, dropped_per_stratum=2)
    inside = sorted(universe.signatures)[0]
    unsampled = next(r["annotation_signature"] for r in occurrences
                     if r["delivery_status"] == DELIVERY_DROPPED
                     and r["annotation_signature"] not in universe.signatures)

    result = filter_labels_to_universe([
        {"experiment_id": EXPERIMENT, "annotation_signature": inside},
        {"experiment_id": "exp-somewhere-else", "annotation_signature": inside},
        {"experiment_id": EXPERIMENT, "annotation_signature": ""},
        {"experiment_id": EXPERIMENT, "annotation_signature": "sig-never-produced"},
        {"experiment_id": EXPERIMENT, "annotation_signature": unsampled},
    ], experiment_id=EXPERIMENT, universe=universe.signatures)

    assert result.kept == (0,)
    assert result.excluded == {
        EXCLUDED_OTHER_EXPERIMENT: 1,
        EXCLUDED_NO_SIGNATURE: 1,
        EXCLUDED_OUT_OF_UNIVERSE: 2,
    }
    assert result.n_excluded == 4


def _write_claim_csv(path: Path, rows: list[dict]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CLAIM_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CLAIM_COLUMNS})
    return path


def _label_row(signature: str, *, rater_1: int, rater_2: int, run_id: str = "run-1",
               experiment_id: str = EXPERIMENT, validator: int = 1) -> dict:
    return {"experiment_id": experiment_id, "run_id": run_id, "claim_id": "claim-shared",
            "annotation_signature": signature, "delivery_status": DELIVERY_DELIVERED,
            "rater_1": rater_1, "rater_2": rater_2, "validator": validator,
            "adjudicated": "", "notes": ""}


def test_obsolete_labels_do_not_change_kappa(tmp_path, occurrences):
    """A signature this experiment never produced must not move the reported agreement."""
    universe = build_annotation_universe(EXPERIMENT, occurrences, dropped_per_stratum=2)
    frame = sorted(universe.signatures)
    # Deliberately mixed labels, so kappa is a real number rather than a degenerate 1.0.
    clean = [_label_row(sig, rater_1=i % 2, rater_2=(i + i // 3) % 2)
             for i, sig in enumerate(frame)]
    polluted = clean + [
        _label_row(f"sig-obsolete-{i}", rater_1=1, rater_2=0) for i in range(20)]

    before = claim_agreement(_write_claim_csv(tmp_path / "clean.csv", clean),
                             occurrences=occurrences, experiment_id=EXPERIMENT,
                             universe=universe)
    after = claim_agreement(_write_claim_csv(tmp_path / "polluted.csv", polluted),
                            occurrences=occurrences, experiment_id=EXPERIMENT,
                            universe=universe)

    assert before["cohens_kappa"] is not None
    assert after["cohens_kappa"] == before["cohens_kappa"]
    assert after["n_items"] == before["n_items"] == len(frame)
    assert after["label_filter"]["excluded_by_reason"] == {EXCLUDED_OBSOLETE: 20}
    assert after["linkage"]["obsolete_signatures"] == 20


def test_labels_from_another_experiment_do_not_change_kappa(tmp_path, occurrences):
    universe = build_annotation_universe(EXPERIMENT, occurrences, dropped_per_stratum=2)
    frame = sorted(universe.signatures)
    clean = [_label_row(sig, rater_1=i % 2, rater_2=(i + i // 3) % 2)
             for i, sig in enumerate(frame)]
    # Same signatures, wrong batch: these describe runs this analysis is not about.
    polluted = clean + [_label_row(sig, rater_1=1, rater_2=0,
                                   experiment_id="exp-somewhere-else", run_id="run-elsewhere")
                        for sig in frame]

    before = claim_agreement(_write_claim_csv(tmp_path / "clean.csv", clean),
                             occurrences=occurrences, experiment_id=EXPERIMENT,
                             universe=universe)
    after = claim_agreement(_write_claim_csv(tmp_path / "polluted.csv", polluted),
                            occurrences=occurrences, experiment_id=EXPERIMENT,
                            universe=universe)

    assert after["cohens_kappa"] == before["cohens_kappa"]
    assert after["label_filter"]["excluded_by_reason"] == {
        EXCLUDED_OTHER_EXPERIMENT: len(frame)}


def test_a_v1_file_with_no_signature_column_cannot_be_reported(tmp_path, occurrences):
    """Every pre-migration label file is like this: no way to tell which proposition it judged."""
    path = tmp_path / "v1.csv"
    pd.DataFrame([{"run_id": "run-1", "claim_id": "claim-shared", "rater_1": 1, "rater_2": 1,
                   "validator": 1, "adjudicated": ""}]).to_csv(path, index=False)

    result = claim_agreement(path, occurrences=occurrences, experiment_id=EXPERIMENT,
                             universe=build_annotation_universe(EXPERIMENT, occurrences))

    assert result["cohens_kappa"] is None
    assert result["n_items"] == 0
    assert "not a measurement" in result["unusable_reason"]


def test_kappa_counts_judgements_not_replicated_occurrence_rows(tmp_path, occurrences):
    """One judgement per proposition; the CSV replicates it across every occurrence."""
    universe = build_annotation_universe(EXPERIMENT, occurrences, dropped_per_stratum=2)
    frame = sorted(universe.signatures)
    rows = []
    for i, signature in enumerate(frame):
        row = _label_row(signature, rater_1=i % 2, rater_2=(i + i // 3) % 2)
        # Adjudicate the disagreements, so every row carries a gold value and the two units
        # below differ only by the replication, not by which rows resolved.
        if row["rater_1"] != row["rater_2"]:
            row["adjudicated"] = 1
        for run in (1, 2, 3, 4):
            rows.append({**row, "run_id": f"run-{run}"})

    result = claim_agreement(_write_claim_csv(tmp_path / "replicated.csv", rows),
                             occurrences=occurrences, experiment_id=EXPERIMENT,
                             universe=universe)

    assert result["n_label_rows"] == len(frame) * 4
    assert result["n_items"] == len(frame)
    # The validator ran per run, so ITS agreement is reported per occurrence and labelled so.
    assert result["n_validator_occurrences"] == len(frame) * 4
    assert result["n_gold_items"] == len(frame)


# ------------------------------------------- the frame reaches the rater's queue
@pytest.fixture(scope="module")
def bundles_with_withheld_claims(annotation_experiment):
    """The real bundles, with each run's first claim copied into ``dropped_claims``.

    The deterministic run delivers everything, so a withheld claim has to be introduced to test
    the withheld path at all. Only the DELIVERY of an existing claim is changed -- the claim's
    content, its evidence and every label stay untouched, so nothing about the recommendation or
    the extraction semantics is altered.
    """
    bundles = [copy.deepcopy(b) for b in annotation_experiment.bundles]
    for index, bundle in enumerate(bundles):
        if not bundle.claims:
            continue
        withheld = copy.deepcopy(bundle.claims[0])
        withheld["text"] = f"SYNTHETIC withheld claim {index}."
        withheld["expected_value"] = 90_000 + index
        withheld["support_status"] = "unsupported"
        bundle.dropped_claims = [withheld]
    return bundles


def test_only_the_pre_registered_withheld_claims_become_items(
        annotation_experiment, bundles_with_withheld_claims):
    """The queue holds every delivered proposition plus exactly the sampled withheld ones."""
    everything = build_items(annotation_experiment.experiment_dir,
                             annotation_experiment.scenarios_path,
                             annotation_experiment.catalog_path,
                             bundles=bundles_with_withheld_claims)
    assert everything.stats.dropped_claim_items > 1, (
        "the fixture produced no withheld claims, so the frame proved nothing")

    sample = sorted(
        i.annotation_signature for i in everything.claim_items
        if i.payload["delivery_status"] == DELIVERY_DROPPED)[:2]
    framed = build_items(annotation_experiment.experiment_dir,
                         annotation_experiment.scenarios_path,
                         annotation_experiment.catalog_path,
                         bundles=bundles_with_withheld_claims,
                         dropped_sample=sample)

    assert framed.stats.dropped_claim_items == 2
    assert sorted(i.annotation_signature for i in framed.claim_items
                  if i.payload["delivery_status"] == DELIVERY_DROPPED) == sample
    # Delivered items are untouched by the withheld frame.
    assert framed.stats.delivered_claim_items == everything.stats.delivered_claim_items
    # And a withheld claim really is on a rater's screen, marked as withheld.
    withheld = next(i for i in framed.claim_items
                    if i.payload["delivery_status"] == DELIVERY_DROPPED)
    assert withheld.payload["delivery_status"] == DELIVERY_DROPPED
    assert withheld.payload["claim_text"]


def test_the_rater_payload_shows_this_signature_and_never_the_validator(
        annotation_experiment, bundles_with_withheld_claims):
    """The rater judges one proposition's own values; the verdict stays analysis-side."""
    built = build_items(annotation_experiment.experiment_dir,
                        annotation_experiment.scenarios_path,
                        annotation_experiment.catalog_path,
                        bundles=bundles_with_withheld_claims)
    required = ("claim_text", "predicate", "claim_field", "claim_job_id", "expected_value",
                "observed_value", "claim_args", "evidence", "annotation_signature",
                "delivery_status")

    for item in built.claim_items:
        assert set(required) <= set(item.payload), sorted(set(required) - set(item.payload))
        assert item.payload["annotation_signature"] == item.annotation_signature
        for forbidden in ("validator", "validator_supported_binary", "support_status",
                          "supported_binary"):
            assert forbidden not in item.payload, f"{item.item_key} leaks {forbidden}"

    # And through the store's rater-facing read, which is what a screen actually gets.
    item = built.claim_items[0]
    with open_store(annotation_experiment.experiment_dir / "annotation-payload-check") as store:
        store.register_raters(RATERS)
        store.add_items([item])
        store.save_assignment_plan(assign_two_raters([item.item_key], RATERS, seed=1))
        rater_item = store.rater_item(RATERS[0], item.item_key)
    serialised = json.dumps(rater_item.payload, default=str)
    assert '"validator"' not in serialised
    # The rater-facing dataclass has no field to carry the analysis side at all.
    assert not hasattr(rater_item, "analysis")


# ------------------------------------ export joins back to the occurrence table
def test_the_export_carries_the_full_experiment_signature_delivery_join(
        tmp_path, annotation_experiment, bundles_with_withheld_claims):
    built = build_items(annotation_experiment.experiment_dir,
                        annotation_experiment.scenarios_path,
                        annotation_experiment.catalog_path,
                        bundles=bundles_with_withheld_claims)
    with open_store(tmp_path / "store") as store:
        store.register_raters(RATERS)
        store.add_items(built.claim_items)
        keys = [item.item_key for item in built.claim_items]
        store.save_assignment_plan(assign_two_raters(keys, RATERS, seed=1))
        for position, item in enumerate(built.claim_items):
            for rater in RATERS:
                store.upsert_annotation(item.item_key, rater, position % 2)
        result = export_annotations(store, tmp_path / "out")

    frame = pd.read_csv(result.claims_path)
    assert list(frame.columns) == CLAIM_COLUMNS
    for column in ("experiment_id", "annotation_signature", "delivery_status"):
        assert frame[column].notna().all(), column
        assert (frame[column].astype(str) != "").all(), column

    experiment_id = annotation_experiment.experiment_dir.name
    assert set(frame["experiment_id"]) == {experiment_id}
    assert set(frame["delivery_status"]) == {DELIVERY_DELIVERED, DELIVERY_DROPPED}
    # Every row joins back to a real occurrence of that signature, and no row invents one.
    occurrence_keys = {(o.run_id, o.claim_id, o.annotation_signature)
                       for item in built.claim_items for o in item.occurrences}
    assert {(r.run_id, r.claim_id, r.annotation_signature)
            for r in frame.itertuples()} == occurrence_keys
    # The manifest counts judgements by signature, not by claim_id.
    counts = result.manifest["counts"]
    assert counts["claim_items_exported"] == frame["annotation_signature"].nunique()
    assert counts["claim_items_exported"] >= counts["claim_ids_exported"]
    assert set(counts["claim_rows_by_delivery_status"]) == {DELIVERY_DELIVERED,
                                                           DELIVERY_DROPPED}
    assert result.hashes[CLAIMS_CSV_FILENAME]


def test_one_claim_id_covering_two_propositions_is_not_a_duplicate_row():
    """The old uniqueness key refused to write ANY file when this happened."""
    shared = [
        {"experiment_id": EXPERIMENT, "run_id": "run-1", "claim_id": "claim-shared",
         "annotation_signature": "sig-low", "delivery_status": DELIVERY_DELIVERED,
         "rater_1": 1, "rater_2": 1, "validator": 1, "adjudicated": "", "notes": ""},
        {"experiment_id": EXPERIMENT, "run_id": "run-1", "claim_id": "claim-shared",
         "annotation_signature": "sig-high", "delivery_status": DELIVERY_DELIVERED,
         "rater_1": 0, "rater_2": 0, "validator": 1, "adjudicated": "", "notes": ""},
    ]
    validate_claim_rows(shared)  # must not raise

    from jobrec_eval.annotation_ui.export import ExportValidationError
    with pytest.raises(ExportValidationError, match="must appear once"):
        validate_claim_rows([shared[0], dict(shared[0])])


# ------------------------------------------------- against the real 210-run pilot
def test_the_real_pilot_frame_is_every_delivered_signature():
    """On the deterministic pilot nothing was withheld, so the frame is all 694 propositions."""
    analysis = sorted(PILOT_ROOT.glob("exp-*/normalized/claim_occurrences.csv"))
    if not analysis:
        pytest.skip("no deterministic pilot on disk")
    rows = list(csv.DictReader(analysis[0].open(encoding="utf-8")))
    experiment_id = analysis[0].parent.parent.name

    universe = build_annotation_universe(experiment_id, rows)

    assert universe.size == 694
    assert len(universe.delivered) == 694
    assert universe.dropped_sampled == ()
    # Rebuilt from the same rows, byte-identical apart from the timestamp.
    again = build_annotation_universe(experiment_id, rows)
    assert {k: v for k, v in universe.manifest().items() if k != "created_at"} == {
        k: v for k, v in again.manifest().items() if k != "created_at"}
    # And the frame is smaller than the occurrence table but larger than the claim_id count.
    assert len({r["claim_id"] for r in rows}) == 416 < universe.size < len(rows)


def test_the_real_pilot_has_no_reusable_label_from_the_v1_store():
    """0/694 overlap is why no claim label was inherited. Asserted, not asserted-in-prose."""
    analysis = sorted(PILOT_ROOT.glob("exp-*/normalized/claim_occurrences.csv"))
    if not analysis:
        pytest.skip("no deterministic pilot on disk")
    rows = list(csv.DictReader(analysis[0].open(encoding="utf-8")))
    experiment_id = analysis[0].parent.parent.name

    # A v1 label file: run ids and claim ids, no signature column at all.
    v1_labels = [{"run_id": r["run_id"], "claim_id": r["claim_id"], "rater_1": 1, "rater_2": 1}
                 for r in rows[:50]]
    report = link_claim_labels(experiment_id, rows, v1_labels,
                               universe=build_annotation_universe(experiment_id, rows))

    assert report.labelled_signatures == 0
    assert report.overlapping_signatures == 0
    assert report.is_stale
    assert report.excluded_label_rows == {EXCLUDED_NO_SIGNATURE: 50}


def test_the_store_round_trip_holds_694_items_for_the_real_pilot(tmp_path):
    """The count that matters, through the real store rather than a dict."""
    analysis = sorted(PILOT_ROOT.glob("exp-*/normalized/claim_occurrences.csv"))
    if not analysis:
        pytest.skip("no deterministic pilot on disk")
    rows = list(csv.DictReader(analysis[0].open(encoding="utf-8")))

    from jobrec_eval.annotation_ui.loader import claim_item_key
    from jobrec_eval.annotation_ui.store import AnnotationItem, ClaimOccurrence

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["annotation_signature"], []).append(row)

    with open_store(tmp_path) as store:
        store.add_items([
            AnnotationItem(
                item_key=claim_item_key(signature), kind=KIND_CLAIM,
                annotation_signature=signature, claim_id=group[0]["claim_id"],
                payload={"claim_text": group[0]["text"], "evidence": []},
                occurrences=tuple(
                    ClaimOccurrence(run_id=r["run_id"], claim_id=r["claim_id"],
                                    experiment_id=r["experiment_id"],
                                    repeat_index=int(r["repeat_index"]),
                                    annotation_signature=signature,
                                    delivery_status=r["delivery_status"],
                                    variant=r["variant"], scenario_id=r["scenario_id"])
                    for r in group))
            for signature, group in sorted(grouped.items())
        ])
        assert store.item_count(KIND_CLAIM) == 694
        stored = {r["annotation_signature"] for r in store._db.execute(
            "SELECT DISTINCT annotation_signature FROM item_occurrences")}
        assert len(stored) == 694
        # Every occurrence row kept its batch, repeat and delivery status.
        blanks = store._db.execute(
            "SELECT COUNT(*) AS n FROM item_occurrences "
            "WHERE experiment_id = '' OR delivery_status = '' OR repeat_index IS NULL"
        ).fetchone()["n"]
        assert blanks == 0
