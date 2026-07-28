"""Store behaviour: rater isolation, blinding, revisable labels, effort, adjudication.

These are the invariants the UI must not be able to break, so they are asserted against the
store API directly rather than through any presentation layer:

- a rater can neither READ nor OVERWRITE another rater's annotation, and cannot touch an item
  they were not assigned. Independence of the two raters is a precondition of Cohen's kappa,
  so a leak would invalidate the reported agreement;
- a rater-facing read cannot return the oracle grade or the validator verdict -- there is no
  field on the returned type to carry them -- and a payload that contains one is refused at
  write time;
- a rater may revise their own label before export, and ``duration_ms`` is recorded so
  annotation effort is reportable;
- adjudication is stored as a recorded verdict, never derived.

All raters and labels here are synthetic (``SYNTHETIC-*``); this file fabricates no human data
that anything downstream could read as collected.
"""

from __future__ import annotations

import json

import pytest

from jobrec_eval.annotation_ui.assignment import assign_two_raters
from jobrec_eval.annotation_ui.store import (
    DB_FILENAME,
    KIND_CLAIM,
    KIND_RELEVANCE,
    META_ASSIGNMENT_SEED,
    SCHEMA_VERSION,
    AnnotationItem,
    BlindingViolationError,
    ClaimOccurrence,
    InvalidLabelError,
    NotAssignedError,
    UnknownRaterError,
    open_store,
)

RATER_A = "SYNTHETIC-RATER-A"
RATER_B = "SYNTHETIC-RATER-B"
SEED = 4242


def synthetic_items(count: int = 6) -> list[AnnotationItem]:
    """Obviously-synthetic items: fake scenarios, fake jobs, fake claims."""
    items = []
    for index in range(count):
        items.append(AnnotationItem(
            item_key=f"rel::SYN-SC-{index:02d}::SYN-job-{index:02d}", kind=KIND_RELEVANCE,
            payload={"scenario": {"scenario_id": f"SYN-SC-{index:02d}", "conversation": []},
                     "job": {"job_id": f"SYN-job-{index:02d}", "title": "Synthetic Role"}},
            analysis={"oracle_grade": index % 4},
            scenario_id=f"SYN-SC-{index:02d}", job_id=f"SYN-job-{index:02d}"))
    items.append(AnnotationItem(
        item_key="clm::SYN-claim-1", kind=KIND_CLAIM,
        payload={"claim_text": "Synthetic claim.", "evidence": []},
        analysis={"validator_supported_binary": {"SYN-run-1": 1}},
        claim_id="SYN-claim-1",
        occurrences=(ClaimOccurrence(run_id="SYN-run-1", claim_id="SYN-claim-1",
                                     variant="full", scenario_id="SYN-SC-00",
                                     validator_label=1, support_status="supported"),)))
    return items


@pytest.fixture()
def store(tmp_path):
    """A store with synthetic items, two synthetic raters and a saved assignment plan."""
    with open_store(tmp_path / "annotation") as store:
        store.register_raters([RATER_A, RATER_B])
        store.add_items(synthetic_items())
        store.save_assignment_plan(assign_two_raters(store.item_keys(), [RATER_A, RATER_B], SEED))
        yield store


def test_store_is_one_file_with_wal_and_a_busy_timeout(tmp_path):
    """WAL plus a busy timeout is what lets two raters work concurrently."""
    with open_store(tmp_path / "annotation") as store:
        assert store.path.name == DB_FILENAME
        assert store.path.is_file()
        connection = store._db  # noqa: SLF001 - pragma check has no public surface
        assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] > 0
        assert store.meta()["schema_version"] == SCHEMA_VERSION


def relevance_keys(store, rater_id=RATER_A):
    """This rater's relevance items, in their queue order (0-3 labels)."""
    return [entry.item_key for entry in store.queue(rater_id) if entry.kind == KIND_RELEVANCE]


def test_a_rater_cannot_read_another_raters_annotation(store):
    """The read side of rater isolation."""
    item_key = relevance_keys(store)[0]
    store.upsert_annotation(item_key, RATER_A, 3, notes="A's reasoning")

    # A reads their own answer.
    own = store.annotation(item_key, RATER_A)
    assert own is not None and own.label == 3 and own.notes == "A's reasoning"

    # B is assigned the same item (two slots) but sees no label of their own, and there is no
    # API that hands them A's row: the query is filtered on rater_id.
    assert store.annotation(item_key, RATER_B) is None
    b_view = store.rater_item(RATER_B, item_key)
    assert b_view.label is None
    assert b_view.notes == ""
    assert "A's reasoning" not in json.dumps(b_view.payload)


def test_a_rater_cannot_overwrite_another_raters_annotation(store):
    """The write side: an upsert only ever touches the caller's own (item, rater) row."""
    item_key = relevance_keys(store)[0]
    store.upsert_annotation(item_key, RATER_A, 3)
    store.upsert_annotation(item_key, RATER_B, 0)

    assert store.annotation(item_key, RATER_A).label == 3
    assert store.annotation(item_key, RATER_B).label == 0

    # Revising B's label leaves A's untouched.
    store.upsert_annotation(item_key, RATER_B, 1)
    assert store.annotation(item_key, RATER_A).label == 3
    assert store.annotation(item_key, RATER_B).label == 1


def test_an_unassigned_rater_is_refused_on_read_and_write(store):
    """A rater outside the item's two slots cannot address it at all."""
    intruder = "SYNTHETIC-RATER-C"
    store.register_raters([RATER_A, RATER_B, intruder])
    item_key = relevance_keys(store)[0]

    with pytest.raises(NotAssignedError):
        store.upsert_annotation(item_key, intruder, 2)
    with pytest.raises(NotAssignedError):
        store.annotation(item_key, intruder)
    with pytest.raises(NotAssignedError):
        store.rater_item(intruder, item_key)
    with pytest.raises(UnknownRaterError):
        store.queue("SYNTHETIC-RATER-NEVER-REGISTERED")


def test_a_rater_may_revise_their_own_label_before_export(store):
    """Upsert, not insert: created_at is kept, updated_at moves, so revisions are visible."""
    item_key = relevance_keys(store)[0]
    first = store.upsert_annotation(item_key, RATER_A, 1, notes="first pass", duration_ms=4000)
    revised = store.upsert_annotation(item_key, RATER_A, 2, notes="reread the turns",
                                      duration_ms=9000)

    assert revised.label == 2
    assert revised.notes == "reread the turns"
    assert revised.duration_ms == 9000
    assert revised.updated_at >= first.updated_at
    # One row per (item, rater), so a revision replaces rather than duplicates.
    assert store.progress(RATER_A).completed == 1


def test_rater_facing_reads_cannot_return_the_analysis_side(store):
    """Blinding is structural: the rater-facing type has no field for the machine's answer."""
    entry = next(e for e in store.queue(RATER_A) if e.kind == KIND_RELEVANCE)
    assert not hasattr(entry, "analysis")
    assert "oracle_grade" not in json.dumps(entry.payload)

    # The analysis value IS stored -- the export and the oracle-vs-human comparison need it --
    # but only through the export view, which takes no rater id at all.
    record = next(r for r in store.iter_export_records() if r.item_key == entry.item_key)
    assert "oracle_grade" in record.analysis


def test_a_payload_carrying_the_machine_answer_is_refused(tmp_path):
    """The store refuses to persist an unblinded payload, whoever builds it."""
    with open_store(tmp_path / "annotation") as store:
        store.register_raters([RATER_A, RATER_B])
        with pytest.raises(BlindingViolationError, match="oracle_grade"):
            store.add_items([AnnotationItem(
                item_key="rel::SYN::leak", kind=KIND_RELEVANCE,
                payload={"job": {"job_id": "SYN"}, "oracle_grade": 3})])
        # Nested just as fatal: the check walks the whole structure.
        with pytest.raises(BlindingViolationError, match="support_status"):
            store.add_items([AnnotationItem(
                item_key="clm::SYN::leak", kind=KIND_CLAIM,
                payload={"claim_text": "x", "evidence": [{"support_status": "supported"}]})])
        assert store.item_count() == 0


def test_labels_outside_the_kind_range_are_refused(store):
    """0-3 for relevance, {0,1} for claims -- enforced on write, not at export time."""
    relevance_key = next(i.item_key for i in store.queue(RATER_A) if i.kind == KIND_RELEVANCE)
    with pytest.raises(InvalidLabelError):
        store.upsert_annotation(relevance_key, RATER_A, 4)
    with pytest.raises(InvalidLabelError):
        store.upsert_annotation(relevance_key, RATER_A, -1)

    claim_key = next(i.item_key for i in store.queue(RATER_A) if i.kind == KIND_CLAIM)
    with pytest.raises(InvalidLabelError):
        store.upsert_annotation(claim_key, RATER_A, 2)
    store.upsert_annotation(claim_key, RATER_A, 1)
    assert store.annotation(claim_key, RATER_A).label == 1


def test_queue_progress_and_effort_are_recorded(store):
    """Progress drives the UI's progress bar; duration drives the reportable effort."""
    queue = store.queue(RATER_A)
    assert [entry.position for entry in queue] == sorted(entry.position for entry in queue)
    assert store.progress(RATER_A).assigned == len(queue)
    assert store.progress(RATER_A).completed == 0
    assert store.next_item(RATER_A).item_key == queue[0].item_key
    assert store.next_item(RATER_A, kind=KIND_CLAIM).kind == KIND_CLAIM

    graded = relevance_keys(store)
    store.upsert_annotation(graded[0], RATER_A, 2, duration_ms=5000)
    store.upsert_annotation(graded[1], RATER_A, 1, duration_ms=15000)

    progress = store.progress(RATER_A)
    assert progress.completed == 2
    assert progress.remaining == progress.assigned - 2
    assert progress.total_duration_ms == 20000
    assert progress.median_duration_ms == 10000
    pending = {e.item_key for e in store.queue(RATER_A, include_done=False)}
    assert graded[0] not in pending and graded[1] not in pending
    assert store.next_item(RATER_A).item_key in pending

    effort = store.annotation_effort()
    assert effort["annotations"] == 2
    assert effort["timed_annotations"] == 2
    assert effort["total_duration_ms"] == 20000


def test_completed_items_disagreements_and_adjudication(store):
    """Both-slots-done, the disagreement worklist and a recorded verdict."""
    keys = [entry.item_key for entry in store.queue(RATER_A) if entry.kind == KIND_RELEVANCE][:3]
    agree, disagree, half = keys

    store.upsert_annotation(agree, RATER_A, 2)
    store.upsert_annotation(agree, RATER_B, 2)
    store.upsert_annotation(disagree, RATER_A, 3)
    store.upsert_annotation(disagree, RATER_B, 1)
    store.upsert_annotation(half, RATER_A, 0)

    assert set(store.completed_item_keys()) == {agree, disagree}
    assert half not in store.completed_item_keys()

    disagreements = store.disagreements()
    assert [d.item_key for d in disagreements] == [disagree]
    assert {disagreements[0].slot_1_label, disagreements[0].slot_2_label} == {1, 3}
    assert disagreements[0].adjudicated is False

    verdict = store.record_adjudication(disagree, "SYNTHETIC-ADJUDICATOR", 2, reason="synthetic")
    assert verdict.final_label == 2
    assert store.adjudication(disagree).final_label == 2
    assert store.disagreements()[0].adjudicated is True
    assert store.disagreements(unadjudicated_only=True) == ()
    with pytest.raises(InvalidLabelError):
        store.record_adjudication(disagree, "SYNTHETIC-ADJUDICATOR", 9)


def test_meta_records_the_seed_and_survives_reopening(tmp_path):
    """The seed travels with the data, so an assignment stays reproducible from the file."""
    directory = tmp_path / "annotation"
    with open_store(directory) as store:
        store.register_raters([RATER_A, RATER_B])
        store.add_items(synthetic_items(2))
        store.save_assignment_plan(assign_two_raters(store.item_keys(), [RATER_A, RATER_B], SEED))
        store.set_meta({"experiment_id": "exp-SYNTHETIC"})

    with open_store(directory, create=False) as reopened:
        assert reopened.meta()[META_ASSIGNMENT_SEED] == str(SEED)
        assert reopened.meta()["experiment_id"] == "exp-SYNTHETIC"
        assert reopened.raters() == (RATER_A, RATER_B)
        assert reopened.item_count() == 3

    with pytest.raises(FileNotFoundError):
        open_store(tmp_path / "no-such-dir", create=False)
