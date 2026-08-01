"""Schema v2 regression on the real 210-run pilot, not on a fixture.

The v1 store keyed claim items on ``claim_id``, a digest of the rendered SENTENCE. Measured on
this pilot: 416 claim ids span 694 propositions, 178 of those ids cover more than one, and 278
propositions -- 40% of the total -- would never have been judged as themselves. Their labels
would have been inherited from whichever occurrence a rater happened to see, over evidence
unioned across up to five different propositions.

These tests run against ``artifacts/pilot_deterministic`` when it exists, and skip when it does
not, so the suite stays runnable on a clean checkout while the numbers are asserted against real
output rather than a synthetic stand-in.
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest

from jobrec_eval.annotation_ui.store import (
    DB_FILENAME,
    KIND_CLAIM,
    SCHEMA_VERSION,
    V1_ARCHIVE_FILENAME,
    V1_ARCHIVE_MANIFEST,
    AnnotationItem,
    SchemaVersionError,
    archive_v1_database,
    open_store,
)

PILOT_ROOT = Path("artifacts/pilot_deterministic")


def _occurrence_rows() -> list[dict]:
    analysis = sorted(PILOT_ROOT.glob("exp-*/normalized/claim_occurrences.csv"))
    if not analysis:
        pytest.skip("no deterministic pilot on disk; run scripts/... pipeline first")
    return list(csv.DictReader(analysis[0].open(encoding="utf-8")))


def test_the_pilot_has_claim_ids_covering_several_propositions():
    """The premise. If this ever stops holding, the rest of the file proves nothing."""
    rows = _occurrence_rows()
    by_claim_id: dict[str, set[str]] = {}
    for row in rows:
        by_claim_id.setdefault(row["claim_id"], set()).add(row["annotation_signature"])

    signatures = {r["annotation_signature"] for r in rows}
    ambiguous = {c: s for c, s in by_claim_id.items() if len(s) > 1}

    assert len(signatures) > len(by_claim_id), (
        f"{len(signatures)} signatures vs {len(by_claim_id)} claim ids")
    assert ambiguous, "no claim_id covered several propositions"
    # The numbers this migration was justified by.
    assert len(by_claim_id) == 416, len(by_claim_id)
    assert len(signatures) == 694, len(signatures)
    assert len(ambiguous) == 178, len(ambiguous)
    merged_away = sum(len(s) for s in ambiguous.values()) - len(ambiguous)
    assert merged_away == 278, merged_away


def test_the_store_holds_one_item_per_signature_not_per_claim_id(tmp_path):
    """694 items, not 416 -- built through the real store, from the real occurrences."""
    rows = _occurrence_rows()
    by_signature: dict[str, list[dict]] = {}
    for row in rows:
        by_signature.setdefault(row["annotation_signature"], []).append(row)

    with open_store(tmp_path) as store:
        store.add_items([
            AnnotationItem(
                item_key=f"clm::{signature}", kind=KIND_CLAIM,
                annotation_signature=signature,
                payload={"claim_text": group[0]["text"], "evidence": []},
                analysis={"occurrence_count": len(group)},
                claim_id=group[0]["claim_id"])
            for signature, group in sorted(by_signature.items())
        ])
        keys = store.item_keys(KIND_CLAIM)

    claim_ids = {r["claim_id"] for r in rows}
    assert len(keys) == len(by_signature) == 694
    assert len(keys) != len(claim_ids), "the store still collapsed to one item per claim_id"


def test_two_propositions_sharing_a_claim_id_get_two_items(tmp_path):
    """The exact collision, taken from real data rather than constructed."""
    rows = _occurrence_rows()
    by_claim_id: dict[str, set[str]] = {}
    for row in rows:
        by_claim_id.setdefault(row["claim_id"], set()).add(row["annotation_signature"])
    claim_id, signatures = next((c, s) for c, s in by_claim_id.items() if len(s) > 1)

    with open_store(tmp_path) as store:
        store.add_items([
            AnnotationItem(item_key=f"clm::{sig}", kind=KIND_CLAIM,
                           annotation_signature=sig,
                           payload={"claim_text": "shared sentence", "evidence": []},
                           claim_id=claim_id)
            for sig in sorted(signatures)
        ])
        assert store.item_count(KIND_CLAIM) == len(signatures) > 1


def test_the_signature_is_required_and_unique_for_claim_items(tmp_path):
    with open_store(tmp_path) as store:
        with pytest.raises(ValueError, match="no annotation_signature"):
            store.add_items([AnnotationItem(item_key="clm::x", kind=KIND_CLAIM,
                                            payload={"claim_text": "t"})])

        store.add_items([AnnotationItem(item_key="clm::sig-a", kind=KIND_CLAIM,
                                        annotation_signature="sig-a",
                                        payload={"claim_text": "t"})])
        # A second item_key claiming the SAME proposition is refused by the unique index.
        with pytest.raises(sqlite3.IntegrityError):
            store.add_items([AnnotationItem(item_key="clm::other", kind=KIND_CLAIM,
                                            annotation_signature="sig-a",
                                            payload={"claim_text": "t"})])


def test_rebuilding_the_store_is_idempotent(tmp_path):
    """Same bundles -> same keys, so the migration can be re-run without doubling anything."""
    item = AnnotationItem(item_key="clm::sig-a", kind=KIND_CLAIM,
                          annotation_signature="sig-a", payload={"claim_text": "t"})
    with open_store(tmp_path) as store:
        store.add_items([item])
        store.add_items([item])
        assert store.item_count(KIND_CLAIM) == 1


# ------------------------------------------------------------- the v1 archive
def _v1_path(directory: Path) -> Path:
    """Where the store actually looks, so the guard is exercised rather than bypassed."""
    return directory / DB_FILENAME


def _write_v1(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        "CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);"
        "CREATE TABLE items (item_key TEXT PRIMARY KEY, kind TEXT);"
        "CREATE TABLE annotations (item_key TEXT, rater_id TEXT, label INTEGER);")
    connection.execute("INSERT INTO meta VALUES ('schema_version', '1')")
    connection.execute("INSERT INTO items VALUES ('clm::claim-1', 'claim')")
    connection.execute("INSERT INTO annotations VALUES ('clm::claim-1', 'r1', 1)")
    connection.commit()
    connection.close()


def test_a_v1_database_is_refused_rather_than_upgraded_in_place(tmp_path):
    """CREATE TABLE IF NOT EXISTS would leave v1 tables without the signature columns."""
    _write_v1(_v1_path(tmp_path))
    with pytest.raises(SchemaVersionError, match="NOT upgraded in place"):
        open_store(tmp_path)


def test_archiving_v1_seals_it_and_carries_no_claim_labels(tmp_path):
    source = _v1_path(tmp_path)
    _write_v1(source)

    manifest = archive_v1_database(source)
    assert manifest is not None
    assert manifest["schema_version"] == "1"
    assert manifest["claim_labels_carried_forward"] is False
    assert manifest["row_counts"]["annotations"] == 1
    assert len(manifest["sha256"]) == 64

    archive = tmp_path / V1_ARCHIVE_FILENAME
    assert archive.is_file()
    assert (tmp_path / V1_ARCHIVE_MANIFEST).is_file()
    # The original is untouched and the archive matches it.
    assert source.read_bytes() == archive.read_bytes()

    # Idempotent: a second call neither re-seals nor overwrites.
    again = archive_v1_database(source)
    assert again == manifest


def test_a_v2_store_does_not_inherit_v1_claim_labels(tmp_path):
    """The migration builds a NEW store; no label crosses over."""
    _write_v1(_v1_path(tmp_path))
    archive_v1_database(_v1_path(tmp_path))
    (_v1_path(tmp_path)).unlink()

    with open_store(tmp_path) as store:
        assert store.item_count(KIND_CLAIM) == 0
        assert store.meta()["schema_version"] == SCHEMA_VERSION
        # The sealed v1 labels are still on disk, and still not in the new store.
        sealed = json.loads((tmp_path / V1_ARCHIVE_MANIFEST).read_text(encoding="utf-8"))
        assert sealed["row_counts"]["annotations"] == 1
