"""Write the two human-label CSVs, the archive dump and the manifest.

The CSVs are the existing contract, already consumed by :mod:`jobrec_eval.annotation` and
selected with ``--relevance-source human`` in :mod:`jobrec_eval.cli`:

- ``relevance_labels_human.csv``: ``scenario_id, job_id, rater_1, rater_2, adjudicated``
  (+ ``notes``), grades 0-3, one row per unique pair;
- ``claim_annotations_human.csv``: ``run_id, claim_id, rater_1, rater_2, validator,
  adjudicated``, labels in ``{0, 1}``, one row per ``(run_id, claim_id)``.

``rater_1``/``rater_2`` are the assignment SLOTS, not rater identities. With N raters in the
pool, keying the columns on identity would put a different person in ``rater_1`` on every row
and make the column meaningless; a slot is a stable position, which is what a pairwise kappa
needs.

The ``adjudicated`` column follows the one rule :mod:`jobrec_eval.annotation` documents, and
is filled only from a RECORDED verdict:

- raters agree -> EMPTY. The consuming side treats concordant raters as their own gold, so
  writing a value would double-count it as an adjudication that never happened.
- disagreement with an adjudication -> the recorded ``final_label``.
- disagreement without one -> EMPTY, so the consuming side reports it as an unadjudicated
  disagreement and excludes it from the human gold instead of averaging it.
- one or both rater labels missing -> the item is not written at all and is counted as
  incomplete.

Nothing here invents or averages a label. The rounded rater mean in
:mod:`jobrec_eval.annotation` is a documented legacy fallback for files with NO ``adjudicated``
column; this exporter always writes the column, so that path is never taken for its files.

Validation runs BEFORE anything is written: a duplicated pair or an out-of-range grade makes
:func:`jobrec_eval.annotation.load_adjudicated_relevance_labels` raise at analysis time, and a
file the consuming loader rejects is worse than no file (checklist items 10/11).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ..annotation import ADJUDICATED_COLUMN
from . import ANNOTATION_UI_VERSION
from .store import (
    KIND_CLAIM,
    KIND_RELEVANCE,
    LABEL_RANGES,
    META_ASSIGNMENT_SEED,
    META_CATALOG_PATH,
    META_EXPERIMENT_DIR,
    META_EXPERIMENT_ID,
    META_SCENARIOS_PATH,
    META_SCHEMA_VERSION,
    SLOTS,
    AnnotationStore,
    ExportRecord,
)

#: Output file names. They must match ``jobrec_eval.cli.HUMAN_RELEVANCE_FILENAME`` /
#: ``HUMAN_CLAIMS_FILENAME``, which is what the pipeline looks for beside ``--scenarios``.
#: ``tests/eval/test_annotation_ui_export.py`` asserts the spellings stay identical rather
#: than importing the CLI (and with it the plotting stack) into this data layer.
RELEVANCE_CSV_FILENAME = "relevance_labels_human.csv"
CLAIMS_CSV_FILENAME = "claim_annotations_human.csv"

#: Append-only dump of the whole store, for ``final_release/human_annotations/``.
DUMP_JSONL_FILENAME = "human_annotations.jsonl"

#: Manifest describing one export.
MANIFEST_FILENAME = "annotation_manifest.json"

#: Relevance CSV column order. ``notes`` is optional in the contract and is written because an
#: adjudicator needs the raters' reasoning; the consuming loader ignores it.
RELEVANCE_COLUMNS = ["scenario_id", "job_id", "rater_1", "rater_2", ADJUDICATED_COLUMN, "notes"]

#: Claim CSV column order. ``validator`` is the system's own verdict for that run, taken from
#: the analysis side of the store -- never from a rater.
CLAIM_COLUMNS = ["run_id", "claim_id", "rater_1", "rater_2", "validator", ADJUDICATED_COLUMN]

#: Valid label values per file, reused from the store so there is one definition of "0-3" and
#: "{0,1}" in this package.
_RELEVANCE_LABELS = LABEL_RANGES[KIND_RELEVANCE]
_CLAIM_LABELS = LABEL_RANGES[KIND_CLAIM]


class ExportValidationError(ValueError):
    """The rows would produce a file the consuming loader rejects, so nothing was written."""


@dataclass(frozen=True)
class IncompleteItem:
    """An item not exported because a rater has not labelled it yet."""

    item_key: str
    kind: str
    labelled_slots: tuple[int, ...]
    missing_raters: tuple[str, ...]


@dataclass(frozen=True)
class ExportResult:
    """Paths, counts and content hashes of one export."""

    relevance_path: Path
    claims_path: Path
    dump_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    incomplete: tuple[IncompleteItem, ...] = ()
    #: Rows written per file name.
    row_counts: dict[str, int] = field(default_factory=dict)
    #: ``file name -> sha256`` for every CSV written.
    hashes: dict[str, str] = field(default_factory=dict)

    def incomplete_count(self, kind: str | None = None) -> int:
        """Items skipped as incomplete, optionally for one kind."""
        if kind is None:
            return len(self.incomplete)
        return sum(1 for item in self.incomplete if item.kind == kind)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _adjudicated_cell(record: ExportRecord) -> int | str:
    """The ``adjudicated`` cell of a completed item: a recorded verdict, or empty.

    Empty deliberately covers two different situations -- concordant raters (nothing to
    adjudicate) and an unadjudicated disagreement (nothing decided yet). Both must read as
    "no adjudicated verdict on this row"; the consuming side then uses the raters' shared
    label in the first case and reports the second as unadjudicated, which is exactly the
    distinction a made-up number would destroy.
    """
    if record.adjudication is not None:
        return int(record.adjudication.final_label)
    return ""


def _notes_cell(record: ExportRecord) -> str:
    """Both raters' notes plus any adjudication reason, attributed by slot."""
    parts = [f"rater_{slot}: {record.slot_notes[slot]}"
             for slot in SLOTS if record.slot_notes.get(slot)]
    if record.adjudication is not None and record.adjudication.reason:
        parts.append(f"adjudication: {record.adjudication.reason}")
    return " | ".join(parts)


def _incomplete(record: ExportRecord) -> IncompleteItem:
    return IncompleteItem(
        item_key=record.item_key, kind=record.kind,
        labelled_slots=tuple(sorted(record.slot_labels)),
        missing_raters=tuple(record.slot_raters[slot] for slot in sorted(record.slot_raters)
                             if slot not in record.slot_labels))


def _check_labels(rows: Sequence[dict[str, Any]], columns: Sequence[str],
                  valid: Sequence[int], what: str) -> None:
    offenders = []
    for row in rows:
        for column in columns:
            value = row.get(column, "")
            if value == "" or value is None:
                continue
            try:
                numeric = int(value)
            except (TypeError, ValueError):
                # A non-numeric cell is reported the same way as an out-of-range one: the
                # consuming loader would fail on it either way.
                offenders.append(f"{what} {row}: {column}={value!r}")
                continue
            if numeric not in valid:
                offenders.append(f"{what} {row}: {column}={value!r}")
    if offenders:
        raise ExportValidationError(
            f"label(s) outside {tuple(valid)}: " + "; ".join(offenders[:10]))


def _check_unique(rows: Sequence[dict[str, Any]], keys: Sequence[str]) -> None:
    seen: dict[tuple, int] = {}
    for row in rows:
        key = tuple(str(row[k]) for k in keys)
        seen[key] = seen.get(key, 0) + 1
    duplicated = sorted(k for k, count in seen.items() if count > 1)
    if duplicated:
        raise ExportValidationError(
            f"every {tuple(keys)} must appear once; duplicated "
            + ", ".join(str(k) for k in duplicated[:10]))


def validate_relevance_rows(rows: Sequence[dict[str, Any]]) -> None:
    """Refuse rows that ``load_adjudicated_relevance_labels`` would reject.

    That loader raises on a grade outside 0-3 and on a duplicated ``(scenario_id, job_id)``
    pair, so both are caught here before a file exists.

    Raises:
        ExportValidationError: A grade is out of range, or a pair appears twice.
    """
    _check_labels(rows, ("rater_1", "rater_2", ADJUDICATED_COLUMN), _RELEVANCE_LABELS,
                  "relevance row")
    _check_unique(rows, ("scenario_id", "job_id"))


def validate_claim_rows(rows: Sequence[dict[str, Any]]) -> None:
    """Refuse claim rows the agreement functions could not read.

    Raises:
        ExportValidationError: A label is not in ``{0, 1}``, or a ``(run_id, claim_id)`` pair
            appears twice.
    """
    _check_labels(rows, ("rater_1", "rater_2", "validator", ADJUDICATED_COLUMN), _CLAIM_LABELS,
                  "claim row")
    _check_unique(rows, ("run_id", "claim_id"))


def relevance_rows(store: AnnotationStore) -> tuple[list[dict[str, Any]],
                                                    tuple[IncompleteItem, ...]]:
    """Relevance CSV rows (one per completed item) plus the items skipped as incomplete."""
    rows: list[dict[str, Any]] = []
    skipped: list[IncompleteItem] = []
    for record in store.iter_export_records(kind=KIND_RELEVANCE):
        if not record.complete:
            skipped.append(_incomplete(record))
            continue
        rows.append({
            "scenario_id": record.scenario_id, "job_id": record.job_id,
            "rater_1": record.slot_labels[1], "rater_2": record.slot_labels[2],
            ADJUDICATED_COLUMN: _adjudicated_cell(record),
            "notes": _notes_cell(record),
        })
    return rows, tuple(skipped)


def claim_rows(store: AnnotationStore) -> tuple[list[dict[str, Any]],
                                                tuple[IncompleteItem, ...]]:
    """Claim CSV rows, expanded back to one row per ``(run_id, claim_id)`` occurrence.

    One human judgement covers every occurrence of a content-addressed claim, so the labels
    are REPLICATED across its rows while ``validator`` stays per run: the validator ran once
    per run and its verdict is what the validator-vs-human agreement compares against.
    """
    rows: list[dict[str, Any]] = []
    skipped: list[IncompleteItem] = []
    for record in store.iter_export_records(kind=KIND_CLAIM):
        if not record.complete:
            skipped.append(_incomplete(record))
            continue
        adjudicated = _adjudicated_cell(record)
        for occurrence in record.occurrences:
            rows.append({
                "run_id": occurrence.run_id, "claim_id": occurrence.claim_id,
                "rater_1": record.slot_labels[1], "rater_2": record.slot_labels[2],
                "validator": ("" if occurrence.validator_label is None
                              else int(occurrence.validator_label)),
                ADJUDICATED_COLUMN: adjudicated,
            })
    return rows, tuple(skipped)


def _write_csv(rows: Sequence[dict[str, Any]], columns: Sequence[str], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(list(rows), columns=list(columns))
    frame.to_csv(path, index=False)
    return path


def _dump_records(store: AnnotationStore, export_id: str, exported_at: str) -> list[dict]:
    """The whole store as JSON records: header, raters, items, labels, adjudications.

    The items carry BOTH sides (payload and analysis) because this is the research archive,
    not a rater-facing view: it is what makes the human pass reproducible from
    ``final_release/human_annotations/`` alone.
    """
    header = {
        "record_type": "export_header", "export_id": export_id, "exported_at": exported_at,
        "annotation_ui_version": ANNOTATION_UI_VERSION, "meta": store.meta(),
        "raters": list(store.raters()), "assignment_counts": store.assignment_counts(),
        "annotation_effort": store.annotation_effort(),
    }
    records: list[dict] = [header]
    for record in store.iter_export_records():
        records.append({
            "record_type": "item", "export_id": export_id, "item_key": record.item_key,
            "kind": record.kind, "scenario_id": record.scenario_id, "job_id": record.job_id,
            "claim_id": record.claim_id, "analysis": record.analysis,
            "occurrences": [
                {"run_id": o.run_id, "claim_id": o.claim_id, "variant": o.variant,
                 "scenario_id": o.scenario_id, "validator_label": o.validator_label,
                 "support_status": o.support_status, "fully_resolved": o.fully_resolved}
                for o in record.occurrences],
            "annotations": [
                {"slot": slot, "rater_id": record.slot_raters[slot],
                 "label": record.slot_labels.get(slot),
                 "notes": record.slot_notes.get(slot, ""),
                 "duration_ms": record.slot_durations.get(slot),
                 "created_at": record.slot_created_at.get(slot),
                 "updated_at": record.slot_updated_at.get(slot)}
                for slot in sorted(record.slot_raters)],
            "adjudication": (None if record.adjudication is None else {
                "adjudicator": record.adjudication.adjudicator,
                "final_label": record.adjudication.final_label,
                "reason": record.adjudication.reason,
                "created_at": record.adjudication.created_at}),
            "complete": record.complete,
        })
    return records


def _append_jsonl(records: Sequence[dict], path: Path) -> Path:
    """Append one export generation to the dump.

    Append-only, not rewrite: an annotation archive is a research record, so a second export
    (after more adjudication, say) adds a new generation tagged with its ``export_id`` instead
    of erasing what was already shipped.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True, default=str))
            handle.write("\n")
    return path


def export_annotations(store: AnnotationStore, out_dir: str | Path, *,
                       release_dir: str | Path | None = None) -> ExportResult:
    """Validate, then write both CSVs, the JSONL dump and the manifest.

    Args:
        store: The annotation store to export.
        out_dir: Where the two CSVs go. Copy or point ``--scenarios``' directory at it to feed
            ``jobrec_eval.cli pipeline --relevance-source human``.
        release_dir: Where the JSONL dump and the manifest go; defaults to ``out_dir``. Point
            it at ``final_release/human_annotations/`` for the reproduction package.

    Returns:
        ExportResult: paths, per-file row counts, per-file SHA-256 and the incomplete items
        that were skipped (counted, never written as blank rows).

    Raises:
        ExportValidationError: The rows would produce a file the consuming loader rejects.
    """
    out = Path(out_dir)
    release = Path(release_dir) if release_dir is not None else out
    relevance, relevance_skipped = relevance_rows(store)
    claims, claim_skipped = claim_rows(store)

    # Validate first: no file is created if either table is unusable.
    validate_relevance_rows(relevance)
    validate_claim_rows(claims)

    relevance_path = _write_csv(relevance, RELEVANCE_COLUMNS, out / RELEVANCE_CSV_FILENAME)
    claims_path = _write_csv(claims, CLAIM_COLUMNS, out / CLAIMS_CSV_FILENAME)

    exported_at = datetime.now(UTC).isoformat()
    hashes = {RELEVANCE_CSV_FILENAME: _sha256_file(relevance_path),
              CLAIMS_CSV_FILENAME: _sha256_file(claims_path)}
    export_id = hashlib.sha256(
        f"{hashes[RELEVANCE_CSV_FILENAME]}|{hashes[CLAIMS_CSV_FILENAME]}|{exported_at}".encode()
    ).hexdigest()[:12]

    dump_path = _append_jsonl(_dump_records(store, export_id, exported_at),
                              release / DUMP_JSONL_FILENAME)

    meta = store.meta()
    incomplete = relevance_skipped + claim_skipped
    disagreements = store.disagreements()
    manifest = {
        "export_id": export_id,
        "exported_at": exported_at,
        "annotation_ui_version": ANNOTATION_UI_VERSION,
        "schema_version": meta.get(META_SCHEMA_VERSION, ""),
        "experiment_id": meta.get(META_EXPERIMENT_ID, ""),
        "experiment_dir": meta.get(META_EXPERIMENT_DIR, ""),
        "scenarios_path": meta.get(META_SCENARIOS_PATH, ""),
        "catalog_path": meta.get(META_CATALOG_PATH, ""),
        "assignment_seed": meta.get(META_ASSIGNMENT_SEED, ""),
        "rater_pool": list(store.raters()),
        "assignment_counts": store.assignment_counts(),
        "counts": {
            "relevance_items": store.item_count(KIND_RELEVANCE),
            "claim_items": store.item_count(KIND_CLAIM),
            "relevance_items_exported": len(relevance),
            "claim_items_exported": len({row["claim_id"] for row in claims}),
            "claim_rows_exported": len(claims),
            "incomplete_relevance_items": len(relevance_skipped),
            "incomplete_claim_items": len(claim_skipped),
            "disagreements": len(disagreements),
            "adjudicated_disagreements": sum(1 for d in disagreements if d.adjudicated),
            "unadjudicated_disagreements": sum(1 for d in disagreements if not d.adjudicated),
        },
        "annotation_effort": store.annotation_effort(),
        "files": {
            RELEVANCE_CSV_FILENAME: {"sha256": hashes[RELEVANCE_CSV_FILENAME],
                                     "rows": len(relevance)},
            CLAIMS_CSV_FILENAME: {"sha256": hashes[CLAIMS_CSV_FILENAME], "rows": len(claims)},
            DUMP_JSONL_FILENAME: {"path": str(dump_path)},
        },
        "adjudication_rule": (
            "adjudicated column carries a recorded verdict only; empty means either "
            "concordant raters (their shared label is the gold) or an unadjudicated "
            "disagreement (excluded from the gold, never averaged)"),
    }
    manifest_path = release / MANIFEST_FILENAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    return ExportResult(
        relevance_path=relevance_path, claims_path=claims_path, dump_path=dump_path,
        manifest_path=manifest_path, manifest=manifest, incomplete=incomplete,
        row_counts={RELEVANCE_CSV_FILENAME: len(relevance), CLAIMS_CSV_FILENAME: len(claims)},
        hashes=hashes)
