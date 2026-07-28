"""SQLite store for the annotation process: raters, items, assignments, labels, adjudication.

One file (:data:`DB_FILENAME`) under the annotation output directory holds the whole
annotation session. WAL journalling plus a busy timeout is what lets two raters work at the
same time through separate processes (the UI agent will serve one connection per request)
without a writer blocking a reader or a concurrent write failing outright.

This is deliberately NOT the experiment PostgreSQL database. That schema is version-tracked
and frozen with the experiment; annotation is research-process data generated afterwards, and
a self-contained file can be hashed and archived into ``final_release/human_annotations/``
without touching the frozen schema or the replay checker.

Two invariants are enforced HERE, in the data layer, rather than in the UI, because a UI bug
must not be able to break them (checklist items 10/11 both rest on them):

1. **Rater isolation.** Every rater-facing method takes a ``rater_id`` and filters on it.
   :meth:`AnnotationStore.upsert_annotation` refuses an ``(item_key, rater_id)`` pair that
   was not assigned, and the annotations table is keyed on that pair, so one rater's write
   can only ever touch their own row. There is no method that takes one rater's id and
   returns another rater's label -- independence of the two raters is a precondition of
   Cohen's kappa, so a leak would invalidate the reported agreement, not merely look untidy.
2. **Blinding.** The rater-facing payload and the analysis-side values live in two separate
   columns. Rater-facing reads never SELECT the analysis column, the rater-facing dataclass
   has no field to carry it, and :meth:`AnnotationStore.add_items` rejects a payload that
   contains a blinded key anywhere in its structure
   (:data:`BLINDED_FIELD_NAMES`). Showing a rater the oracle grade or the validator verdict
   would turn an independent human judgement into agreement with the machine.

``duration_ms`` is recorded per annotation because annotation effort is reportable: the
thesis states that two raters labelled N items, and the median time per item is what makes
that claim checkable.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from . import ANNOTATION_UI_VERSION

#: Single-file store, created inside the annotation output directory.
DB_FILENAME = "annotation.sqlite3"

#: Schema layout version, recorded in ``meta``.
SCHEMA_VERSION = "1"

#: Item kinds. ``relevance`` items are graded 0-3 (checklist item 10); ``claim`` items are
#: judged supported/unsupported as 1/0 (checklist item 11).
KIND_RELEVANCE = "relevance"
KIND_CLAIM = "claim"
KINDS = (KIND_RELEVANCE, KIND_CLAIM)

#: Valid labels per kind, enforced on write so an out-of-range value can never reach the
#: export (``jobrec_eval.annotation.load_adjudicated_relevance_labels`` raises on a grade
#: outside 0-3, and refusing it at the source is cheaper than discovering it at export time).
LABEL_RANGES: dict[str, tuple[int, ...]] = {
    KIND_RELEVANCE: (0, 1, 2, 3),
    KIND_CLAIM: (0, 1),
}

#: Exactly two rater slots per item. See :mod:`~jobrec_eval.annotation_ui.assignment` for
#: why the count is fixed at two.
SLOTS = (1, 2)

#: Keys that must never appear in a rater-facing payload: the automatic oracle's grade and
#: the claim validator's verdict, under every spelling used in this repository
#: (``jobrec_eval.relevance``, ``jobrec_eval.loaders.normalize`` and
#: ``jobrec.domain.recommendation.ResponseClaim``). :meth:`AnnotationStore.add_items` walks
#: the payload and refuses any of them, so blinding does not depend on the caller being
#: careful.
BLINDED_FIELD_NAMES = frozenset({
    "oracle_grade",
    "oracle_version",
    "relevance_grade",
    "support_status",
    "supported_binary",
    "validator",
    "validator_supported_binary",
    "validator_support_status",
})

# ------------------------------------------------------------------------- meta keys
META_EXPERIMENT_ID = "experiment_id"
META_EXPERIMENT_DIR = "experiment_dir"
META_SCENARIOS_PATH = "scenarios_path"
META_CATALOG_PATH = "catalog_path"
META_ASSIGNMENT_SEED = "assignment_seed"
META_SCHEMA_VERSION = "schema_version"
META_ANNOTATION_UI_VERSION = "annotation_ui_version"
META_CREATED_AT = "created_at"

#: Busy timeout in milliseconds. A rater saving a label while another rater's page loads must
#: wait for the lock, not fail.
BUSY_TIMEOUT_MS = 5000

_SCHEMA = """
CREATE TABLE IF NOT EXISTS meta (
    key        TEXT PRIMARY KEY,
    value      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS raters (
    rater_id     TEXT PRIMARY KEY,
    display_name TEXT NOT NULL DEFAULT '',
    created_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS items (
    item_key      TEXT PRIMARY KEY,
    kind          TEXT NOT NULL CHECK (kind IN ('relevance', 'claim')),
    payload_json  TEXT NOT NULL,
    analysis_json TEXT NOT NULL,
    scenario_id   TEXT,
    job_id        TEXT,
    claim_id      TEXT,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS item_occurrences (
    item_key          TEXT NOT NULL REFERENCES items(item_key) ON DELETE CASCADE,
    run_id            TEXT NOT NULL,
    claim_id          TEXT NOT NULL,
    variant           TEXT NOT NULL DEFAULT '',
    scenario_id       TEXT NOT NULL DEFAULT '',
    validator_label   INTEGER,
    support_status    TEXT NOT NULL DEFAULT '',
    fully_resolved    INTEGER NOT NULL DEFAULT 1,
    PRIMARY KEY (item_key, run_id, claim_id)
);

CREATE TABLE IF NOT EXISTS assignments (
    item_key   TEXT NOT NULL REFERENCES items(item_key) ON DELETE CASCADE,
    rater_id   TEXT NOT NULL REFERENCES raters(rater_id),
    slot       INTEGER NOT NULL CHECK (slot IN (1, 2)),
    position   INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (item_key, slot),
    UNIQUE (item_key, rater_id)
);

CREATE TABLE IF NOT EXISTS annotations (
    item_key    TEXT NOT NULL REFERENCES items(item_key) ON DELETE CASCADE,
    rater_id    TEXT NOT NULL REFERENCES raters(rater_id),
    label       INTEGER NOT NULL,
    notes       TEXT NOT NULL DEFAULT '',
    flags       TEXT NOT NULL DEFAULT '',
    duration_ms INTEGER,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (item_key, rater_id)
);

CREATE TABLE IF NOT EXISTS adjudications (
    item_key    TEXT PRIMARY KEY REFERENCES items(item_key) ON DELETE CASCADE,
    adjudicator TEXT NOT NULL,
    final_label INTEGER NOT NULL,
    reason      TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_assignments_rater ON assignments(rater_id, position);
CREATE INDEX IF NOT EXISTS idx_annotations_item ON annotations(item_key);
CREATE INDEX IF NOT EXISTS idx_occurrences_item ON item_occurrences(item_key);
"""


class AnnotationStoreError(RuntimeError):
    """Base class for store misuse that must not be silently tolerated."""


class UnknownRaterError(AnnotationStoreError):
    """A rater id that was never registered was used."""


class NotAssignedError(AnnotationStoreError):
    """A rater touched an item they were not assigned.

    Raised for both reads and writes. This is the mechanical form of rater isolation: a
    rater can only ever address their OWN assigned items, so one rater's judgement cannot
    be seen or overwritten by another and the two label sets stay independent.
    """


class BlindingViolationError(AnnotationStoreError):
    """A rater-facing payload carried an oracle grade or validator verdict.

    Refused at write time rather than filtered, because a payload that contains the machine's
    answer is a bug in the item builder and silently stripping it would hide that bug.
    """


class InvalidLabelError(AnnotationStoreError):
    """A label outside the valid range for the item's kind."""


@dataclass(frozen=True)
class ClaimOccurrence:
    """One ``(run_id, claim_id)`` row a deduplicated claim item stands for.

    A content-addressed ``claim_id`` is shared by every run that produced the identical
    claim, so one human judgement covers all of them; the export expands the item back to
    one CSV row per occurrence. ``validator_label`` is per occurrence because the validator
    ran per run -- it is analysis-side data and never reaches a rater.
    """

    run_id: str
    claim_id: str
    variant: str = ""
    scenario_id: str = ""
    validator_label: int | None = None
    support_status: str = ""
    fully_resolved: bool = True


@dataclass(frozen=True)
class AnnotationItem:
    """An item as written into the store: rater-facing payload + analysis-side values apart.

    ``payload`` is everything a rater may see. ``analysis`` holds the blinded values (oracle
    grade, validator verdict) that the export and the oracle-vs-human comparison need. The
    two are separate fields here, separate columns in SQLite and separate return types on
    read, so there is no code path that hands ``analysis`` to a rater.
    """

    item_key: str
    kind: str
    payload: dict[str, Any]
    analysis: dict[str, Any] = field(default_factory=dict)
    scenario_id: str | None = None
    job_id: str | None = None
    claim_id: str | None = None
    occurrences: tuple[ClaimOccurrence, ...] = ()


@dataclass(frozen=True)
class RaterItem:
    """What a rater's screen gets. Has NO field for the analysis-side values, by design.

    ``label``/``notes``/``flags`` are the rater's OWN current answer (so the UI can show a
    revisable form); they are read under the same ``rater_id`` filter as everything else.
    """

    item_key: str
    kind: str
    position: int
    slot: int
    payload: dict[str, Any]
    label: int | None = None
    notes: str = ""
    flags: str = ""
    duration_ms: int | None = None

    @property
    def done(self) -> bool:
        """True when this rater has already saved a label for the item."""
        return self.label is not None


@dataclass(frozen=True)
class AnnotationRecord:
    """One rater's saved answer for one item."""

    item_key: str
    rater_id: str
    label: int
    notes: str
    flags: str
    duration_ms: int | None
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class Progress:
    """A rater's queue progress, for a progress bar and for the effort report."""

    rater_id: str
    assigned: int
    completed: int
    total_duration_ms: int
    median_duration_ms: float | None

    @property
    def remaining(self) -> int:
        return self.assigned - self.completed

    @property
    def fraction_complete(self) -> float:
        return (self.completed / self.assigned) if self.assigned else 0.0


@dataclass(frozen=True)
class Disagreement:
    """A completed item whose two raters chose different labels."""

    item_key: str
    kind: str
    slot_1_rater: str
    slot_2_rater: str
    slot_1_label: int
    slot_2_label: int
    scenario_id: str | None
    job_id: str | None
    claim_id: str | None
    adjudicated_label: int | None
    adjudicator: str | None

    @property
    def adjudicated(self) -> bool:
        return self.adjudicated_label is not None


@dataclass(frozen=True)
class Adjudication:
    """A recorded adjudication verdict for one item."""

    item_key: str
    adjudicator: str
    final_label: int
    reason: str
    created_at: str


@dataclass(frozen=True)
class ExportRecord:
    """Everything the export needs for one item. Analysis-side, never rater-facing."""

    item_key: str
    kind: str
    scenario_id: str | None
    job_id: str | None
    claim_id: str | None
    analysis: dict[str, Any]
    occurrences: tuple[ClaimOccurrence, ...]
    slot_raters: dict[int, str]
    slot_labels: dict[int, int]
    slot_notes: dict[int, str]
    slot_durations: dict[int, int | None]
    adjudication: Adjudication | None
    #: First-save and last-save stamps per slot. A revision is visible as
    #: ``created_at != updated_at``, so the archived dump shows that a rater changed their
    #: mind rather than hiding it.
    slot_created_at: dict[int, str] = field(default_factory=dict)
    slot_updated_at: dict[int, str] = field(default_factory=dict)

    @property
    def complete(self) -> bool:
        """Both slots carry a label, i.e. the item is exportable."""
        return all(slot in self.slot_labels for slot in SLOTS)

    @property
    def raters_agree(self) -> bool:
        return self.complete and self.slot_labels[1] == self.slot_labels[2]


def _utcnow() -> str:
    return datetime.now(UTC).isoformat()


def _blinded_keys_in(payload: Any) -> list[str]:
    """Blinded key names found anywhere in a nested payload, in first-seen order."""
    found: list[str] = []
    stack: list[Any] = [payload]
    while stack:
        node = stack.pop()
        if isinstance(node, Mapping):
            for key, value in node.items():
                if str(key) in BLINDED_FIELD_NAMES and key not in found:
                    found.append(str(key))
                stack.append(value)
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
    return found


def _median(values: Sequence[int]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2


class AnnotationStore:
    """The annotation session, backed by one SQLite file.

    Open with :func:`open_store` (or :meth:`open`), which applies the pragmas. Usable as a
    context manager; ``close()`` is safe to call twice.
    """

    def __init__(self, connection: sqlite3.Connection, path: Path) -> None:
        self._db = connection
        self._path = path

    # ------------------------------------------------------------------ lifecycle
    @classmethod
    def open(cls, out_dir: str | Path, *, create: bool = True) -> AnnotationStore:
        """Open (and by default create) the store under ``out_dir``.

        WAL is set so a reader is never blocked by the writer, and ``busy_timeout`` so a
        concurrent write waits instead of raising ``database is locked`` at a rater.
        """
        directory = Path(out_dir)
        path = directory / DB_FILENAME
        if not create and not path.is_file():
            raise FileNotFoundError(f"no annotation store at {path}")
        directory.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        connection.execute("PRAGMA foreign_keys=ON")
        connection.executescript(_SCHEMA)
        store = cls(connection, path)
        store._seed_version_meta()
        return store

    def _seed_version_meta(self) -> None:
        existing = self.meta()
        defaults = {
            META_SCHEMA_VERSION: SCHEMA_VERSION,
            META_ANNOTATION_UI_VERSION: ANNOTATION_UI_VERSION,
            META_CREATED_AT: _utcnow(),
        }
        self.set_meta({k: v for k, v in defaults.items() if k not in existing})

    @property
    def path(self) -> Path:
        """Filesystem path of the SQLite file (hashed into the export manifest)."""
        return self._path

    def close(self) -> None:
        try:
            self._db.close()
        except sqlite3.ProgrammingError:  # pragma: no cover - already closed
            pass

    def __enter__(self) -> AnnotationStore:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    # ---------------------------------------------------------------------- meta
    def set_meta(self, values: Mapping[str, Any]) -> None:
        """Upsert provenance keys (experiment id, input paths, assignment seed, versions)."""
        with self._db:
            self._db.executemany(
                "INSERT INTO meta(key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                [(str(k), "" if v is None else str(v)) for k, v in values.items()],
            )

    def meta(self) -> dict[str, str]:
        """Every recorded meta key, so the manifest can be rebuilt from the file alone."""
        return {row["key"]: row["value"] for row in self._db.execute("SELECT key, value FROM meta")}

    # -------------------------------------------------------------------- raters
    def register_raters(self, rater_ids: Iterable[str],
                        display_names: Mapping[str, str] | None = None) -> tuple[str, ...]:
        """Register the rater pool (idempotent); returns the ids in registration order."""
        names = display_names or {}
        ordered = tuple(dict.fromkeys(str(r) for r in rater_ids))
        if not ordered:
            raise ValueError("no rater ids given")
        now = _utcnow()
        with self._db:
            self._db.executemany(
                "INSERT INTO raters(rater_id, display_name, created_at) VALUES (?, ?, ?) "
                "ON CONFLICT(rater_id) DO UPDATE SET display_name = excluded.display_name",
                [(r, str(names.get(r, "")), now) for r in ordered],
            )
        return ordered

    def raters(self) -> tuple[str, ...]:
        """Registered rater ids, in registration order."""
        rows = self._db.execute("SELECT rater_id FROM raters ORDER BY created_at, rater_id")
        return tuple(row["rater_id"] for row in rows)

    def _require_rater(self, rater_id: str) -> str:
        row = self._db.execute("SELECT 1 FROM raters WHERE rater_id = ?", (rater_id,)).fetchone()
        if row is None:
            raise UnknownRaterError(f"rater {rater_id!r} is not registered")
        return rater_id

    # --------------------------------------------------------------------- items
    def add_items(self, items: Iterable[AnnotationItem]) -> int:
        """Insert items (idempotent on ``item_key``); returns the number written.

        Refuses a payload containing any :data:`BLINDED_FIELD_NAMES` key. That check lives
        here, in the store, so blinding holds for every writer -- the loader, a future
        re-import, or a UI-side fixture -- and not only for the one caller that remembered.
        """
        rows: list[tuple] = []
        occurrence_rows: list[tuple] = []
        now = _utcnow()
        for item in items:
            if item.kind not in KINDS:
                raise ValueError(f"unknown item kind {item.kind!r}; expected one of {KINDS}")
            leaked = _blinded_keys_in(item.payload)
            if leaked:
                raise BlindingViolationError(
                    f"item {item.item_key!r} payload carries blinded field(s) "
                    f"{', '.join(sorted(leaked))}; the oracle grade and the validator verdict "
                    f"belong in AnnotationItem.analysis, never in what a rater sees")
            rows.append((
                item.item_key, item.kind, json.dumps(item.payload, sort_keys=True, default=str),
                json.dumps(item.analysis, sort_keys=True, default=str),
                item.scenario_id, item.job_id, item.claim_id, now,
            ))
            for occurrence in item.occurrences:
                occurrence_rows.append((
                    item.item_key, occurrence.run_id, occurrence.claim_id, occurrence.variant,
                    occurrence.scenario_id, occurrence.validator_label,
                    occurrence.support_status, int(occurrence.fully_resolved),
                ))
        with self._db:
            self._db.executemany(
                "INSERT INTO items(item_key, kind, payload_json, analysis_json, scenario_id, "
                "job_id, claim_id, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(item_key) DO UPDATE SET payload_json = excluded.payload_json, "
                "analysis_json = excluded.analysis_json", rows)
            self._db.executemany(
                "INSERT INTO item_occurrences(item_key, run_id, claim_id, variant, scenario_id, "
                "validator_label, support_status, fully_resolved) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(item_key, run_id, claim_id) DO UPDATE SET "
                "validator_label = excluded.validator_label, "
                "support_status = excluded.support_status, "
                "fully_resolved = excluded.fully_resolved", occurrence_rows)
        return len(rows)

    def item_keys(self, kind: str | None = None) -> tuple[str, ...]:
        """Item keys in a stable order (the order assignment consumes)."""
        if kind is None:
            rows = self._db.execute("SELECT item_key FROM items ORDER BY item_key")
        else:
            rows = self._db.execute(
                "SELECT item_key FROM items WHERE kind = ? ORDER BY item_key", (kind,))
        return tuple(row["item_key"] for row in rows)

    def item_count(self, kind: str | None = None) -> int:
        if kind is None:
            row = self._db.execute("SELECT COUNT(*) AS n FROM items").fetchone()
        else:
            row = self._db.execute("SELECT COUNT(*) AS n FROM items WHERE kind = ?",
                                   (kind,)).fetchone()
        return int(row["n"])

    # ---------------------------------------------------------------- assignment
    def save_assignment_plan(self, plan: Any) -> int:
        """Persist an :class:`~jobrec_eval.annotation_ui.assignment.AssignmentPlan`.

        Takes the plan object (duck-typed on ``assignments`` / ``seed`` / ``rater_pool``) so
        the store does not import the assignment module and the two stay independently
        testable. The seed is written into ``meta`` because a reproducible assignment is only
        reproducible if the seed travels with the data.
        """
        now = _utcnow()
        rows = [(a.item_key, a.rater_id, int(a.slot), int(a.position), now)
                for a in plan.assignments]
        for rater_id in {r for _, r, _, _, _ in rows}:
            self._require_rater(rater_id)
        with self._db:
            self._db.executemany(
                "INSERT INTO assignments(item_key, rater_id, slot, position, created_at) "
                "VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(item_key, slot) DO UPDATE SET rater_id = excluded.rater_id, "
                "position = excluded.position", rows)
        self.set_meta({META_ASSIGNMENT_SEED: plan.seed})
        return len(rows)

    def assignment_counts(self) -> dict[str, int]:
        """Items assigned per rater, for the load-balance check and the manifest."""
        rows = self._db.execute(
            "SELECT rater_id, COUNT(*) AS n FROM assignments GROUP BY rater_id ORDER BY rater_id")
        return {row["rater_id"]: int(row["n"]) for row in rows}

    # ------------------------------------------------------------- rater-facing
    # Every query below filters on rater_id and selects payload_json only. The analysis
    # column is not named in any of them, which is what makes blinding structural rather
    # than a convention.
    _QUEUE_SQL = """
        SELECT a.item_key, a.slot, a.position, i.kind, i.payload_json,
               n.label, n.notes, n.flags, n.duration_ms
          FROM assignments a
          JOIN items i ON i.item_key = a.item_key
          LEFT JOIN annotations n ON n.item_key = a.item_key AND n.rater_id = a.rater_id
         WHERE a.rater_id = ?
    """

    def queue(self, rater_id: str, *, include_done: bool = True,
              kind: str | None = None) -> tuple[RaterItem, ...]:
        """This rater's queue in their recorded shuffle order.

        The order is the seeded per-rater shuffle stored as ``position``: two raters see the
        same items in different orders, so a systematic ordering effect (fatigue, drift)
        cannot line up between them and inflate agreement.
        """
        self._require_rater(rater_id)
        sql = self._QUEUE_SQL
        params: list[Any] = [rater_id]
        if not include_done:
            sql += " AND n.label IS NULL"
        if kind is not None:
            sql += " AND i.kind = ?"
            params.append(kind)
        sql += " ORDER BY a.position, a.item_key"
        return tuple(self._rater_item(row) for row in self._db.execute(sql, params))

    def next_item(self, rater_id: str, kind: str | None = None) -> RaterItem | None:
        """The rater's first unlabelled item, or ``None`` when their queue is done."""
        pending = self.queue(rater_id, include_done=False, kind=kind)
        return pending[0] if pending else None

    def rater_item(self, rater_id: str, item_key: str) -> RaterItem:
        """One item for one rater.

        Raises:
            NotAssignedError: The item is not on this rater's queue. This is the read side of
                rater isolation: a rater cannot fetch an item (or, therefore, the answer of
                the rater it was assigned to) that is not theirs.
        """
        self._require_rater(rater_id)
        row = self._db.execute(self._QUEUE_SQL + " AND a.item_key = ?",
                               (rater_id, item_key)).fetchone()
        if row is None:
            raise NotAssignedError(
                f"item {item_key!r} is not assigned to rater {rater_id!r}")
        return self._rater_item(row)

    @staticmethod
    def _rater_item(row: sqlite3.Row) -> RaterItem:
        return RaterItem(
            item_key=row["item_key"], kind=row["kind"], position=int(row["position"]),
            slot=int(row["slot"]), payload=json.loads(row["payload_json"]),
            label=None if row["label"] is None else int(row["label"]),
            notes=row["notes"] or "", flags=row["flags"] or "",
            duration_ms=None if row["duration_ms"] is None else int(row["duration_ms"]),
        )

    def upsert_annotation(self, item_key: str, rater_id: str, label: int, *,
                          notes: str = "", flags: str = "",
                          duration_ms: int | None = None) -> AnnotationRecord:
        """Save or revise THIS rater's label for an item they were assigned.

        Upsert, not insert: a rater may revise their own judgement any time before export,
        which is normal for a labelling pass with a written guideline. ``created_at`` is kept
        from the first save and ``updated_at`` moves, so a revision is visible in the dump.

        Raises:
            NotAssignedError: ``(item_key, rater_id)`` is not an assignment. Together with the
                ``(item_key, rater_id)`` primary key this makes overwriting another rater's
                label impossible, not merely discouraged.
            InvalidLabelError: The label is outside the item kind's range.
        """
        self._require_rater(rater_id)
        row = self._db.execute(
            "SELECT i.kind FROM assignments a JOIN items i ON i.item_key = a.item_key "
            "WHERE a.item_key = ? AND a.rater_id = ?", (item_key, rater_id)).fetchone()
        if row is None:
            raise NotAssignedError(
                f"rater {rater_id!r} is not assigned item {item_key!r}; a rater can only "
                f"write their own assigned annotations")
        kind = row["kind"]
        valid = LABEL_RANGES[kind]
        if int(label) not in valid:
            raise InvalidLabelError(
                f"label {label!r} is not valid for a {kind} item; expected one of {valid}")
        if duration_ms is not None and int(duration_ms) < 0:
            raise ValueError("duration_ms cannot be negative")
        now = _utcnow()
        with self._db:
            self._db.execute(
                "INSERT INTO annotations(item_key, rater_id, label, notes, flags, duration_ms, "
                "created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(item_key, rater_id) DO UPDATE SET label = excluded.label, "
                "notes = excluded.notes, flags = excluded.flags, "
                "duration_ms = excluded.duration_ms, updated_at = excluded.updated_at "
                "WHERE annotations.rater_id = excluded.rater_id",
                (item_key, rater_id, int(label), notes, flags,
                 None if duration_ms is None else int(duration_ms), now, now))
        saved = self.annotation(item_key, rater_id)
        assert saved is not None  # just written under the same (item_key, rater_id) key
        return saved

    def annotation(self, item_key: str, rater_id: str) -> AnnotationRecord | None:
        """THIS rater's saved answer for one item, or ``None``.

        Filtered on ``rater_id``, so the method cannot return another rater's label whatever
        the caller passes.

        Raises:
            NotAssignedError: The item is not assigned to this rater.
        """
        self._require_rater(rater_id)
        assigned = self._db.execute(
            "SELECT 1 FROM assignments WHERE item_key = ? AND rater_id = ?",
            (item_key, rater_id)).fetchone()
        if assigned is None:
            raise NotAssignedError(f"item {item_key!r} is not assigned to rater {rater_id!r}")
        row = self._db.execute(
            "SELECT item_key, rater_id, label, notes, flags, duration_ms, created_at, updated_at "
            "FROM annotations WHERE item_key = ? AND rater_id = ?",
            (item_key, rater_id)).fetchone()
        if row is None:
            return None
        return AnnotationRecord(
            item_key=row["item_key"], rater_id=row["rater_id"], label=int(row["label"]),
            notes=row["notes"] or "", flags=row["flags"] or "",
            duration_ms=None if row["duration_ms"] is None else int(row["duration_ms"]),
            created_at=row["created_at"], updated_at=row["updated_at"])

    def progress(self, rater_id: str) -> Progress:
        """Assigned/completed counts and annotation effort for one rater."""
        self._require_rater(rater_id)
        assigned = int(self._db.execute(
            "SELECT COUNT(*) AS n FROM assignments WHERE rater_id = ?",
            (rater_id,)).fetchone()["n"])
        durations = [int(row["duration_ms"]) for row in self._db.execute(
            "SELECT duration_ms FROM annotations WHERE rater_id = ? AND duration_ms IS NOT NULL",
            (rater_id,))]
        completed = int(self._db.execute(
            "SELECT COUNT(*) AS n FROM annotations WHERE rater_id = ?",
            (rater_id,)).fetchone()["n"])
        return Progress(rater_id=rater_id, assigned=assigned, completed=completed,
                        total_duration_ms=sum(durations),
                        median_duration_ms=_median(durations))

    def annotation_effort(self) -> dict[str, Any]:
        """Effort summary across all raters -- reportable annotation cost.

        The thesis claims two raters labelled N items; median seconds per item is what makes
        that claim checkable, so the numbers come from recorded ``duration_ms`` and never
        from an estimate.
        """
        durations = [int(row["duration_ms"]) for row in self._db.execute(
            "SELECT duration_ms FROM annotations WHERE duration_ms IS NOT NULL")]
        return {
            "annotations": int(self._db.execute(
                "SELECT COUNT(*) AS n FROM annotations").fetchone()["n"]),
            "timed_annotations": len(durations),
            "total_duration_ms": sum(durations),
            "median_duration_ms": _median(durations),
            "per_rater": {rater: self.progress(rater).total_duration_ms
                          for rater in self.raters()},
        }

    # ------------------------------------------------------- completion / review
    def completed_item_keys(self, kind: str | None = None) -> tuple[str, ...]:
        """Items where BOTH slots carry a label, i.e. the exportable set."""
        sql = ("SELECT a.item_key FROM assignments a "
               "JOIN annotations n ON n.item_key = a.item_key AND n.rater_id = a.rater_id "
               "JOIN items i ON i.item_key = a.item_key ")
        params: list[Any] = []
        if kind is not None:
            sql += "WHERE i.kind = ? "
            params.append(kind)
        sql += "GROUP BY a.item_key HAVING COUNT(DISTINCT a.slot) = 2 ORDER BY a.item_key"
        return tuple(row["item_key"] for row in self._db.execute(sql, params))

    def disagreements(self, kind: str | None = None, *,
                      unadjudicated_only: bool = False) -> tuple[Disagreement, ...]:
        """Completed items whose two raters chose different labels.

        This is the adjudication worklist (checklist items 10/11 both require "export
        disagreement cases" then "complete adjudication").
        """
        out: list[Disagreement] = []
        for record in self.iter_export_records(kind=kind):
            if not record.complete or record.raters_agree:
                continue
            if unadjudicated_only and record.adjudication is not None:
                continue
            out.append(Disagreement(
                item_key=record.item_key, kind=record.kind,
                slot_1_rater=record.slot_raters[1], slot_2_rater=record.slot_raters[2],
                slot_1_label=record.slot_labels[1], slot_2_label=record.slot_labels[2],
                scenario_id=record.scenario_id, job_id=record.job_id, claim_id=record.claim_id,
                adjudicated_label=(record.adjudication.final_label
                                   if record.adjudication else None),
                adjudicator=(record.adjudication.adjudicator if record.adjudication else None),
            ))
        return tuple(out)

    def record_adjudication(self, item_key: str, adjudicator: str, final_label: int,
                            reason: str = "") -> Adjudication:
        """Record the adjudicated verdict for one item.

        Only the recorded verdict may ever fill the export's ``adjudicated`` column; nothing
        here derives a verdict from the two labels.

        Raises:
            InvalidLabelError: The verdict is outside the item kind's label range.
            KeyError: No such item.
        """
        row = self._db.execute("SELECT kind FROM items WHERE item_key = ?",
                               (item_key,)).fetchone()
        if row is None:
            raise KeyError(f"no item {item_key!r}")
        valid = LABEL_RANGES[row["kind"]]
        if int(final_label) not in valid:
            raise InvalidLabelError(
                f"adjudicated label {final_label!r} is not valid for a {row['kind']} item; "
                f"expected one of {valid}")
        now = _utcnow()
        with self._db:
            self._db.execute(
                "INSERT INTO adjudications(item_key, adjudicator, final_label, reason, "
                "created_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(item_key) DO UPDATE SET adjudicator = excluded.adjudicator, "
                "final_label = excluded.final_label, reason = excluded.reason, "
                "created_at = excluded.created_at",
                (item_key, adjudicator, int(final_label), reason, now))
        return Adjudication(item_key=item_key, adjudicator=adjudicator,
                            final_label=int(final_label), reason=reason, created_at=now)

    def adjudication(self, item_key: str) -> Adjudication | None:
        row = self._db.execute(
            "SELECT item_key, adjudicator, final_label, reason, created_at "
            "FROM adjudications WHERE item_key = ?", (item_key,)).fetchone()
        if row is None:
            return None
        return Adjudication(item_key=row["item_key"], adjudicator=row["adjudicator"],
                            final_label=int(row["final_label"]), reason=row["reason"] or "",
                            created_at=row["created_at"])

    # --------------------------------------------------------------- export side
    def iter_export_records(self, kind: str | None = None) -> Iterator[ExportRecord]:
        """Every item with its slots, labels, adjudication and analysis-side values.

        The export and the analysis read this; a rater never does. It takes no ``rater_id``
        precisely because it is not a rater-facing view -- that asymmetry is the blinding
        boundary.
        """
        sql = ("SELECT item_key, kind, scenario_id, job_id, claim_id, analysis_json "
               "FROM items ")
        params: list[Any] = []
        if kind is not None:
            sql += "WHERE kind = ? "
            params.append(kind)
        sql += "ORDER BY item_key"
        for row in self._db.execute(sql, params).fetchall():
            item_key = row["item_key"]
            slot_raters: dict[int, str] = {}
            slot_labels: dict[int, int] = {}
            slot_notes: dict[int, str] = {}
            slot_durations: dict[int, int | None] = {}
            slot_created: dict[int, str] = {}
            slot_updated: dict[int, str] = {}
            for assignment in self._db.execute(
                "SELECT a.slot, a.rater_id, n.label, n.notes, n.duration_ms, n.created_at, "
                "n.updated_at FROM assignments a "
                "LEFT JOIN annotations n ON n.item_key = a.item_key AND n.rater_id = a.rater_id "
                "WHERE a.item_key = ? ORDER BY a.slot", (item_key,)):
                slot = int(assignment["slot"])
                slot_raters[slot] = assignment["rater_id"]
                if assignment["label"] is not None:
                    slot_labels[slot] = int(assignment["label"])
                    slot_notes[slot] = assignment["notes"] or ""
                    slot_durations[slot] = (None if assignment["duration_ms"] is None
                                            else int(assignment["duration_ms"]))
                    slot_created[slot] = assignment["created_at"] or ""
                    slot_updated[slot] = assignment["updated_at"] or ""
            occurrences = tuple(
                ClaimOccurrence(
                    run_id=occ["run_id"], claim_id=occ["claim_id"], variant=occ["variant"],
                    scenario_id=occ["scenario_id"],
                    validator_label=(None if occ["validator_label"] is None
                                     else int(occ["validator_label"])),
                    support_status=occ["support_status"],
                    fully_resolved=bool(occ["fully_resolved"]),
                )
                for occ in self._db.execute(
                    "SELECT run_id, claim_id, variant, scenario_id, validator_label, "
                    "support_status, fully_resolved FROM item_occurrences WHERE item_key = ? "
                    "ORDER BY run_id, claim_id", (item_key,)))
            yield ExportRecord(
                item_key=item_key, kind=row["kind"], scenario_id=row["scenario_id"],
                job_id=row["job_id"], claim_id=row["claim_id"],
                analysis=json.loads(row["analysis_json"]), occurrences=occurrences,
                slot_raters=slot_raters, slot_labels=slot_labels, slot_notes=slot_notes,
                slot_durations=slot_durations, adjudication=self.adjudication(item_key),
                slot_created_at=slot_created, slot_updated_at=slot_updated)


def open_store(out_dir: str | Path, *, create: bool = True) -> AnnotationStore:
    """Open the annotation store under ``out_dir`` (see :meth:`AnnotationStore.open`)."""
    return AnnotationStore.open(out_dir, create=create)
