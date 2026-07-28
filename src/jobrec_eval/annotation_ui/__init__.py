"""Headless data layer for the multi-rater human annotation tool (checklist items 10/11).

Checklist item 10 (human relevance labelling) and item 11 (human explanation-grounding
labelling) both require the same process: export the unique units of work, have TWO raters
label each unit independently, measure agreement, adjudicate the disagreements and hand the
adjudicated labels back to :mod:`jobrec_eval.annotation`. This package implements that
process WITHOUT any web layer, so every step is testable from a plain pytest run and the UI
is a thin presentation layer over the modules here:

- :mod:`~jobrec_eval.annotation_ui.loader` turns real run bundles into deduplicated,
  rater-facing annotation items (and keeps the oracle grade / validator verdict out of
  them -- see :data:`~jobrec_eval.annotation_ui.store.BLINDED_FIELD_NAMES`),
- :mod:`~jobrec_eval.annotation_ui.assignment` allocates exactly two distinct raters per
  item from a seed, reproducibly,
- :mod:`~jobrec_eval.annotation_ui.store` persists raters, items, assignments,
  annotations and adjudications in one SQLite file, and is the layer that enforces
  per-rater isolation and blinding,
- :mod:`~jobrec_eval.annotation_ui.export` writes the two CSVs
  :mod:`jobrec_eval.annotation` already consumes, plus an append-only JSONL dump and a
  manifest for ``final_release/human_annotations/``.

Why a separate SQLite file and not the experiment PostgreSQL database: the PostgreSQL
schema is version-tracked and frozen for the experiment (checklist item 17), while
annotation is research-PROCESS data produced after the runs. Writing it into the frozen
schema would either require a migration inside the freeze or store research bookkeeping in
tables the replay checker verifies. One self-contained file under the annotation output
directory can be archived, hashed and shipped in the reproduction package as-is.

No human label is ever fabricated here. Everything in this package moves labels that a
real rater typed; the tests use obviously-synthetic rater ids (``SYNTHETIC-*``) so a test
fixture can never be mistaken for a collected label.
"""

from __future__ import annotations

#: Bumped when the SQLite schema changes shape. Recorded in the store's ``meta`` table and
#: in ``annotation_manifest.json`` so an archived annotation file states which layout it was
#: written with.
ANNOTATION_UI_VERSION = "1.0.0"
