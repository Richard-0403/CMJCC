"""Human annotation support, adjudication and inter-rater / oracle-vs-human agreement.

No human labels are fabricated. This module:
1. Exports annotation templates (relevance + claim) pre-filled with the
   automatic values, with empty rater and adjudication columns, for the pairs
   actually returned to candidates (the set that matters for the reported metrics).
2. If human label files are present, computes weighted Cohen's kappa (quadratic)
   for relevance and Cohen's kappa for claims, plus oracle-vs-human and
   validator-vs-human agreement against the ADJUDICATED human gold.
3. Loads an adjudicated human relevance file into the same label table shape the
   automatic oracle produces, so :class:`~jobrec_eval.metrics.MetricsComputer` can
   consume either one without knowing which it got (checklist item 10).

Human files (optional), as exported by the annotation tool:

- ``evaluation/data/relevance_labels_human.csv`` — ``scenario_id, job_id, rater_1,
  rater_2, adjudicated`` (relevance 0-3), optional ``notes``.
- ``evaluation/data/claim_annotations_human.csv`` — ``run_id, claim_id, rater_1,
  rater_2, validator, adjudicated`` (1 = supported, 0 = unsupported).

Adjudication rule (one rule, used by every function here):

- an ``adjudicated`` value on the row IS the human gold for that row;
- otherwise, two raters that agree are their own gold (nothing to adjudicate);
- otherwise the row is **unadjudicated**. It is reported as such (never averaged)
  and excluded from the human gold, so a disagreement cannot be published as if a
  human had resolved it.

The historical ``round((rater_1 + rater_2) / 2)`` averaging survives only as a
documented FALLBACK for a legacy file that carries no ``adjudicated`` column at
all, and only for the agreement statistics. The returned dicts report which path
was taken in ``adjudication_source`` so the report can never present the
heuristic as adjudicated, and the fallback is never used to build the label table
that produces the ranking metrics.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

try:
    from sklearn.metrics import cohen_kappa_score
except Exception:  # pragma: no cover
    cohen_kappa_score = None

#: Column the annotation tool writes with the adjudicated verdict for a row.
ADJUDICATED_COLUMN = "adjudicated"

#: ``adjudication_source`` value: the human gold came from the ``adjudicated`` column
#: (plus rows where both raters already agreed).
ADJUDICATION_COLUMN = "adjudicated_column"

#: ``adjudication_source`` value: the file carried no ``adjudicated`` column, so rater
#: disagreements were resolved by the legacy rounded-mean heuristic. Reported so the
#: report cannot present a heuristic as adjudication.
ADJUDICATION_ROUNDED_MEAN = "rounded_rater_mean_fallback"

#: ``rater_id`` stamped onto a human relevance label table, mirroring the oracle's
#: ``auto_oracle`` (:func:`jobrec_eval.relevance.grade_catalog`).
HUMAN_RATER_ID = "human_adjudicated"

#: Columns of a relevance label table as :func:`jobrec_eval.relevance.grade_lookup` and
#: :func:`jobrec_eval.relevance.ideal_grades` consume it. The oracle table carries these
#: plus its own diagnostic columns (``hard_violation_observed``, ``role_fit``,
#: ``skill_fit``, ``oracle_version``); human raters produce none of those, so the human
#: table omits them rather than inventing values, and nothing downstream reads them.
RELEVANCE_LABEL_COLUMNS = ["scenario_id", "job_id", "rater_id", "relevance_grade"]

#: Valid relevance grades (:mod:`jobrec_eval.relevance` grade definition).
_MIN_GRADE, _MAX_GRADE = 0, 3


class MissingAdjudicatedLabelsError(RuntimeError):
    """No adjudicated human labels are available where the caller requires them.

    Raised instead of silently falling back to the automatic oracle: publishing oracle
    numbers under a human-labels heading would misreport the construct being measured.
    """


def export_relevance_template(recommendations: pd.DataFrame, labels: pd.DataFrame,
                              out_path: str | Path) -> Path:
    """Emit a relevance annotation template for the returned (scenario, job) pairs.

    The empty ``adjudicated`` column is part of the contract: it is what a completed
    two-rater adjudication fills in, and the only column
    :func:`load_adjudicated_relevance_labels` will accept as human gold for a
    disagreement.
    """
    grade = {(r.scenario_id, r.job_id): int(r.relevance_grade) for r in labels.itertuples()}
    pairs = recommendations[["scenario_id", "job_id"]].drop_duplicates()
    rows = []
    for r in pairs.itertuples():
        rows.append({
            "scenario_id": r.scenario_id, "job_id": r.job_id,
            "oracle_grade": grade.get((r.scenario_id, r.job_id), ""),
            "rater_1": "", "rater_2": "", ADJUDICATED_COLUMN: "", "notes": "",
        })
    df = pd.DataFrame(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def export_claim_template(claims: pd.DataFrame, out_path: str | Path) -> Path:
    """Emit a claim annotation template (validator pre-fill + empty rater/adjudication cols)."""
    rows = []
    for c in claims.itertuples():
        rows.append({
            "run_id": c.run_id, "claim_id": c.claim_id, "claim_type": c.claim_type,
            # The PROPOSITION, alongside the type. A rater judging "is this supported" needs
            # to know what was asserted and about which field and job; the claim type alone
            # ("ranking_reason") does not say. Blank for a pre-P0-4 bundle.
            "predicate": getattr(c, "predicate", None),
            "claim_field": getattr(c, "claim_field", None),
            "claim_job_id": getattr(c, "claim_job_id", None),
            "validator": c.supported_binary, "rater_1": "", "rater_2": "",
            ADJUDICATED_COLUMN: "", "notes": "",
        })
    df = pd.DataFrame(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def _kappa(a, b, weights=None):
    if cohen_kappa_score is None or len(a) == 0:
        return None
    try:
        return float(cohen_kappa_score(a, b, weights=weights))
    except Exception:
        return None


# ------------------------------------------------------------------ adjudication
@dataclass(frozen=True)
class _GoldLabels:
    """Per-row human gold plus the bookkeeping the report needs to qualify it."""

    #: Gold value per input row; ``None`` where the row is unadjudicated.
    values: list[int | None]
    #: Which path produced the gold: :data:`ADJUDICATION_COLUMN` or
    #: :data:`ADJUDICATION_ROUNDED_MEAN`.
    source: str
    #: Rows resolved by an explicit ``adjudicated`` value.
    n_adjudicated: int
    #: Rows needing no adjudication because both raters agreed.
    n_concordant: int
    #: Rows where the raters disagreed and no ``adjudicated`` value exists. Excluded
    #: from the gold under :data:`ADJUDICATION_COLUMN`; resolved by the rounded-mean
    #: heuristic (and still counted here) under :data:`ADJUDICATION_ROUNDED_MEAN`.
    n_unadjudicated: int


def _adjudicated_values(frame: pd.DataFrame) -> list[int | None] | None:
    """The ``adjudicated`` column as ints, or ``None`` when the file has no such column.

    A blank cell reads as ``None`` (not adjudicated yet), which is what keeps a
    half-finished adjudication from being counted as complete.
    """
    if ADJUDICATED_COLUMN not in frame.columns:
        return None
    coerced = pd.to_numeric(frame[ADJUDICATED_COLUMN], errors="coerce")
    return [None if pd.isna(v) else int(v) for v in coerced]


def _gold_labels(rater_1: list[int], rater_2: list[int],
                 adjudicated: list[int | None] | None) -> _GoldLabels:
    """Resolve the human gold per row under the module's single adjudication rule.

    ``adjudicated is None`` means the file carried no ``adjudicated`` column; only then
    is the legacy ``round((rater_1 + rater_2) / 2)`` heuristic applied, and the result is
    labelled :data:`ADJUDICATION_ROUNDED_MEAN` so no caller can mistake it for
    adjudication. With the column present, an unadjudicated disagreement yields ``None``
    and is counted, never averaged.
    """
    values: list[int | None] = []
    n_adjudicated = n_concordant = n_unadjudicated = 0
    for index, (a, b) in enumerate(zip(rater_1, rater_2, strict=True)):
        verdict = adjudicated[index] if adjudicated is not None else None
        if verdict is not None:
            values.append(int(verdict))
            n_adjudicated += 1
        elif a == b:
            values.append(int(a))
            n_concordant += 1
        else:
            n_unadjudicated += 1
            # Legacy files only: documented averaging fallback, flagged in the source.
            values.append(int(round((a + b) / 2)) if adjudicated is None else None)
    return _GoldLabels(
        values=values,
        source=(ADJUDICATION_COLUMN if adjudicated is not None else ADJUDICATION_ROUNDED_MEAN),
        n_adjudicated=n_adjudicated, n_concordant=n_concordant,
        n_unadjudicated=n_unadjudicated,
    )


def _paired(left: list, right: list[int | None]) -> tuple[list, list]:
    """The rows where both sides carry a value, so kappa is computed on real pairs."""
    pairs = [(a, b) for a, b in zip(left, right, strict=True)
             if a is not None and b is not None and not pd.isna(a)]
    return [a for a, _ in pairs], [b for _, b in pairs]


def _gold_report(gold: _GoldLabels, n_gold_items: int) -> dict:
    """The adjudication-provenance block shared by both agreement dicts."""
    return {
        "adjudication_source": gold.source,
        "n_adjudicated": gold.n_adjudicated,
        "n_rater_concordant": gold.n_concordant,
        "unadjudicated_disagreements": gold.n_unadjudicated,
        "n_gold_items": n_gold_items,
    }


def relevance_agreement(human_path: str | Path, oracle_labels: pd.DataFrame) -> dict | None:
    """Weighted kappa between two human raters and oracle-vs-ADJUDICATED-human agreement.

    The human side of ``oracle_vs_human_weighted_kappa`` is the adjudicated gold (see the
    module docstring), computed over the rows that carry both an oracle grade and a gold
    value. ``adjudication_source`` states whether that gold came from the ``adjudicated``
    column or from the legacy rounded-mean fallback, and
    ``unadjudicated_disagreements`` counts the rows the two raters disagreed on with no
    adjudicated verdict (checklist item 10).
    """
    path = Path(human_path)
    if not path.exists():
        return None
    h = pd.read_csv(path)
    h = h.dropna(subset=["rater_1", "rater_2"])
    if h.empty:
        return None
    r1 = h["rater_1"].astype(int).tolist()
    r2 = h["rater_2"].astype(int).tolist()
    gold = _gold_labels(r1, r2, _adjudicated_values(h))
    grade = {(r.scenario_id, r.job_id): int(r.relevance_grade) for r in oracle_labels.itertuples()}
    oracle = [grade.get((row.scenario_id, row.job_id)) for row in h.itertuples()]
    oracle_paired, human_paired = _paired(oracle, gold.values)
    return {
        "n_items": len(h),
        "raw_agreement_raters": float((pd.Series(r1) == pd.Series(r2)).mean()),
        "weighted_kappa_raters": _kappa(r1, r2, weights="quadratic"),
        "oracle_vs_human_weighted_kappa": _kappa(oracle_paired, human_paired,
                                                 weights="quadratic"),
        **_gold_report(gold, len(human_paired)),
        "human_label_path": str(path),
    }


def claim_agreement(human_path: str | Path) -> dict | None:
    """Cohen's kappa between two claim raters and validator-vs-ADJUDICATED-human agreement.

    Same adjudication rule as :func:`relevance_agreement`: the ``adjudicated`` column is
    the gold when present, concordant raters are their own gold, and an unadjudicated
    disagreement is counted and excluded rather than averaged.
    """
    path = Path(human_path)
    if not path.exists():
        return None
    h = pd.read_csv(path).dropna(subset=["rater_1", "rater_2"])
    if h.empty:
        return None
    r1 = h["rater_1"].astype(int).tolist()
    r2 = h["rater_2"].astype(int).tolist()
    gold = _gold_labels(r1, r2, _adjudicated_values(h))
    out = {"n_items": len(h),
           "raw_agreement": float((pd.Series(r1) == pd.Series(r2)).mean()),
           "cohens_kappa": _kappa(r1, r2)}
    n_gold = sum(1 for v in gold.values if v is not None)
    if "validator" in h.columns:
        validator = pd.to_numeric(h["validator"], errors="coerce").tolist()
        validator_paired, human_paired = _paired(validator, gold.values)
        out["validator_vs_human_kappa"] = _kappa([int(v) for v in validator_paired],
                                                 human_paired)
        n_gold = len(human_paired)
    out.update(_gold_report(gold, n_gold))
    out["human_label_path"] = str(path)
    return out


# ------------------------------------------------- adjudicated human label table
@dataclass(frozen=True)
class AdjudicatedRelevanceLabels:
    """An adjudicated human relevance table plus the provenance of the file behind it."""

    #: Label table in the shape :func:`jobrec_eval.relevance.grade_lookup` /
    #: :func:`jobrec_eval.relevance.ideal_grades` consume (:data:`RELEVANCE_LABEL_COLUMNS`).
    labels: pd.DataFrame
    #: Path, content hash and per-row tallies, so a reader can tell WHICH labels
    #: produced the numbers.
    provenance: dict


def label_file_provenance(path: str | Path) -> dict:
    """Path, SHA-256 and byte size of a label file, for the analysis plan."""
    p = Path(path)
    data = p.read_bytes()
    return {"path": str(p), "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def load_adjudicated_relevance_labels(
    human_path: str | Path,
) -> AdjudicatedRelevanceLabels | None:
    """Load an adjudicated human relevance file as a drop-in relevance label table.

    The result carries exactly :data:`RELEVANCE_LABEL_COLUMNS` with the same dtypes the
    oracle table uses (``scenario_id``/``job_id``/``rater_id`` as strings,
    ``relevance_grade`` as ``int``), so :class:`~jobrec_eval.metrics.MetricsComputer`
    consumes it without special-casing and every metric derived from its grade lookup
    (NDCG@5, Precision@5, mean graded relevance) is recomputed from human labels.

    A row contributes its adjudicated grade, or the raters' shared grade when they
    agreed. Rows where the raters disagreed with no adjudicated verdict are DROPPED and
    counted in the provenance: the rounded-mean heuristic is never used to build the
    table the published ranking metrics come from.

    Returns ``None`` when the file does not exist, carries no :data:`ADJUDICATED_COLUMN`,
    or yields no graded pair -- i.e. when no adjudicated labels are available. Callers
    that require them raise :class:`MissingAdjudicatedLabelsError` rather than falling
    back to the oracle.

    Raises:
        ValueError: A grade is outside 0-3, or a ``(scenario_id, job_id)`` pair is
            labelled more than once, so the file cannot be read unambiguously.
    """
    path = Path(human_path)
    if not path.is_file():
        return None
    raw = pd.read_csv(path)
    rows_in_file = int(len(raw))
    frame = raw.dropna(subset=["rater_1", "rater_2"])
    adjudicated = _adjudicated_values(frame)
    if adjudicated is None:
        return None
    gold = _gold_labels(frame["rater_1"].astype(int).tolist(),
                        frame["rater_2"].astype(int).tolist(), adjudicated)

    resolved = pd.Series(gold.values, index=frame.index, dtype="object")
    keep = resolved.notna()
    graded, grades = frame[keep], resolved[keep].astype(int)
    if graded.empty:
        return None

    out_of_range = (grades < _MIN_GRADE) | (grades > _MAX_GRADE)
    if out_of_range.any():
        offenders = [f"({row.scenario_id}, {row.job_id})={grade}"
                     for row, grade in zip(graded[out_of_range].itertuples(),
                                           grades[out_of_range], strict=True)]
        raise ValueError(
            f"{path}: relevance grades must be {_MIN_GRADE}-{_MAX_GRADE}; got "
            + ", ".join(offenders))

    labels = pd.DataFrame({
        "scenario_id": graded["scenario_id"].astype(str),
        "job_id": graded["job_id"].astype(str),
        "rater_id": HUMAN_RATER_ID,
        "relevance_grade": grades.astype(int),
    }, columns=RELEVANCE_LABEL_COLUMNS).reset_index(drop=True)

    duplicated = labels[labels.duplicated(subset=["scenario_id", "job_id"], keep=False)]
    if len(duplicated):
        pairs = sorted({f"({r.scenario_id}, {r.job_id})" for r in duplicated.itertuples()})
        raise ValueError(
            f"{path}: every (scenario_id, job_id) pair must be labelled once; duplicated "
            + ", ".join(pairs))

    provenance = {
        **label_file_provenance(path),
        "rows_in_file": rows_in_file,
        "graded_pairs": int(len(labels)),
        "scenarios": int(labels["scenario_id"].nunique()),
        "adjudicated_pairs": gold.n_adjudicated,
        "rater_concordant_pairs": gold.n_concordant,
        "unadjudicated_disagreements_dropped": gold.n_unadjudicated,
        "adjudication_source": gold.source,
    }
    return AdjudicatedRelevanceLabels(labels=labels, provenance=provenance)
