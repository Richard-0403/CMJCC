"""Human annotation support and inter-rater / oracle-vs-human agreement.

No human labels are fabricated. This module:
1. Exports annotation templates (relevance + claim) pre-filled with the
   automatic values, with empty rater columns, for the pairs actually returned
   to candidates (the set that matters for the reported metrics).
2. If human label files are present, computes weighted Cohen's kappa (quadratic)
   for relevance and Cohen's kappa for claims, plus oracle-vs-human agreement.

Human files (optional):
- evaluation/data/relevance_labels_human.csv  (cols: scenario_id, job_id,
  rater_1, rater_2)
- evaluation/data/claim_annotations_human.csv (cols: run_id, claim_id,
  rater_1, rater_2)   values: supported=1 / unsupported=0
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

try:
    from sklearn.metrics import cohen_kappa_score
except Exception:  # pragma: no cover
    cohen_kappa_score = None


def export_relevance_template(recommendations: pd.DataFrame, labels: pd.DataFrame,
                              out_path: str | Path) -> Path:
    """Emit a relevance annotation template for the returned (scenario, job) pairs."""
    grade = {(r.scenario_id, r.job_id): int(r.relevance_grade) for r in labels.itertuples()}
    pairs = recommendations[["scenario_id", "job_id"]].drop_duplicates()
    rows = []
    for r in pairs.itertuples():
        rows.append({
            "scenario_id": r.scenario_id, "job_id": r.job_id,
            "oracle_grade": grade.get((r.scenario_id, r.job_id), ""),
            "rater_1": "", "rater_2": "", "notes": "",
        })
    df = pd.DataFrame(rows)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    return out


def export_claim_template(claims: pd.DataFrame, out_path: str | Path) -> Path:
    """Emit a claim annotation template (validator pre-fill + empty rater cols)."""
    rows = []
    for c in claims.itertuples():
        rows.append({
            "run_id": c.run_id, "claim_id": c.claim_id, "claim_type": c.claim_type,
            "validator": c.supported_binary, "rater_1": "", "rater_2": "", "notes": "",
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


def relevance_agreement(human_path: str | Path, oracle_labels: pd.DataFrame) -> dict | None:
    """Weighted kappa between two human raters and oracle-vs-human agreement."""
    path = Path(human_path)
    if not path.exists():
        return None
    h = pd.read_csv(path)
    h = h.dropna(subset=["rater_1", "rater_2"])
    if h.empty:
        return None
    r1 = h["rater_1"].astype(int).tolist()
    r2 = h["rater_2"].astype(int).tolist()
    grade = {(r.scenario_id, r.job_id): int(r.relevance_grade) for r in oracle_labels.itertuples()}
    oracle = [grade.get((row.scenario_id, row.job_id)) for row in h.itertuples()]
    human_adj = [round((a + b) / 2) for a, b in zip(r1, r2, strict=False)]  # simple adjudication
    return {
        "n_items": len(h),
        "raw_agreement_raters": float((pd.Series(r1) == pd.Series(r2)).mean()),
        "weighted_kappa_raters": _kappa(r1, r2, weights="quadratic"),
        "oracle_vs_human_weighted_kappa": _kappa(oracle, human_adj, weights="quadratic"),
    }


def claim_agreement(human_path: str | Path) -> dict | None:
    path = Path(human_path)
    if not path.exists():
        return None
    h = pd.read_csv(path).dropna(subset=["rater_1", "rater_2"])
    if h.empty:
        return None
    r1 = h["rater_1"].astype(int).tolist()
    r2 = h["rater_2"].astype(int).tolist()
    out = {"n_items": len(h),
           "raw_agreement": float((pd.Series(r1) == pd.Series(r2)).mean()),
           "cohens_kappa": _kappa(r1, r2)}
    if "validator" in h.columns:
        adj = [round((a + b) / 2) for a, b in zip(r1, r2, strict=False)]
        out["validator_vs_human_kappa"] = _kappa(h["validator"].astype(int).tolist(), adj)
    return out
