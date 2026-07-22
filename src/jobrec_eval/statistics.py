"""Paired statistics: bootstrap CIs, McNemar, Wilcoxon, effect sizes, Holm.

Analysis unit is the scenario. Comparisons are paired by scenario (metrics are
first averaged over repeats). Binary task-success uses run-level pairing by
(scenario, repeat_index) for McNemar. Small samples are handled by emphasising
raw differences and bootstrap CIs, not normal-theory tests.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

try:
    from scipy import stats as _scipy_stats
except Exception:  # pragma: no cover
    _scipy_stats = None


def paired_bootstrap_ci(diffs: np.ndarray, iterations: int = 5000, seed: int = 2026,
                        level: float = 0.95) -> tuple[float, float, float]:
    """Return (mean_diff, ci_low, ci_high) via percentile bootstrap over pairs."""
    diffs = np.asarray(diffs, dtype=float)
    n = len(diffs)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(iterations)
    for i in range(iterations):
        idx = rng.integers(0, n, n)
        means[i] = diffs[idx].mean()
    lo = float(np.percentile(means, (1 - level) / 2 * 100))
    hi = float(np.percentile(means, (1 + level) / 2 * 100))
    return (float(diffs.mean()), lo, hi)


def cohens_dz(diffs: np.ndarray) -> float | None:
    diffs = np.asarray(diffs, dtype=float)
    if len(diffs) < 2 or diffs.std(ddof=1) == 0:
        return None
    return float(diffs.mean() / diffs.std(ddof=1))


def wilcoxon_p(a: np.ndarray, b: np.ndarray) -> float | None:
    a, b = np.asarray(a, float), np.asarray(b, float)
    if _scipy_stats is None or len(a) < 1:
        return None
    if np.all(a - b == 0):
        return 1.0
    try:
        return float(_scipy_stats.wilcoxon(a, b, zero_method="wilcox").pvalue)
    except Exception:
        return None


def rank_biserial(diffs: np.ndarray) -> float | None:
    diffs = np.asarray(diffs, float)
    nz = diffs[diffs != 0]
    if len(nz) == 0:
        return 0.0
    return float((np.sum(nz > 0) - np.sum(nz < 0)) / len(nz))


def mcnemar(full_bin: np.ndarray, other_bin: np.ndarray) -> dict:
    """McNemar on paired binary outcomes. Returns discordant counts + p."""
    full_bin = np.asarray(full_bin, int)
    other_bin = np.asarray(other_bin, int)
    b = int(np.sum((full_bin == 1) & (other_bin == 0)))  # full only
    c = int(np.sum((full_bin == 0) & (other_bin == 1)))  # other only
    p = None
    if _scipy_stats is not None and (b + c) > 0:
        # exact binomial on discordant pairs
        p = float(_scipy_stats.binomtest(min(b, c), b + c, 0.5).pvalue)
    return {"full_only": b, "other_only": c, "n_discordant": b + c, "p_value": p}


def holm(pvalues: list[float | None]) -> list[float | None]:
    """Holm-Bonferroni adjustment; None p-values pass through."""
    idx = [i for i, p in enumerate(pvalues) if p is not None]
    m = len(idx)
    adjusted: list[float | None] = list(pvalues)
    ordered = sorted(idx, key=lambda i: pvalues[i])
    prev = 0.0
    for rank, i in enumerate(ordered):
        adj = min(1.0, (m - rank) * pvalues[i])
        adj = max(adj, prev)
        adjusted[i] = adj
        prev = adj
    return adjusted


def _paired(scenario_variant: pd.DataFrame, metric: str, base: str, other: str):
    piv = scenario_variant.pivot_table(index="scenario_id", columns="variant", values=metric)
    if base not in piv.columns or other not in piv.columns:
        return np.array([]), np.array([])
    both = piv[[base, other]].dropna()
    return both[base].to_numpy(), both[other].to_numpy()


def compare(scenario_variant: pd.DataFrame, run_metrics: pd.DataFrame, metric: str,
            base: str, other: str, iterations: int = 5000, seed: int = 2026,
            subset_scenarios: set[str] | None = None) -> dict:
    """Full vs `other` on `metric`, paired by scenario. Returns a result row."""
    sv = scenario_variant
    if subset_scenarios is not None:
        sv = sv[sv["scenario_id"].isin(subset_scenarios)]
    base_vals, other_vals = _paired(sv, metric, base, other)
    n = len(base_vals)
    result = {
        "metric": metric, "base": base, "other": other, "n_pairs": n,
        "base_mean": float(np.mean(base_vals)) if n else None,
        "other_mean": float(np.mean(other_vals)) if n else None,
        "delta": None, "ci_low": None, "ci_high": None,
        "p_value": None, "effect_size": None, "effect_type": None,
    }
    if n == 0:
        return result
    diffs = base_vals - other_vals
    mean_diff, lo, hi = paired_bootstrap_ci(diffs, iterations, seed)
    result.update({"delta": mean_diff, "ci_low": lo, "ci_high": hi})

    if metric == "task_success":
        # McNemar on run-level binary pairing by (scenario, repeat_index)
        piv = run_metrics
        if subset_scenarios is not None:
            piv = piv[piv["scenario_id"].isin(subset_scenarios)]
        wide = piv.pivot_table(index=["scenario_id", "repeat_index"], columns="variant",
                               values="task_success")
        if base in wide.columns and other in wide.columns:
            w = wide[[base, other]].dropna()
            mc = mcnemar(w[base].to_numpy(), w[other].to_numpy())
            result["p_value"] = mc["p_value"]
            result["effect_size"] = rank_biserial(diffs)
            result["effect_type"] = "rank_biserial"
            result["mcnemar"] = mc
    else:
        result["p_value"] = wilcoxon_p(base_vals, other_vals)
        dz = cohens_dz(diffs)
        if dz is not None:
            result["effect_size"] = dz
            result["effect_type"] = "cohens_dz"
        else:
            result["effect_size"] = rank_biserial(diffs)
            result["effect_type"] = "rank_biserial"
    return result


def contribution_table(scenario_variant: pd.DataFrame, run_metrics: pd.DataFrame,
                       metrics: list[str], other: str, subsets: dict[str, set[str] | None],
                       iterations: int = 5000, seed: int = 2026) -> pd.DataFrame:
    """Build a Δ table (full vs `other`) across metrics and scenario subsets."""
    rows = []
    for subset_name, subset in subsets.items():
        pvals_idx = []
        for metric in metrics:
            r = compare(scenario_variant, run_metrics, metric, "full", other, iterations, seed, subset)
            r["subset"] = subset_name
            rows.append(r)
            pvals_idx.append(len(rows) - 1)
        # Holm within each subset across metrics
        adj = holm([rows[i]["p_value"] for i in pvals_idx])
        for j, i in enumerate(pvals_idx):
            rows[i]["p_value_holm"] = adj[j]
    return pd.DataFrame(rows)
