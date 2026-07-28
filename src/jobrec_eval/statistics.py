"""Paired statistics: bootstrap CIs, McNemar, Wilcoxon, effect sizes, Holm.

Analysis unit is the scenario. Comparisons are paired by scenario (metrics are
first averaged over repeats). Binary task-success is also paired at the scenario
level for McNemar: repeats are collapsed to one binary per scenario (majority
vote; even-repeat ties -> 0) before pairing, so ``n_pairs`` equals the number of
scenarios rather than the number of runs. Small samples are handled by
emphasising raw differences and bootstrap CIs, not normal-theory tests.
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


def aggregate_scenario_success(run_metrics: pd.DataFrame, variant: str,
                               subset: set[str] | None = None) -> pd.Series:
    """Collapse repeats to ONE binary task-success outcome per scenario.

    For the given ``variant``, group the run-level ``task_success`` values by
    ``scenario_id`` and reduce each scenario's repeats to a single binary via
    MAJORITY VOTE: strictly more successes than failures -> 1, otherwise 0.
    Ties (only possible with an even number of repeats) resolve conservatively
    to 0 (not-success). Deterministic runs (one repeat per scenario) are
    unaffected, and duplicating deterministic repeats does not change the
    outcome (R6.2, R6.7, R6.8).

    Args:
        run_metrics: run-level DataFrame with at least ``scenario_id``,
            ``variant``, and ``task_success`` columns.
        variant: the variant to aggregate.
        subset: optional set of ``scenario_id`` values to restrict to.

    Returns:
        A ``pd.Series`` of int (0/1) indexed by ``scenario_id`` (sorted), named
        ``variant``. Empty if no matching rows are present.
    """
    df = run_metrics[run_metrics["variant"] == variant]
    if subset is not None:
        df = df[df["scenario_id"].isin(subset)]
    if len(df) == 0:
        return pd.Series(dtype=int, name=variant)
    successes = df.groupby("scenario_id")["task_success"].sum()
    counts = df.groupby("scenario_id")["task_success"].size()
    # Majority vote: successes strictly greater than failures -> 1, else 0.
    # Equivalent to successes * 2 > total; ties (== total) resolve to 0.
    binary = (successes * 2 > counts).astype(int)
    binary = binary.sort_index()
    binary.name = variant
    return binary


#: What a comparison row's ``base_mean``/``other_mean``/``delta``/CI/p/effect describe.
#: Recorded per row so a reader never has to infer it -- the two estimands are NOT
#: interchangeable and a row that silently mixed them is unreadable.
#:
#: * :data:`ESTIMAND_SCENARIO_MEAN` -- each scenario's metric averaged over its repeats,
#:   then paired across variants (continuous metrics).
#: * :data:`ESTIMAND_BINARY` -- each scenario collapsed to ONE binary success by majority
#:   vote over repeats (even-repeat ties resolve to 0), then paired. This is the
#:   pre-registered estimand for task success, and now the ONLY one its row reports.
ESTIMAND_SCENARIO_MEAN = "scenario_mean_over_repeats"
ESTIMAND_BINARY = "scenario_binary_majority_vote"


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

    if metric == "task_success":
        # Scenario-level McNemar: collapse repeats to ONE binary per scenario
        # (majority vote; even-repeat ties -> 0) BEFORE pairing, so that
        # ``n_pairs`` equals the number of scenarios present in both variants,
        # never scenario x repeat. Duplicating deterministic repeats therefore
        # cannot inflate the sample or shrink the p-value (R6.3/6.4/6.7/6.8).
        rm = run_metrics
        if subset_scenarios is not None:
            rm = rm[rm["scenario_id"].isin(subset_scenarios)]
        rm = rm[rm["variant"].isin([base, other])]
        base_bin = aggregate_scenario_success(rm, base, subset_scenarios)
        other_bin = aggregate_scenario_success(rm, other, subset_scenarios)
        # Align/pair on the intersection of scenario ids (inner join + dropna).
        paired = pd.concat([base_bin, other_bin], axis=1, join="inner").dropna()

        total_run_count = int(len(rm))
        scenario_count = int(rm["scenario_id"].nunique())
        repeats_per_scenario = (
            int(rm["repeat_index"].nunique())
            if "repeat_index" in rm.columns and total_run_count
            else 0
        )
        valid_pairs = int(len(paired))

        result["scenario_count"] = scenario_count
        result["total_run_count"] = total_run_count
        result["repeats_per_scenario"] = repeats_per_scenario
        result["valid_pairs"] = valid_pairs
        # Task-success pairs at the SCENARIO level (R6.3).
        result["n_pairs"] = valid_pairs
        result["estimand"] = ESTIMAND_BINARY
        # The FRACTIONAL view (each scenario's success RATE across repeats), kept under
        # its own names. It used to be what ``base_mean``/``other_mean``/``delta``/the CI
        # reported while the p-value, the effect size and ``n_pairs`` described the
        # collapsed binaries -- two different estimands printed as one row, differing by
        # a visible amount (0.9778 vs 1.0000 on a mean, 0.1825 vs 0.1905 on a delta) and
        # with an ``n`` that belonged to neither. The row now reports the pre-registered
        # binary estimand throughout; the rate stays available beside it.
        result["base_repeat_mean"] = float(np.mean(base_vals))
        result["other_repeat_mean"] = float(np.mean(other_vals))
        result["repeat_mean_delta"] = float(np.mean(base_vals - other_vals))

        if valid_pairs:
            base_arr = paired.iloc[:, 0].to_numpy()
            other_arr = paired.iloc[:, 1].to_numpy()
            binary_diffs = (base_arr - other_arr).astype(float)
            result["base_mean"] = float(np.mean(base_arr))
            result["other_mean"] = float(np.mean(other_arr))
            mean_diff, lo, hi = paired_bootstrap_ci(binary_diffs, iterations, seed)
            result.update({"delta": mean_diff, "ci_low": lo, "ci_high": hi})
            mc = mcnemar(base_arr, other_arr)
            result["p_value"] = mc["p_value"]
            result["effect_size"] = rank_biserial(binary_diffs)
            result["effect_type"] = "rank_biserial"
            result["mcnemar"] = mc
            result["discordant_pairs"] = mc["n_discordant"]
        else:
            # No validly paired scenario: the binary estimand is undefined, so no mean,
            # delta or CI is reported for it (the fractional columns above still are).
            result.update({"base_mean": None, "other_mean": None,
                           "delta": None, "ci_low": None, "ci_high": None})
            result["discordant_pairs"] = 0
    else:
        diffs = base_vals - other_vals
        mean_diff, lo, hi = paired_bootstrap_ci(diffs, iterations, seed)
        result.update({"delta": mean_diff, "ci_low": lo, "ci_high": hi})
        result["estimand"] = ESTIMAND_SCENARIO_MEAN
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
