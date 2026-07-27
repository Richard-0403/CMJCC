"""Unit tests for evaluation statistics.

The analysis unit is the scenario: ``compare`` pairs on ``scenario_id`` for every
metric, including binary ``task_success`` (repeats are collapsed to one binary per
scenario by majority vote before pairing). Fixtures below therefore use several
repeats per scenario and expect ``n_pairs`` to equal the scenario count, never the
run count (R6.3, R6.4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jobrec_eval.statistics import (
    aggregate_scenario_success,
    compare,
    holm,
    mcnemar,
    paired_bootstrap_ci,
    rank_biserial,
)

# scenario -> variant -> task_success per repeat (3 repeats each, majority vote in
# comments). Scenario-level binaries: full = 1,1,0,1 / no_memory = 0,1,1,0
# => discordant pairs = 3 (full_only=2 on s1/s4, other_only=1 on s3).
_SUCCESS_BY_REPEAT = {
    "s1": {"full": [1, 1, 1], "no_memory": [0, 0, 1]},  # 1 vs 0
    "s2": {"full": [1, 1, 0], "no_memory": [1, 1, 1]},  # 1 vs 1
    "s3": {"full": [0, 0, 0], "no_memory": [0, 1, 1]},  # 0 vs 1
    "s4": {"full": [1, 1, 0], "no_memory": [0, 0, 0]},  # 1 vs 0
}
_NDCG_BY_VARIANT = {"full": 0.8, "no_memory": 0.5}


def _run_metrics(success_by_repeat: dict | None = None) -> pd.DataFrame:
    """Run-level frame: one row per scenario x variant x repeat."""
    rows = []
    for scenario_id, variants in (success_by_repeat or _SUCCESS_BY_REPEAT).items():
        for variant, successes in variants.items():
            for repeat_index, success in enumerate(successes):
                rows.append({
                    "run_id": f"{scenario_id}-{variant}-{repeat_index}",
                    "scenario_id": scenario_id,
                    "variant": variant,
                    "repeat_index": repeat_index,
                    "task_success": success,
                    "ndcg_at_5": _NDCG_BY_VARIANT[variant],
                })
    return pd.DataFrame(rows)


def _scenario_variant(run_metrics: pd.DataFrame) -> pd.DataFrame:
    """Average repeats within each scenario x variant (the paired unit)."""
    return (run_metrics
            .groupby(["scenario_id", "variant"], as_index=False)[["task_success", "ndcg_at_5"]]
            .mean())


def test_bootstrap_seed_reproducible():
    diffs = np.array([0.1, -0.05, 0.2, 0.0, 0.15, -0.1, 0.3])
    a = paired_bootstrap_ci(diffs, iterations=2000, seed=2026)
    b = paired_bootstrap_ci(diffs, iterations=2000, seed=2026)
    assert a == b
    # mean diff is exact
    assert a[0] == np.mean(diffs)


def test_bootstrap_ci_ordering():
    diffs = np.array([0.2, 0.3, 0.25, 0.4, 0.35])
    mean, lo, hi = paired_bootstrap_ci(diffs, iterations=2000, seed=1)
    assert lo <= mean <= hi
    assert lo > 0  # all-positive differences


def test_mcnemar_discordant_counts():
    full = np.array([1, 1, 0, 1])
    other = np.array([0, 1, 0, 0])
    mc = mcnemar(full, other)
    assert mc["full_only"] == 2  # indices 0 and 3
    assert mc["other_only"] == 0
    assert mc["n_discordant"] == 2


def test_holm_monotone_and_scaled():
    adj = holm([0.01, 0.04, 0.2])
    # sorted asc: 0.01*3=0.03, 0.04*2=0.08, 0.2*1=0.2 (monotone non-decreasing)
    assert adj[0] == 0.03
    assert adj[1] == 0.08
    assert adj[2] == 0.2
    assert adj[0] <= adj[1] <= adj[2]


def test_holm_passes_none_through():
    adj = holm([0.01, None, 0.04])
    assert adj[1] is None


def test_rank_biserial():
    assert rank_biserial(np.array([1, 1, -1, 0])) == (2 - 1) / 3
    assert rank_biserial(np.array([0, 0])) == 0.0


def test_task_success_pairs_at_scenario_level():
    """4 scenarios x 3 repeats x 2 variants = 24 runs, but only 4 paired samples."""
    rm = _run_metrics()
    sv = _scenario_variant(rm)
    assert len(rm) == 24

    res = compare(sv, rm, "task_success", "full", "no_memory", iterations=200)

    # n_pairs is the number of scenarios, not the number of runs (R6.3).
    assert res["n_pairs"] == 4
    assert res["n_pairs"] != len(rm[rm["variant"] == "full"])
    # Reported bookkeeping (R6.4).
    assert res["scenario_count"] == 4
    assert res["total_run_count"] == 24
    assert res["repeats_per_scenario"] == 3
    assert res["valid_pairs"] == 4
    assert res["discordant_pairs"] == 3
    # McNemar is computed on the scenario-level binaries, not the raw runs.
    assert res["mcnemar"]["full_only"] == 2
    assert res["mcnemar"]["other_only"] == 1
    assert res["mcnemar"]["n_discordant"] == 3
    assert res["effect_type"] == "rank_biserial"
    assert res["effect_size"] == pytest.approx((2 - 1) / 3)
    # Means still come from the repeat-averaged scenario values.
    assert res["base_mean"] == pytest.approx((1 + 2 / 3 + 0 + 2 / 3) / 4)
    assert res["other_mean"] == pytest.approx((1 / 3 + 1 + 2 / 3 + 0) / 4)


def test_task_success_scenario_binaries_use_majority_vote():
    rm = _run_metrics()
    full = aggregate_scenario_success(rm, "full")
    other = aggregate_scenario_success(rm, "no_memory")
    assert list(full.index) == ["s1", "s2", "s3", "s4"]
    assert list(full) == [1, 1, 0, 1]
    assert list(other) == [0, 1, 1, 0]


def test_task_success_pairs_only_scenarios_present_in_both_variants():
    success = {k: dict(v) for k, v in _SUCCESS_BY_REPEAT.items()}
    success["s5"] = {"full": [1, 1, 1]}  # no no_memory counterpart
    rm = _run_metrics(success)
    sv = _scenario_variant(rm)

    res = compare(sv, rm, "task_success", "full", "no_memory", iterations=200)

    assert res["scenario_count"] == 5
    assert res["valid_pairs"] == 4
    assert res["n_pairs"] == 4
    assert res["total_run_count"] == 27


def test_task_success_subset_restricts_scenario_pairs():
    rm = _run_metrics()
    sv = _scenario_variant(rm)

    res = compare(sv, rm, "task_success", "full", "no_memory", iterations=200,
                  subset_scenarios={"s1", "s2"})

    assert res["n_pairs"] == 2
    assert res["scenario_count"] == 2
    assert res["valid_pairs"] == 2
    assert res["total_run_count"] == 12
    assert res["discordant_pairs"] == 1  # s1 only


def test_continuous_metric_also_pairs_at_scenario_level():
    rm = _run_metrics()
    sv = _scenario_variant(rm)

    res = compare(sv, rm, "ndcg_at_5", "full", "no_memory", iterations=200)

    assert res["n_pairs"] == 4
    assert res["delta"] == pytest.approx(0.3)
    assert res["effect_type"] in {"cohens_dz", "rank_biserial"}
    # Run-level bookkeeping fields are task-success specific.
    assert "scenario_count" not in res


def test_compare_with_no_shared_scenarios_returns_empty_result():
    rm = _run_metrics()
    sv = _scenario_variant(rm)

    res = compare(sv, rm, "task_success", "full", "missing_variant", iterations=200)

    assert res["n_pairs"] == 0
    assert res["base_mean"] is None
    assert res["p_value"] is None

def _majority(successes: list[int]) -> int:
    """Scenario-level binary: strictly more successes than failures -> 1, else 0."""
    return int(sum(successes) * 2 > len(successes))


@st.composite
def _success_frames(draw) -> tuple[dict, int]:
    """Generate `{scenario: {variant: [success per repeat]}}` plus the repeat count.

    Varies the scenario count, the repeats-per-scenario, the per-repeat outcomes, and
    which variants each scenario appears under (so some scenarios are unpaired).
    """
    scenario_count = draw(st.integers(min_value=1, max_value=5))
    repeats = draw(st.integers(min_value=1, max_value=4))
    outcomes = st.lists(st.integers(min_value=0, max_value=1),
                        min_size=repeats, max_size=repeats)
    success: dict[str, dict[str, list[int]]] = {}
    for i in range(scenario_count):
        presence = draw(st.sampled_from(("both", "full_only", "no_memory_only")))
        variants: dict[str, list[int]] = {}
        if presence in ("both", "full_only"):
            variants["full"] = draw(outcomes)
        if presence in ("both", "no_memory_only"):
            variants["no_memory"] = draw(outcomes)
        success[f"s{i}"] = variants
    return success, repeats


# Feature: cmjcc-experiment-readiness, Property 8: Task-success McNemar pairs at the
# scenario level
@settings(max_examples=100, deadline=None)
@given(_success_frames())
def test_property_task_success_pairs_at_scenario_level(case: tuple[dict, int]):
    """`n_pairs` is the count of scenarios shared by both variants, never the run count.

    **Validates: Requirements 6.1, 6.3**
    """
    success, repeats = case
    rm = _run_metrics(success)
    sv = _scenario_variant(rm)
    shared = sorted(s for s, v in success.items() if {"full", "no_memory"} <= set(v))

    res = compare(sv, rm, "task_success", "full", "no_memory", iterations=100)

    if not shared:
        # Nothing pairs: no scenario carries both variants.
        assert res["n_pairs"] == 0
        assert res["p_value"] is None
        return

    # Pairing happens at the scenario level (R6.1, R6.3).
    assert res["n_pairs"] == len(shared)
    assert res["valid_pairs"] == len(shared)
    assert res["scenario_count"] == len(success)
    assert res["total_run_count"] == len(rm)
    assert res["repeats_per_scenario"] == repeats
    assert res["n_pairs"] <= res["scenario_count"]
    if repeats > 1:
        # Repeats never create extra independent pairs (R6.3).
        assert res["n_pairs"] < res["total_run_count"]
        assert res["n_pairs"] != len(rm[rm["variant"] == "full"])

    # McNemar counts come from the scenario-level binaries (majority vote per scenario).
    base_bin = {s: _majority(success[s]["full"]) for s in shared}
    other_bin = {s: _majority(success[s]["no_memory"]) for s in shared}
    expected_full_only = sum(1 for s in shared if base_bin[s] == 1 and other_bin[s] == 0)
    expected_other_only = sum(1 for s in shared if base_bin[s] == 0 and other_bin[s] == 1)

    assert res["mcnemar"]["full_only"] == expected_full_only
    assert res["mcnemar"]["other_only"] == expected_other_only
    assert res["mcnemar"]["n_discordant"] == expected_full_only + expected_other_only
    assert res["discordant_pairs"] == expected_full_only + expected_other_only
    assert res["discordant_pairs"] <= res["n_pairs"]

    # The same scenario-level binaries are what the aggregation helper reports.
    agg_base = aggregate_scenario_success(rm, "full")
    agg_other = aggregate_scenario_success(rm, "no_memory")
    assert {s: int(agg_base[s]) for s in shared} == base_bin
    assert {s: int(agg_other[s]) for s in shared} == other_bin

# Feature: cmjcc-experiment-readiness, Property 9: Deterministic repeat duplication does not
# change pairs or p-values
@settings(max_examples=100, deadline=None)
@given(_success_frames(), st.integers(min_value=2, max_value=4))
def test_property_deterministic_repeat_duplication_preserves_pairs_and_pvalues(
    case: tuple[dict, int], duplication: int
):
    """Replicating identical deterministic repeats leaves `n_pairs` and `p_value` untouched.

    Only the run-level bookkeeping (`total_run_count`, `repeats_per_scenario`) grows, because
    repeats collapse to one binary per scenario before pairing.

    **Validates: Requirements 6.2, 6.7, 6.8**
    """
    success, _ = case
    # Deterministic baseline: exactly ONE repeat per scenario x variant (R6.5).
    single = {s: {v: [_majority(o)] for v, o in variants.items()}
              for s, variants in success.items()}
    # Same deterministic outcome replicated across `duplication` repeats (R6.8).
    duplicated = {s: {v: o * duplication for v, o in variants.items()}
                  for s, variants in single.items()}

    rm_single = _run_metrics(single)
    rm_dup = _run_metrics(duplicated)
    res_single = compare(_scenario_variant(rm_single), rm_single, "task_success",
                         "full", "no_memory", iterations=200, seed=2026)
    res_dup = compare(_scenario_variant(rm_dup), rm_dup, "task_success",
                      "full", "no_memory", iterations=200, seed=2026)

    # Repeats create no extra independent paired samples (R6.2, R6.7), and duplication
    # never shrinks the p-value (R6.8).
    assert res_dup["n_pairs"] == res_single["n_pairs"]
    assert res_dup["p_value"] == res_single["p_value"]

    if res_single["n_pairs"] == 0:
        # No scenario carries both variants: duplication cannot manufacture a pair.
        assert res_dup["p_value"] is None
        return

    # Duplication adds runs and repeats, and nothing else.
    assert res_dup["total_run_count"] == res_single["total_run_count"] * duplication
    assert res_single["repeats_per_scenario"] == 1
    assert res_dup["repeats_per_scenario"] == duplication
    assert res_dup["scenario_count"] == res_single["scenario_count"]

    assert res_dup["valid_pairs"] == res_single["valid_pairs"]
    assert res_dup["discordant_pairs"] == res_single["discordant_pairs"]
    assert res_dup["mcnemar"] == res_single["mcnemar"]

    # The scenario-level binaries themselves are duplication-invariant.
    for variant in ("full", "no_memory"):
        agg_single = aggregate_scenario_success(rm_single, variant)
        agg_dup = aggregate_scenario_success(rm_dup, variant)
        assert list(agg_dup.index) == list(agg_single.index)
        assert list(agg_dup) == list(agg_single)
