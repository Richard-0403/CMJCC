"""Unit tests for evaluation statistics."""

from __future__ import annotations

import numpy as np

from jobrec_eval.statistics import holm, mcnemar, paired_bootstrap_ci, rank_biserial


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
