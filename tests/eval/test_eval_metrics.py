"""Unit tests for evaluation metric formulas (hand-calculated fixtures)."""

from __future__ import annotations

import math

import pytest

from jobrec_eval.metrics import dcg, mean_graded_relevance, ndcg_at_k, precision_at_k


def test_dcg_hand_calculated():
    # grades [3,2,0,1,0]
    expected = 7 / math.log2(2) + 3 / math.log2(3) + 0 + 1 / math.log2(5) + 0
    assert dcg([3, 2, 0, 1, 0]) == pytest.approx(expected, rel=1e-9)
    assert dcg([3, 2, 0, 1, 0]) == pytest.approx(9.3234659, rel=1e-6)


def test_ndcg_hand_calculated():
    ndcg = ndcg_at_k([3, 2, 0, 1, 0], [3, 2, 1, 0, 0], k=5)
    assert ndcg == pytest.approx(0.9926198, rel=1e-6)


def test_ndcg_idcg_zero_is_none():
    # no relevant items exist -> N/A, never silently 0
    assert ndcg_at_k([0, 0, 0], [0, 0, 0], k=5) is None


def test_ndcg_perfect_ranking():
    assert ndcg_at_k([3, 2, 1], [3, 2, 1], k=5) == pytest.approx(1.0)


def test_precision_denominator_uses_returned():
    # returns fewer than k -> denominator is returned count
    assert precision_at_k([3, 2, 0], k=5, threshold=2, returned=3) == pytest.approx(2 / 3)
    assert precision_at_k([3, 2, 0, 1, 0], k=5, threshold=2, returned=5) == pytest.approx(2 / 5)


def test_precision_empty_returns_none():
    assert precision_at_k([], k=5, threshold=2, returned=0) is None


def test_mean_graded_relevance():
    assert mean_graded_relevance([3, 2, 0, 1]) == pytest.approx(1.5)
    assert mean_graded_relevance([]) is None
