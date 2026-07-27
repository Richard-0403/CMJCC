"""Unit tests for evaluation metric formulas (hand-calculated fixtures)."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jobrec_eval.metrics import dcg, mean_graded_relevance, ndcg_at_k, precision_at_k
from jobrec_eval.metrics_extra import clarification_efficiency


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


# --------------------------------------------------------------- R7.4/R7.5 efficiency scoring

# Slot universe the generators draw from; 6 slots so an "acceptable" subset of up to 3 always
# leaves at least 3 non-acceptable (Unnecessary) slots to draw wasted asks from.
_SLOTS = ("location", "salary_range", "remote_policy", "seniority", "skills", "industry")
# Realistic turn bound: the initial turn plus ExperimentConfig.max_dialogue_turns (default 6).
_MAX_TURNS = 7

_BACKGROUND_ROW = st.fixed_dictionaries({
    "acceptable": st.lists(st.sampled_from(_SLOTS), max_size=3, unique=True),
    "asked": st.lists(st.sampled_from(_SLOTS), max_size=3, unique=True),
    "expected": st.booleans(),
    "turns": st.integers(min_value=1, max_value=_MAX_TURNS),
})


@st.composite
def _skip_vs_ask_cases(draw) -> dict:
    """Generate a focal scenario answered two ways plus identical background runs.

    The focal scenario expects a clarification over a non-empty ``acceptable_slots`` set.
    Variant ``asks`` targets at least one acceptable slot (a Necessary clarification);
    variant ``skips`` targets only non-acceptable slots (or nothing at all), so it reaches a
    terminal state without the necessary question. Turn counts and wasted asks are drawn
    independently for the two variants, so the skipping run may well look "cheaper" on turns.
    """
    acceptable = draw(st.lists(st.sampled_from(_SLOTS), min_size=1, max_size=3, unique=True))
    unnecessary = [s for s in _SLOTS if s not in set(acceptable)]

    asked_necessary = draw(st.lists(st.sampled_from(acceptable), min_size=1,
                                    max_size=len(acceptable), unique=True))
    ask_waste = draw(st.lists(st.sampled_from(unnecessary), max_size=2, unique=True))
    skip_asked = draw(st.lists(st.sampled_from(unnecessary), max_size=2, unique=True))

    return {
        "acceptable": acceptable,
        "asked_necessary": asked_necessary,
        "ask_target": asked_necessary + ask_waste,
        "skip_target": skip_asked,
        "ask_turns": draw(st.integers(min_value=1, max_value=_MAX_TURNS)),
        "skip_turns": draw(st.integers(min_value=1, max_value=_MAX_TURNS)),
        # None exercises the "no turn column at all" fallback in clarification_efficiency.
        "turns_column": draw(st.sampled_from(("response_turns", "turn_count", None))),
        "background": draw(st.lists(_BACKGROUND_ROW, max_size=3)),
    }


def _efficiency_frame(case: dict) -> pd.DataFrame:
    """Build a run_metrics frame with an ``asks`` and a ``skips`` variant.

    Both variants cover the same scenarios; only the focal scenario's asked slots (and its
    turn count) differ, so the two variants are otherwise equal.
    """
    focal = (("asks", case["ask_target"], case["ask_turns"]),
             ("skips", case["skip_target"], case["skip_turns"]))
    rows = []
    for variant, target, turns in focal:
        rows.append({
            "run_id": f"focal-{variant}", "scenario_id": "focal", "variant": variant,
            "acceptable_slots": ";".join(case["acceptable"]),
            "clarification_target": ";".join(target),
            "clarification_expected": True,
            "_turns": turns,
        })
        for i, bg in enumerate(case["background"]):
            rows.append({
                "run_id": f"bg{i}-{variant}", "scenario_id": f"bg{i}", "variant": variant,
                "acceptable_slots": ";".join(bg["acceptable"]),
                "clarification_target": ";".join(bg["asked"]),
                "clarification_expected": bg["expected"],
                "_turns": bg["turns"],
            })
    frame = pd.DataFrame(rows)
    turns_column = case["turns_column"]
    if turns_column is not None:
        frame[turns_column] = frame["_turns"]
    return frame.drop(columns=["_turns"])


# Feature: cmjcc-experiment-readiness, Property 12: Skipping a necessary clarification is
# never scored more efficient
@settings(max_examples=100)
@given(_skip_vs_ask_cases())
def test_property_skipping_necessary_clarification_is_never_more_efficient(case: dict):
    """A run that skips a Necessary clarification never out-scores one that asked it.

    Necessary vs Unnecessary is decided by membership in ``acceptable_slots`` (R7.4), and the
    skip penalty dominates any turn / wasted-ask advantage, so the skipping variant scores
    strictly worse even when it used fewer turns (R7.5).

    **Validates: Requirements 7.4, 7.5**
    """
    result = clarification_efficiency(_efficiency_frame(case)).set_index("variant")
    asks, skips = result.loc["asks"], result.loc["skips"]

    # R7.5: skipping is never more efficient -- and here strictly less, since the penalty
    # outweighs any turn-count or wasted-ask advantage the skipping run may hold.
    assert skips["efficiency_score"] <= asks["efficiency_score"]
    assert skips["efficiency_score"] < asks["efficiency_score"]

    # R7.4: the focal scenario's necessary slots are credited to the asking run only, and the
    # skipping run is the one recorded as having missed a necessary clarification.
    assert asks["runs"] == skips["runs"] == 1 + len(case["background"])
    assert asks["necessary_asked"] == skips["necessary_asked"] + len(case["asked_necessary"])
    assert skips["necessary_missed"] == asks["necessary_missed"] + 1
