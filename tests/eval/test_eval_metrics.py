"""Unit tests for evaluation metric formulas (hand-calculated fixtures)."""

from __future__ import annotations

import math

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from jobrec_eval.metrics import dcg, mean_graded_relevance, ndcg_at_k, precision_at_k
from jobrec_eval.metrics_extra import (
    clarification_efficiency,
    clarification_efficiency_per_run,
)


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

# ------------------------------------- R7.4/R7.5 unresolved dialogues are not efficiency
# A run can ask the necessary question and still never resolve the ambiguity: the dialogue
# loop may stop on ``continuation_disabled`` (the one_shot condition), ``max_turns``,
# ``cannot_answer`` or ``repeated_slot``. In every one of those cases the FINAL recorded
# response is still a ``clarification``, which is the signal the score reads -- no
# termination reason is enumerated, so a new guard is covered automatically.

def _dialogue_row(variant: str, *, scenario: str, acceptable: str, asked: str,
                  response_type: str, turns: float, expected: bool = True) -> dict:
    """One run_metrics row in the shape the pipeline records for a dialogue run."""
    return {
        "run_id": f"{scenario}-{variant}", "scenario_id": scenario, "variant": variant,
        "acceptable_slots": acceptable, "clarification_asked_slots": asked,
        "clarification_expected": expected, "response_type": response_type,
        "response_turns": turns,
    }


def _three_way_frame(*, resolved_turns: float, abandoned_turns: float,
                     skipped_turns: float) -> pd.DataFrame:
    """One clarification scenario answered three ways, one variant each.

    ``resolved`` asks the acceptable slot and ends on a recommendation; ``abandoned`` asks
    the same slot but ends with the question still pending; ``skipped`` never asks and
    guesses a recommendation instead. Turn counts are supplied independently so the
    ordering can be probed with the "worse" runs made artificially cheap.
    """
    return pd.DataFrame([
        _dialogue_row("resolved", scenario="focal", acceptable="location", asked="location",
                      response_type="recommendation", turns=resolved_turns),
        _dialogue_row("abandoned", scenario="focal", acceptable="location", asked="location",
                      response_type="clarification", turns=abandoned_turns),
        _dialogue_row("skipped", scenario="focal", acceptable="location", asked="",
                      response_type="recommendation", turns=skipped_turns),
    ])


@settings(max_examples=100)
@given(resolved_turns=st.integers(min_value=1, max_value=_MAX_TURNS),
       abandoned_turns=st.integers(min_value=1, max_value=_MAX_TURNS),
       skipped_turns=st.integers(min_value=1, max_value=_MAX_TURNS))
def test_property_unresolved_dialogue_is_never_more_efficient(
        resolved_turns: int, abandoned_turns: int, skipped_turns: int):
    """The three-tier ordering holds for every combination of turn counts (R7.4/R7.5).

    ``asked & resolved`` > ``asked & abandoned`` > ``necessary clarification skipped``.
    Turn count cannot buy a better tier: the abandoned run is scored worse than the
    resolved one even when it used a single turn and the resolved one used the maximum.

    **Validates: Requirements 7.4, 7.5**
    """
    table = clarification_efficiency(_three_way_frame(
        resolved_turns=resolved_turns, abandoned_turns=abandoned_turns,
        skipped_turns=skipped_turns)).set_index("variant")

    resolved = table.loc["resolved", "efficiency_score"]
    abandoned = table.loc["abandoned", "efficiency_score"]
    skipped = table.loc["skipped", "efficiency_score"]

    assert abandoned < resolved
    assert skipped < abandoned
    # The abandoned dialogue is counted where the score can be audited against it, and it
    # is NOT recorded as a missed clarification (the question was asked).
    assert table.loc["abandoned", "asked_unresolved"] == 1
    assert table.loc["abandoned", "necessary_missed"] == 0
    assert table.loc["resolved", "asked_unresolved"] == 0
    assert table.loc["skipped", "necessary_missed"] == 1


def test_unresolved_penalty_needs_no_termination_reason_column():
    """Every unresolved termination is caught by the final response, not by its reason.

    The four guards (``continuation_disabled``, ``max_turns``, ``cannot_answer``,
    ``repeated_slot``) all leave a pending clarification as the final response, so the same
    penalty fires for each without the score enumerating any of them -- deliberately
    diverging from ``casestudies.DIALOGUE_NOT_CONTINUED_REASONS``, which names only
    ``continuation_disabled`` for its own attribution purpose.
    """
    rows = []
    for reason in ("continuation_disabled", "max_turns", "cannot_answer", "repeated_slot"):
        row = _dialogue_row(reason, scenario="focal", acceptable="location",
                            asked="location", response_type="clarification", turns=1.0)
        row["termination_reason"] = reason
        rows.append(row)
    resolved = _dialogue_row("resolved", scenario="focal", acceptable="location",
                             asked="location", response_type="recommendation", turns=6.0)
    resolved["termination_reason"] = "recommendation"
    table = clarification_efficiency(pd.DataFrame([*rows, resolved])).set_index("variant")

    scores = {v: table.loc[v, "efficiency_score"] for v in table.index}
    for reason in ("continuation_disabled", "max_turns", "cannot_answer", "repeated_slot"):
        assert scores[reason] < scores["resolved"], reason
        assert table.loc[reason, "asked_unresolved"] == 1


def test_frame_without_response_type_scores_as_before():
    """Pre-trace frames carry no evidence of abandonment and keep their old scores."""
    frame = _three_way_frame(resolved_turns=2, abandoned_turns=1, skipped_turns=1
                             ).drop(columns=["response_type"])
    table = clarification_efficiency(frame).set_index("variant")

    assert table.loc["resolved", "efficiency_score"] == pytest.approx(-2.0)
    assert table.loc["abandoned", "efficiency_score"] == pytest.approx(-1.0)
    assert (table["asked_unresolved"] == 0).all()


# ------------------------------------------------------- regression: one_shot vs no_memory
def _one_shot_vs_no_memory_frame() -> pd.DataFrame:
    """The seven clarification-dependent scenarios as the real official run recorded them.

    ``one_shot`` asks the acceptable slot and stops on ``continuation_disabled`` with the
    question pending (1 turn, nothing returned, task_success 0); ``no_memory`` asks the same
    slot, the simulated user answers, and it recommends (2 turns, 5 results,
    task_success 1). Before the unresolved-dialogue penalty existed, ``one_shot`` scored
    -1.0 per run against ``no_memory``'s -2.0 and therefore ranked as MORE efficient.
    """
    rows = []
    for i in range(7):
        scenario = f"SC-CLARIFY-{i}"
        one_shot = _dialogue_row(
            "one_shot", scenario=scenario, acceptable="target_roles",
            asked="target_roles", response_type="clarification", turns=1.0)
        one_shot |= {"termination_reason": "continuation_disabled", "returned_count": 0,
                     "task_success": 0, "clarification_asked": True,
                     "clarification_answered": False}
        no_memory = _dialogue_row(
            "no_memory", scenario=scenario, acceptable="target_roles",
            asked="target_roles", response_type="recommendation", turns=2.0)
        no_memory |= {"termination_reason": "recommendation", "returned_count": 5,
                      "task_success": 1, "clarification_asked": True,
                      "clarification_answered": True}
        rows += [one_shot, no_memory]
    return pd.DataFrame(rows)


def test_regression_one_shot_does_not_out_score_no_memory():
    """The reported defect: abandoning all 7 dialogues must not read as more efficient.

    ``one_shot`` asked the right question every time and never acted on the answer, so its
    shorter dialogue must no longer out-rank ``no_memory``, which asked and then answered.

    **Validates: Requirements 7.4, 7.5**
    """
    table = clarification_efficiency(_one_shot_vs_no_memory_frame()).set_index("variant")

    assert table.loc["one_shot", "efficiency_score"] < \
        table.loc["no_memory", "efficiency_score"]
    # Neither variant skipped the question, so the ranking is decided by resolution alone.
    assert table.loc["one_shot", "necessary_missed"] == 0
    assert table.loc["no_memory", "necessary_missed"] == 0
    # ... and the reason is auditable next to the score.
    assert table.loc["one_shot", "asked_unresolved"] == 7
    assert table.loc["no_memory", "asked_unresolved"] == 0
    assert table.loc["one_shot", "efficiency_score"] == pytest.approx(-1001.0)
    assert table.loc["no_memory", "efficiency_score"] == pytest.approx(-2.0)

    # One definition: the per-run column the report ranks on carries the same numbers.
    frame = _one_shot_vs_no_memory_frame()
    per_run = clarification_efficiency_per_run(frame).groupby(frame["variant"]).mean()
    assert per_run.loc["one_shot"] == pytest.approx(
        table.loc["one_shot", "efficiency_score"])
    assert per_run.loc["no_memory"] == pytest.approx(
        table.loc["no_memory", "efficiency_score"])


def test_regression_survives_a_csv_round_trip(tmp_path):
    """The ranking is read the same way from a reloaded CSV (booleans/text columns)."""
    frame = _one_shot_vs_no_memory_frame()
    path = tmp_path / "run_metrics.csv"
    frame.to_csv(path, index=False)

    table = clarification_efficiency(pd.read_csv(path, dtype=str)).set_index("variant")
    assert table.loc["one_shot", "efficiency_score"] < \
        table.loc["no_memory", "efficiency_score"]
    assert table.loc["one_shot", "asked_unresolved"] == 7
