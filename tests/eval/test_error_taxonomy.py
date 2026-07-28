"""Root-cause error taxonomy: a truncated dialogue is not a memory failure.

:func:`jobrec_eval.casestudies.error_taxonomy` assigns one primary root cause per
task-unsuccessful run. It used to send every ``one_shot`` clarification failure to
``stale_or_missing_memory (ablation)``, which mislabelled the ``one_shot`` truncation
runs: those end on the turn that ASKED, with ``termination_reason ==
"continuation_disabled"``, because the variant may not continue the dialogue
(``FeatureFlags.use_multi_turn_continuation`` off) -- nothing about memory.

These tests pin the recorded-column signal, the precedence between the rules, and the
meanings of the neighbouring categories, over frames shaped exactly like
``metrics/run_metrics.csv``.
"""

from __future__ import annotations

import pandas as pd

from jobrec.evaluation.experiment_runner import TERMINATION_CONTINUATION_DISABLED
from jobrec_eval.casestudies import error_taxonomy

TRUNCATED = "missing_dialogue_continuation (ablation)"
MEMORY = "stale_or_missing_memory (ablation)"
CONSTRAINTS = "missing_constraint_enforcement (ablation)"
DIALOGUE_EVIDENCE = "missing_dialogue_evidence (baseline)"
CLARIFICATION_QUALITY = "under/over_clarification"


def _row(**overrides) -> dict:
    """A task-unsuccessful run row with the columns the classifier reads."""
    row = {
        "scenario_id": "SC-X-01",
        "variant": "full",
        "response_type": "recommendation",
        "task_success": 0,
        "returned_count": 5,
        "hcsr": 1.0,
        "clarification_expected": False,
        "clarification_asked": False,
        "clarification_answered": False,
        "clarification_repeated_slot": False,
        "termination_reason": "recommendation",
    }
    row.update(overrides)
    return row


def _truncated_row(**overrides) -> dict:
    """The observed ``one_shot`` shape: asked, never answered, nothing returned."""
    return _row(**{
        "variant": "one_shot", "response_type": "clarification", "returned_count": 0,
        "hcsr": None, "clarification_expected": True, "clarification_asked": True,
        "clarification_answered": False,
        "termination_reason": TERMINATION_CONTINUATION_DISABLED, **overrides})


def _memory_row(**overrides) -> dict:
    """The observed ``SC-D-*`` shape: an UNexpected ask for forgotten evidence.

    A memory-ablated variant asks for the role/currency stated in an earlier turn. The
    scenario never expected a clarification, so it never entered the dialogue loop and no
    terminal state was recorded -- the ask itself is the symptom of the missing memory.
    """
    return _row(**{
        "variant": "no_memory", "response_type": "clarification", "returned_count": 0,
        "hcsr": None, "clarification_expected": False, "clarification_asked": True,
        "clarification_answered": False, "termination_reason": None, **overrides})


def _categorise(rows: list[dict]) -> dict[str, int]:
    table = error_taxonomy(pd.DataFrame(rows))
    return dict(zip(table.error_category, table["count"], strict=True))


def _category_of(row: dict) -> str:
    table = error_taxonomy(pd.DataFrame([row]))
    assert len(table) == 1
    return table.iloc[0].error_category


# ------------------------------------------------- the relabelled truncation failure
def test_continuation_disabled_run_is_a_dialogue_continuation_failure():
    """A single-turn truncation is attributed to continuation, never to memory."""
    assert _category_of(_truncated_row()) == TRUNCATED


def test_truncation_does_not_land_in_the_memory_category():
    counts = _categorise([_truncated_row(scenario_id=f"SC-B-0{i}") for i in range(1, 6)])
    assert counts == {TRUNCATED: 5}
    assert MEMORY not in counts


def test_truncation_is_recognised_for_any_variant_not_only_one_shot():
    """The rule reads the recorded terminal state, not the variant name.

    ``memory.use_multi_turn_continuation: false`` can produce this state under any
    variant; the cause is the same mechanism, so the category must be the same.
    """
    for variant in ("full", "no_memory", "no_context", "profile_only"):
        assert _category_of(_truncated_row(variant=variant)) == TRUNCATED, variant


def test_truncation_of_an_unexpected_ask_is_still_a_continuation_failure():
    """A single-turn run that asked an UNEXPECTED question is truncated all the same.

    This is the class that used to be misfiled: the runner only stamped
    ``continuation_disabled`` when the scenario expected a clarification, so these runs
    arrived with no terminal state and the fallback (classify by expectation) sent them
    to the memory category. The runner now records the reason for them too, and the rule
    must honour it rather than re-deriving anything from the expectation.
    """
    row = _truncated_row(clarification_expected=False)
    assert _category_of(row) == TRUNCATED
    assert _category_of(_truncated_row(variant="no_memory",
                                       clarification_expected=False)) == TRUNCATED


def test_a_dialogue_that_answered_an_earlier_ask_and_ended_on_another_is_truncated():
    """Answering ask #1 and ending on ask #2 is still a truncation.

    The observed SC-D-11 / SC-D-12 shape. ``clarification_answered`` is a whole-dialogue
    flag, so requiring it to be false asked "was ANY question ever answered?" when the
    question at hand is "is the FINAL one still open?" -- and sent these two runs to the
    memory category instead.
    """
    row = _truncated_row(clarification_answered=True, clarification_expected=False)
    assert _category_of(row) == TRUNCATED


def test_truncation_without_a_recorded_reason_still_classified_by_expectation():
    """Frames predating ``termination_reason`` fall back to the scenario expectation.

    An EXPECTED clarification that was asked and never answered is a dialogue that
    stopped one turn early, whichever run-metrics vintage recorded it.
    """
    row = _truncated_row()
    row.pop("termination_reason")
    assert _category_of(row) == TRUNCATED


# ------------------------------------------------- neighbouring categories unchanged
def test_genuine_memory_failure_still_lands_in_stale_or_missing_memory():
    assert _category_of(_memory_row()) == MEMORY
    assert _category_of(_memory_row(variant="one_shot")) == MEMORY


def test_no_context_hard_constraint_violation_still_lands_in_constraints():
    row = _row(variant="no_context", hcsr=0.6, returned_count=5)
    assert _category_of(row) == CONSTRAINTS


def test_no_memory_recommendation_failure_keeps_its_old_category():
    """The observed ``no_memory`` SC-B-04 row: a recommendation, not a clarification.

    It fell into ``other`` before the truncation category existed and must still do so;
    the new rule only fires on an unresolved clarification.
    """
    row = _row(variant="no_memory", hcsr=0.6, clarification_asked=True,
               clarification_answered=True, clarification_expected=True)
    assert _category_of(row) == "other"


def test_profile_only_repeated_slot_stays_with_the_baseline_category():
    """``repeated_slot`` is NOT folded into the truncation category.

    Continuation was available and used here (the slot was answered, then re-asked), so
    the cause is the dialogue evidence ``profile_only`` never consumes.
    """
    row = _row(variant="profile_only", response_type="clarification", returned_count=0,
               hcsr=None, clarification_expected=True, clarification_asked=True,
               clarification_answered=True, clarification_repeated_slot=True,
               termination_reason="repeated_slot")
    assert _category_of(row) == DIALOGUE_EVIDENCE


def test_max_turns_and_cannot_answer_are_not_truncation_failures():
    """Guard exits of a variant that COULD continue keep their own categories."""
    for reason in ("max_turns", "cannot_answer"):
        memory_variant = _memory_row(termination_reason=reason,
                                     clarification_expected=True)
        assert _category_of(memory_variant) == MEMORY, reason
        full_variant = _memory_row(variant="full", termination_reason=reason,
                                   clarification_expected=True)
        assert _category_of(full_variant) == CLARIFICATION_QUALITY, reason


# ------------------------------------------------------------------------ precedence
def test_truncation_outranks_the_variant_rules_it_overlaps():
    """Documented precedence: the recorded terminal state wins over variant inference.

    A truncated run returned nothing, so the memory / constraint / baseline evidence the
    other rules reason about does not exist for it.
    """
    overlapping = {
        "no_memory": MEMORY, "one_shot": MEMORY,
        "no_context": "no_context_other", "profile_only": DIALOGUE_EVIDENCE,
    }
    for variant in overlapping:
        assert _category_of(_truncated_row(variant=variant)) == TRUNCATED, variant


def test_constraint_rule_is_untouched_because_the_two_never_overlap():
    """A hard-constraint violation is a recommendation; a truncation is not."""
    counts = _categorise([
        _row(variant="no_context", hcsr=0.5, scenario_id="SC-E-01"),
        _truncated_row(scenario_id="SC-B-01"),
    ])
    assert counts == {CONSTRAINTS: 1, TRUNCATED: 1}


def test_successful_runs_are_never_categorised():
    rows = [_truncated_row(), _row(task_success=1, variant="one_shot")]
    assert _categorise(rows) == {TRUNCATED: 1}


def test_most_affected_variant_is_reported_for_the_new_category():
    rows = [_truncated_row(scenario_id="SC-B-01"), _truncated_row(scenario_id="SC-B-02"),
            _truncated_row(scenario_id="SC-B-03", variant="no_memory")]
    table = error_taxonomy(pd.DataFrame(rows))
    row = table[table.error_category == TRUNCATED].iloc[0]
    assert row["count"] == 3
    assert row["most_affected_variant"] == "one_shot"
    assert row["percentage"] == 100.0
