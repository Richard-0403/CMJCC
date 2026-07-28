"""Dialogue-level clarification scoring and metrics (R7.3, R7.4, R7.8).

Checklist items 1 and 2. A clarification-dependent scenario is correct when the system
ASKS the necessary question, the simulated user ANSWERS it, and the system THEN returns a
correct recommendation -- so the final response of a correct run is a ``recommendation``,
not a ``clarification``. These tests drive the real
:class:`jobrec_eval.metrics.MetricsComputer` over run bundles carrying the per-turn
``dialogue_trace.jsonl`` shapes that ``jobrec.evaluation.exporters.trace_record`` writes,
and check that:

* the full clarify-then-recommend dialogue scores ``task_success == 1``;
* skipping the necessary question, asking only the wrong slot, or stalling on a guard
  (``max_turns`` / ``cannot_answer``) scores 0;
* clarification recall is no longer 0 for a run that asked correctly and then recommended;
* a bundle with no dialogue evidence still scores by the old final-response rule.

Reference contexts are real :class:`~jobrec.domain.constraints.JobContextState` bundles and
the recommended jobs come from the real catalog, so HCSR is recomputed by the real
:class:`~jobrec.agents.job_context_agent.JobContextAgent` rather than stubbed.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import pytest

from jobrec.catalog import load_catalog
from jobrec.config import load_config
from jobrec.domain.constraints import JobContextState
from jobrec_eval.loaders import RunBundle
from jobrec_eval.metrics import MetricsComputer, dialogue_view
from jobrec_eval.metrics_extra import (
    clarification_efficiency,
    clarification_metrics,
    failure_detection_rate,
    recovery_success_rate,
)
from jobrec_eval.scenarios import Scenario
from tests.conftest import CATALOG_PATH

#: The slot the clarification-dependent scenario declares acceptable (as the shipped
#: scenario set does for its type-B/G scenarios).
SLOT = "target_roles"
#: A slot the reference does NOT accept, used for the wrong-target run.
OTHER_SLOT = "preferred_locations"

CLARIFY_SCENARIO = "SC-CLARIFY"
RECOMMEND_SCENARIO = "SC-RECOMMEND"


# --------------------------------------------------------------------------- fixtures
def _scenario(scenario_id: str, *, clarification: bool) -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        scenario_type="clarification" if clarification else "recommendation",
        difficulty="medium",
        memory_dependency="none",
        context_dependency="low",
        no_match_expected=False,
        clarification_expected=clarification,
        acceptable_slots=[SLOT] if clarification else [],
        expected_response="clarification" if clarification else "recommendation",
        turns=["I am looking for something in Kuala Lumpur."],
    )


@pytest.fixture(scope="module")
def catalog():
    return load_catalog(CATALOG_PATH)


@pytest.fixture(scope="module")
def computer(catalog) -> MetricsComputer:
    """The real scorer over both scenarios, with an authoritative reference context.

    The reference bundle carries no constraints, so every recommended job is eligible and
    HCSR recomputes to 1.0 -- which keeps these tests about the dialogue rule rather than
    about constraint evaluation (covered elsewhere).
    """
    config = load_config("configs/experiment_full.yaml", base_dir="configs")
    context = JobContextState(
        context_id="ctx-clarify",
        active_search_id="as-clarify",
        catalog_snapshot_id="cat-clarify",
        constraints=[],
        normalized_at=datetime.now(UTC),
    ).model_dump(mode="json")
    references = {
        CLARIFY_SCENARIO: {"job_context": context},
        RECOMMEND_SCENARIO: {"job_context": context},
    }
    scenarios = {
        CLARIFY_SCENARIO: _scenario(CLARIFY_SCENARIO, clarification=True),
        RECOMMEND_SCENARIO: _scenario(RECOMMEND_SCENARIO, clarification=False),
    }
    labels = pd.DataFrame(columns=["scenario_id", "job_id", "relevance_grade"])
    return MetricsComputer(config, catalog, references, labels, scenarios)


# ----------------------------------------------------------------- bundle construction
def _record(action: str, slot: str | None, reason: str | None = None) -> dict:
    """One ``dialogue_trace.jsonl`` record, in the shape ``trace_record`` writes.

    ``clarification_slot`` is the slot the turn is about: the slot the system asked on a
    scripted turn, or the slot the simulated user answered on a loop turn. Only the final
    record carries a termination reason (the runner stamps it there).
    """
    return {
        "user_utterance": "…",
        "system_action": action,
        "clarification_slot": slot,
        "extracted_value": {},
        "state_version": {"dialogue_state": 1, "candidate_state": 1},
        "termination_reason": reason,
    }


def _clarification(slot: str) -> dict:
    """A ``clarification.json`` payload for a still-pending ask."""
    return {
        "clarification_id": "clar-1",
        "target_fields": [slot],
        "reason_code": "missing_hard_constraint",
        "question_text": "Which roles are you targeting?",
    }


def _bundle(
    *,
    scenario_id: str = CLARIFY_SCENARIO,
    response_type: str = "recommendation",
    trace: list[dict] | None,
    clarification: dict | None = None,
    returned: bool = True,
    variant: str = "full",
    catalog=None,
) -> RunBundle:
    """A run bundle in the JSON shapes ``load_bundles`` reads from disk."""
    job_id = catalog[0].job_id if returned and catalog else None
    decision = {
        "selected_job_ids": [job_id] if job_id else [],
        "ranked_jobs": ([{"job_id": job_id, "rank": 1, "total_score": 0.9,
                          "eligibility_result_id": "elig-1",
                          "features": [{"name": "role_match", "evidence_ids": ["ev-1"]}],
                          "skill_gaps": []}] if job_id else []),
        "eligibility_results": [],
        "no_match": False,
        "no_match_reason_codes": [],
    }
    return RunBundle(
        variant=variant,
        scenario_id=scenario_id,
        run_index=0,
        path=Path("."),
        run_record={"run_id": f"run-{scenario_id}-{variant}-{response_type}",
                    "success": True, "total_latency_ms": 12.0},
        decision=decision,
        response={"response_type": response_type},
        claims=[{"claim_id": "claim-1", "claim_type": "job_attribute",
                 "support_status": "supported", "evidence_ids": ["ev-1"]}],
        handoffs=[{"handoff_id": "ho-1", "validation_passed": True, "status": "completed"}],
        evidence_log=[{"stage": "understanding", "status": "success"}],
        latency={},
        active_search=None,
        job_context=None,
        clarification=clarification,
        dialogue_trace=trace,
    )


def _row(computer: MetricsComputer, bundle: RunBundle) -> pd.Series:
    return computer.run_metrics([bundle]).iloc[0]


# ---------------------------------------------------------- dialogue-level task success
def test_asking_then_recommending_is_a_task_success(computer, catalog):
    """The correct clarify-then-recommend dialogue scores 1, not 0 (checklist item 1).

    The system asked the acceptable slot, the simulated user answered it (the dialogue
    continued past the ask) and the final turn returned an eligible, grounded
    recommendation -- so the run succeeded even though the FINAL response is a
    recommendation rather than a clarification.

    **Validates: Requirements 7.4**
    """
    bundle = _bundle(
        trace=[_record("clarification", SLOT),
               _record("recommendation", SLOT, reason="recommendation")],
        catalog=catalog,
    )
    row = _row(computer, bundle)

    assert row["task_success"] == 1
    assert row["partial_task_score"] == pytest.approx(1.0)
    # The recommendation really did clear the quality bars it is credited for.
    assert row["hcsr"] == pytest.approx(1.0)
    assert row["grounded_claim_count"] > 0
    # And the dialogue-level columns describe what happened.
    assert row["clarification_asked"]
    assert row["clarification_answered"]
    assert row["clarification_asked_slots"] == SLOT
    assert row["response_turns"] == 2
    assert row["termination_reason"] == "recommendation"


def test_skipping_the_necessary_clarification_is_never_a_success(computer, catalog):
    """A run that guessed straight to a good recommendation still scores 0.

    Same terminal recommendation as the successful run above -- eligible, grounded -- but
    the necessary question was never asked, so the outcome was not earned.

    **Validates: Requirements 7.4**
    """
    bundle = _bundle(
        trace=[_record("recommendation", None, reason="recommendation")],
        catalog=catalog,
    )
    row = _row(computer, bundle)

    assert row["task_success"] == 0
    assert not row["clarification_asked"]
    assert row["clarification_asked_slots"] == ""
    # The recommendation itself was fine: the zero is about the skipped clarification.
    assert row["hcsr"] == pytest.approx(1.0)
    assert row["grounded_claim_count"] > 0


def test_asking_only_the_wrong_slot_is_not_a_success(computer, catalog):
    """Asking a clarification on a non-acceptable slot only does not count (R7.4)."""
    bundle = _bundle(
        trace=[_record("clarification", OTHER_SLOT),
               _record("recommendation", OTHER_SLOT, reason="recommendation")],
        catalog=catalog,
    )
    row = _row(computer, bundle)

    assert row["task_success"] == 0
    assert row["clarification_asked"]
    assert row["clarification_asked_slots"] == OTHER_SLOT


@pytest.mark.parametrize(
    ("reason", "trace_factory"),
    [
        # Still clarifying when the turn cap fired: the ask was answered but the dialogue
        # never reached a terminal recommendation.
        ("max_turns", lambda: [_record("clarification", SLOT),
                               _record("clarification", SLOT, reason="max_turns")]),
        # The simulated user could not answer, so the loop stopped on the first ask.
        ("cannot_answer", lambda: [_record("clarification", SLOT, reason="cannot_answer")]),
        # The system re-asked an answered slot and the guard stopped the dialogue.
        ("repeated_slot", lambda: [_record("clarification", SLOT),
                                   _record("clarification", SLOT, reason="repeated_slot")]),
    ],
)
def test_dialogues_that_never_reach_a_terminal_outcome_score_zero(
    computer, catalog, reason, trace_factory
):
    """A run still sitting on a clarification when a guard fired is not a success (R7.6/7.7)."""
    bundle = _bundle(
        response_type="clarification",
        trace=trace_factory(),
        clarification=_clarification(OTHER_SLOT),
        returned=False,
        catalog=catalog,
    )
    row = _row(computer, bundle)

    assert row["task_success"] == 0
    assert row["termination_reason"] == reason
    assert row["clarification_asked"]
    # The repeated-slot guard is recorded as a repeat; the other two are not.
    assert bool(row["clarification_repeated_slot"]) == (reason == "repeated_slot")


def test_bundles_without_dialogue_evidence_keep_the_previous_rule(computer, catalog):
    """No trace: score the final response as before, so older bundles still work.

    ``write_run_bundle`` derives a single-record trace with no termination reason for
    non-loop callers, and older bundles have no trace at all. Both are scored by the
    previous final-response rule rather than being penalized for missing evidence.

    **Validates: Requirements 7.4**
    """
    for trace in (None, [_record("clarification", SLOT)]):
        clarifying = _bundle(
            response_type="clarification",
            trace=trace,
            clarification=_clarification(SLOT),
            returned=False,
            catalog=catalog,
        )
        assert _row(computer, clarifying)["task_success"] == 1, trace
        assert not dialogue_view(clarifying).scored

        # ... and the old rule's converse: a final response that is not a clarification
        # scores 0 for a clarification-dependent scenario when there is no trace to score.
        recommending = _bundle(trace=trace, catalog=catalog)
        assert _row(computer, recommending)["task_success"] == 0, trace


def test_non_clarification_scenarios_are_unaffected(computer, catalog):
    """A recommendation-expected scenario keeps its existing verdict (R9.2)."""
    good = _bundle(scenario_id=RECOMMEND_SCENARIO,
                   trace=[_record("recommendation", None, reason="recommendation")],
                   catalog=catalog)
    empty = _bundle(scenario_id=RECOMMEND_SCENARIO,
                    trace=[_record("recommendation", None, reason="recommendation_empty")],
                    returned=False, catalog=catalog)

    assert _row(computer, good)["task_success"] == 1
    assert _row(computer, empty)["task_success"] == 0


# ------------------------------------------------------------- trace-derived metrics
def test_recall_credits_a_run_that_asked_correctly_then_recommended(computer, catalog):
    """Clarification recall is no longer 0 for an asked-then-recommended run.

    The ``full`` variant asks the acceptable slot and then recommends; the ``no_memory``
    variant guesses without asking. Recall, precision and the useful/unnecessary rates all
    reflect the whole dialogue instead of the final response.

    **Validates: Requirements 7.3, 7.4**
    """
    asked = _bundle(
        trace=[_record("clarification", SLOT),
               _record("recommendation", SLOT, reason="recommendation")],
        catalog=catalog,
    )
    skipped = _bundle(
        variant="no_memory",
        trace=[_record("recommendation", None, reason="recommendation")],
        catalog=catalog,
    )
    run_metrics = computer.run_metrics([asked, skipped])
    table = clarification_metrics(run_metrics).set_index("variant")

    full = table.loc["full"]
    assert full["clarifications"] == 1
    assert full["expected_clarification"] == 1
    assert full["useful"] == 1
    assert full["recall"] == pytest.approx(1.0)
    assert full["necessary_recall"] == pytest.approx(1.0)
    assert full["precision"] == pytest.approx(1.0)
    assert full["useful_rate"] == pytest.approx(1.0)
    assert full["unnecessary_rate"] == pytest.approx(0.0)
    assert full["answered_rate"] == pytest.approx(1.0)
    assert full["repeated_question_rate"] == pytest.approx(0.0)

    # The variant that never asked has recall 0 and N/A (never 0.0) ask-level rates.
    skipped_row = table.loc["no_memory"]
    assert skipped_row["clarifications"] == 0
    assert skipped_row["necessary_recall"] == pytest.approx(0.0)
    assert skipped_row["precision"] is None or pd.isna(skipped_row["precision"])
    assert skipped_row["useful_rate"] is None or pd.isna(skipped_row["useful_rate"])
    assert (skipped_row["repeated_question_rate"] is None
            or pd.isna(skipped_row["repeated_question_rate"]))


def test_clarification_metrics_survive_a_csv_round_trip(computer, catalog, tmp_path):
    """Reloading run_metrics from CSV (booleans as text) yields the same table.

    The metrics CSV is what the report stage reads back, so the trace-derived boolean
    columns must be interpreted the same way after a round trip.
    """
    asked = _bundle(
        trace=[_record("clarification", SLOT),
               _record("recommendation", SLOT, reason="recommendation")],
        catalog=catalog,
    )
    skipped = _bundle(
        variant="no_memory",
        trace=[_record("recommendation", None, reason="recommendation")],
        catalog=catalog,
    )
    run_metrics = computer.run_metrics([asked, skipped])
    csv_path = tmp_path / "run_metrics.csv"
    run_metrics.to_csv(csv_path, index=False)

    before = clarification_metrics(run_metrics)
    after = clarification_metrics(pd.read_csv(csv_path, dtype=str))
    pd.testing.assert_frame_equal(before, after, check_dtype=False)


def test_repeated_question_rate_reads_the_guard(computer, catalog):
    """A run whose repeated-slot guard fired is counted in the repeat rate (R7.7)."""
    repeated = _bundle(
        response_type="clarification",
        trace=[_record("clarification", SLOT),
               _record("clarification", SLOT, reason="repeated_slot")],
        clarification=_clarification(SLOT),
        returned=False,
        catalog=catalog,
    )
    table = clarification_metrics(computer.run_metrics([repeated])).set_index("variant")

    assert table.loc["full", "repeated"] == 1
    assert table.loc["full", "repeated_question_rate"] == pytest.approx(1.0)


def test_per_run_efficiency_column_matches_the_variant_score(computer, catalog):
    """The per-run ``clarification_efficiency`` column is the same score the table means.

    One definition of the score feeds both, and it keeps the R7.5 invariant: the run that
    skipped the necessary clarification scores strictly worse than the one that asked it,
    even though it used fewer turns.

    **Validates: Requirements 7.4, 7.5**
    """
    asked = _bundle(
        trace=[_record("clarification", SLOT),
               _record("recommendation", SLOT, reason="recommendation")],
        catalog=catalog,
    )
    skipped = _bundle(
        variant="no_memory",
        trace=[_record("recommendation", None, reason="recommendation")],
        catalog=catalog,
    )
    run_metrics = computer.run_metrics([asked, skipped])
    table = clarification_efficiency(run_metrics).set_index("variant")

    per_run = run_metrics.set_index("variant")["clarification_efficiency"]
    assert per_run.loc["full"] == pytest.approx(table.loc["full", "efficiency_score"])
    assert per_run.loc["no_memory"] == pytest.approx(
        table.loc["no_memory", "efficiency_score"])
    # Fewer turns, yet strictly less efficient: the skip dominates (R7.5).
    assert run_metrics.set_index("variant")["response_turns"].loc["no_memory"] < \
        run_metrics.set_index("variant")["response_turns"].loc["full"]
    assert per_run.loc["no_memory"] < per_run.loc["full"]


def test_fault_columns_are_present_and_report_na_without_injected_faults(computer, catalog):
    """The four R10.8 columns exist on every row and stay False for a clean run.

    The main deterministic experiment injects nothing, so the detection / recovery rates
    must read N/A rather than a misleading 1.000.

    **Validates: Requirements 10.8**
    """
    run_metrics = computer.run_metrics([
        _bundle(trace=[_record("clarification", SLOT),
                       _record("recommendation", SLOT, reason="recommendation")],
                catalog=catalog),
    ])

    for column in ("failure_injected", "failure_detected", "recoverable", "recovered"):
        assert column in run_metrics.columns
        assert not run_metrics[column].any(), column
    assert failure_detection_rate(run_metrics) is None
    assert recovery_success_rate(run_metrics) is None
