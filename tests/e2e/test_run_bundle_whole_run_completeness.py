"""Whole-run completeness of the exported run bundle (R7.3/R11.1).

A run is a DIALOGUE, but the bundle used to export only its FINAL turn: the final
turn's ``run_record.json`` and the final turn's ``model_calls.jsonl``. For a
clarification scenario -- the very scenarios this thesis studies -- that silently
discarded every earlier turn's model calls, so the archived evidence for a 3-turn run
was neither replayable nor countable towards tokens, cost or latency.

These tests drive the real :class:`ExperimentRunner` over the real pipeline with the
mock provider under ``mode=hybrid``, which produces genuine ``LLMCallRecord``s on every
turn without any network access. The assertions are about the ARCHIVE, not about
metrics: metric definitions deliberately still score the final turn, because the final
response is the run's answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobrec.config import load_config
from jobrec.domain.enums import RunMode
from jobrec.evaluation.experiment_runner import ExperimentRunner

CATALOG_PATH = "data/processed/jobs.jsonl"

#: An opening turn that states location/mode/salary but NO role, so the system asks for
#: one and the clarification loop has to spend at least one further turn answering it.
#: A multi-turn run is the only condition under which the dropped-turn bug is observable.
_TURNS = ["I am looking for something in Kuala Lumpur, hybrid, around RM5000."]

_SLOTS = ["target_roles", "preferred_locations", "work_modes"]


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


@pytest.fixture(scope="module")
def tiny_catalog(tmp_path_factory) -> str:
    """A 6-job slice of the real catalog: enough to recommend, small enough to be fast."""
    jobs = [
        json.loads(line)
        for line in Path(CATALOG_PATH).read_text().splitlines()
        if line.strip()
    ]
    tiny = [job for job in jobs if job["city"] == "Kuala Lumpur"][:6]
    assert len(tiny) == 6
    path = tmp_path_factory.mktemp("catalog") / "jobs.jsonl"
    path.write_text("\n".join(json.dumps(job) for job in tiny) + "\n")
    return str(path)


def _scenario() -> dict:
    return {
        "scenario_id": "SC-BUNDLE-01",
        "scenario_type": "clarification",
        "profile": {
            "candidate_id": "SC-BUNDLE-01-cand",
            "skills": ["Python"],
            "years_experience": 2,
        },
        "turns": list(_TURNS),
        "clarification_expected": True,
        "acceptable_slots": list(_SLOTS),
        "expects": {"response_type": "clarification"},
    }


def _run(tmp_path, tiny_catalog, *, mode: RunMode) -> dict:
    """Drive one multi-turn run through the real runner and return its run row."""
    config = load_config("configs/experiment_full.yaml", base_dir="configs")
    config = config.model_copy(deep=True)
    config.experiment.max_dialogue_turns = 4
    # ``provider="mock"`` keeps ``make_provider`` on MockLLMProvider, so hybrid mode
    # exercises the real call-recording path with no network and no API key.
    config.llm.mode = mode
    config.llm.provider = "mock"
    scenarios_path = tmp_path / "scenarios.jsonl"
    scenarios_path.write_text(json.dumps(_scenario()) + "\n")
    runner = ExperimentRunner(
        config, tiny_catalog, str(scenarios_path), out_dir=str(tmp_path / "out"))
    row, failure = runner._run_one("full", _scenario(), 0, tmp_path / "out" / "exp")
    assert failure is None, failure
    return row


def test_every_turn_of_a_multi_turn_run_is_archived(tmp_path, tiny_catalog) -> None:
    """``turn_records.jsonl`` has one row per turn, and the last row IS ``run_record.json``.

    ``run_record.json`` keeps its exact previous meaning (the final turn), so no existing
    reader changes behaviour; the earlier turns become recoverable instead of being
    dropped on the floor.

    **Validates: Requirements 7.3, 11.1**
    """
    row = _run(tmp_path, tiny_catalog, mode=RunMode.HYBRID)
    run_dir = Path(row["run_dir"])

    assert row["response_turns"] > 1, "the scenario did not actually go multi-turn"

    turn_rows = _read_jsonl(run_dir / "turn_records.jsonl")
    assert len(turn_rows) == row["response_turns"]
    assert [r["turn_index"] for r in turn_rows] == list(range(len(turn_rows)))

    run_record = json.loads((run_dir / "run_record.json").read_text())
    assert turn_rows[-1]["run_id"] == run_record["run_id"], (
        "the last turn record must be the final turn described by run_record.json")
    assert turn_rows[-1]["total_latency_ms"] == run_record["total_latency_ms"]
    # The earlier turns are distinct runs of the pipeline, not copies of the final one.
    assert len({r["run_id"] for r in turn_rows}) == len(turn_rows)
    # The turn that asked the clarification is identifiable as such.
    assert any(r["asked_clarification"] for r in turn_rows[:-1]), turn_rows


def test_model_calls_of_every_turn_are_exported_and_attributed(tmp_path, tiny_catalog) -> None:
    """``model_calls.jsonl`` covers all turns and says which turn each call belongs to.

    This is the regression that mattered most: with only the final turn exported, a
    multi-turn hybrid run's earlier calls appeared in NO artifact, so replay was
    incomplete and token/cost accounting undercounted.

    **Validates: Requirements 7.3, 11.1**
    """
    row = _run(tmp_path, tiny_catalog, mode=RunMode.HYBRID)
    run_dir = Path(row["run_dir"])

    calls = _read_jsonl(run_dir / "model_calls.jsonl")
    turn_rows = _read_jsonl(run_dir / "turn_records.jsonl")
    assert len(turn_rows) > 1

    # Every call is attributed to a real turn, and the per-turn counts agree exactly
    # with what each turn reported -- so no call can be silently lost or duplicated.
    assert calls, "hybrid mode recorded no model calls at all"
    assert all("turn_index" in c and "turn_run_id" in c for c in calls)
    per_turn: dict[int, int] = {}
    by_turn_run_id: dict[int, set[str]] = {}
    for call in calls:
        per_turn[call["turn_index"]] = per_turn.get(call["turn_index"], 0) + 1
        by_turn_run_id.setdefault(call["turn_index"], set()).add(call["turn_run_id"])
    for turn in turn_rows:
        index = turn["turn_index"]
        assert per_turn.get(index, 0) == turn["model_call_count"], (index, per_turn)
        if index in by_turn_run_id:
            assert by_turn_run_id[index] == {turn["run_id"]}, index

    # Calls exist for a turn OTHER than the last: the exact evidence that used to vanish.
    assert len([i for i, n in per_turn.items() if n > 0]) > 1, per_turn
    assert any(i != turn_rows[-1]["turn_index"] for i in per_turn), per_turn

    # Prompts are still never written, and each row is still replay-keyed.
    assert not any("prompt" in c for c in calls)
    assert all(c.get("call_id") for c in calls)

    totals = json.loads((run_dir / "run_totals.json").read_text())
    assert totals["model_call_total"] == len(calls)
    assert totals["turn_count"] == len(turn_rows)
    assert totals["model_call_coverage"]["expects_calls"] is True


def test_clarification_phrasing_call_is_recorded(tmp_path, tiny_catalog) -> None:
    """The clarification rephrasing call appears in the bundle.

    In hybrid mode the clarification question is rephrased by the provider. That is a
    real, billed call, but its record was discarded at the call site, so it was missing
    from every artifact and from all cost/replay accounting.

    **Validates: Requirements 7.3, 11.1**
    """
    row = _run(tmp_path, tiny_catalog, mode=RunMode.HYBRID)
    calls = _read_jsonl(Path(row["run_dir"]) / "model_calls.jsonl")
    purposes = [c["purpose"] for c in calls]
    assert "clarification" in purposes, purposes


def test_run_totals_sum_latency_across_turns(tmp_path, tiny_catalog) -> None:
    """``run_totals.json`` aggregates the whole dialogue; ``component_latency.json`` does not.

    Latency and call counts are the only genuinely additive quantities, so they are the
    only things aggregated. ``component_latency.json`` stays final-turn-scoped on
    purpose: ``retrieval_results.json`` reads the same record, and mixing a whole-run
    sum into it would make that snapshot incoherent.

    **Validates: Requirements 7.3, 11.1**
    """
    row = _run(tmp_path, tiny_catalog, mode=RunMode.HYBRID)
    run_dir = Path(row["run_dir"])
    turn_rows = _read_jsonl(run_dir / "turn_records.jsonl")
    totals = json.loads((run_dir / "run_totals.json").read_text())
    final_component = json.loads((run_dir / "component_latency.json").read_text())

    assert len(turn_rows) > 1
    expected = round(sum(r["total_latency_ms"] for r in turn_rows), 3)
    assert totals["total_latency_ms"] == pytest.approx(expected)
    assert totals["final_turn_total_latency_ms"] == turn_rows[-1]["total_latency_ms"]
    # A real dialogue costs more than its last turn alone.
    assert totals["total_latency_ms"] > totals["final_turn_total_latency_ms"]

    assert final_component == turn_rows[-1]["component_latency_ms"]
    for name, summed in totals["component_latency_ms"].items():
        per_turn_sum = sum(r["component_latency_ms"].get(name, 0.0) for r in turn_rows)
        assert summed == pytest.approx(per_turn_sum, abs=1e-3), name


def test_deterministic_run_is_not_flagged_for_missing_calls(tmp_path, tiny_catalog) -> None:
    """Zero model calls is CORRECT for the deterministic backend, not a coverage gap.

    The call-coverage signal is reported rather than enforced, and it must not accuse
    the deterministic condition -- which is the thesis's baseline -- of missing data it
    was never supposed to have.

    **Validates: Requirements 7.3, 11.1**
    """
    row = _run(tmp_path, tiny_catalog, mode=RunMode.DETERMINISTIC)
    run_dir = Path(row["run_dir"])
    totals = json.loads((run_dir / "run_totals.json").read_text())

    assert totals["model_call_total"] == 0
    assert totals["model_call_coverage"]["expects_calls"] is False
    assert totals["model_call_coverage"]["complete"] is True
    assert _read_jsonl(run_dir / "model_calls.jsonl") == []
    # The per-turn archive still exists for a deterministic run.
    assert len(_read_jsonl(run_dir / "turn_records.jsonl")) == row["response_turns"]
