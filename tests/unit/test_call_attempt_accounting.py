"""HTTP attempts are counted, not indexed.

``_failed_call_record`` stored the retry's 1-based INDEX under the key ``attempts``, so four
consecutive failures recorded 1, 2, 3, 4 and any sum over them reported TEN attempts for four
requests. That number is the denominator the endpoint fitness decision rests on -- fallback
rate per attempt -- so inflating it made the endpoint look better per-attempt than it is.

The index is still useful (it says which retry a record belongs to) and is kept under its own
name, so nothing is lost by making ``attempts`` mean what it says.
"""

from __future__ import annotations

import json
from pathlib import Path

from jobrec.llm.provider import LLMError
from jobrec.orchestration.orchestrator import _failed_call_record


class _Provider:
    name = "remote"
    model = "test-model"


def _record(attempt: int):
    return _failed_call_record("intent_extraction", "prompt", LLMError("dropped"),
                               _Provider(), attempt)


def test_a_failed_record_counts_one_attempt_not_its_index():
    for attempt in (1, 2, 3, 4):
        meta = _record(attempt).metadata
        assert meta["attempts"] == 1, f"attempt {attempt} reported {meta['attempts']}"
        assert meta["attempt_index"] == attempt


def test_four_failures_sum_to_four_http_attempts_not_ten():
    """The exact arithmetic that was wrong: 1+2+3+4 = 10 for four requests."""
    records = [_record(i) for i in (1, 2, 3, 4)]
    assert sum(r.metadata["attempts"] for r in records) == 4
    assert sum(r.metadata["attempt_index"] for r in records) == 10, (
        "the index sum is the number the old code reported as attempts")


def _four_failed_attempts(root, *, fallback_field: str | None = None):
    """One logical call that failed four times, laid out as an experiment directory."""
    import json

    run_dir = root / "full" / "SC-X-01" / "0"
    run_dir.mkdir(parents=True)
    with (run_dir / "model_calls.jsonl").open("w", encoding="utf-8") as fh:
        for index in (1, 2, 3, 4):
            record = _record(index)
            fh.write(json.dumps({
                "call_id": record.call_id, "purpose": "intent_extraction",
                "parsed_ok": False, "raw_response": "",
                "turn_run_id": "run-1", "turn_index": 0,
                "response_metadata": dict(record.metadata),
            }) + "\n")
    snapshot = None
    if fallback_field is not None:
        snapshot = {"preferences": [{
            "field_name": fallback_field,
            "metadata": {"extraction_source": "rule_fallback"}}]}
    (run_dir / "dialogue_state.json").write_text(
        json.dumps({"turns": [{"speaker": "candidate", "turn_id": "t0",
                               "extraction_snapshot": snapshot}]}), encoding="utf-8")
    return run_dir


def test_the_diagnosis_reports_four_attempts_for_four_failures(tmp_path):
    """End to end through the report that feeds the endpoint decision."""
    from jobrec.evaluation.llm_call_audit import audit_llm_calls

    _four_failed_attempts(tmp_path)

    counts = audit_llm_calls(tmp_path)["counts"]
    assert counts["http_attempts"] == 4, counts
    # Four records, one logical call (same run, turn and purpose), which never succeeded.
    assert counts["call_records"] == 4
    assert counts["logical_calls"] == 1
    assert counts["failed_logical_calls"] == 1
    assert counts["retry_attempts"] == 3


def test_the_manifest_summary_is_the_same_numbers_the_audit_script_prints(tmp_path):
    """One implementation, so a manifest's fallback rate cannot drift from the audit's.

    The rate is what a pre-registered threshold is checked against. Two definitions of "logical
    call" -- one in the runner, one in a script -- would let a batch pass in its own manifest
    and fail in the audit, with no way to tell which number the thesis quoted.
    """
    from jobrec.evaluation.llm_call_audit import (
        MANIFEST_SUMMARY_KEYS,
        audit_llm_calls,
        manifest_summary,
    )

    _four_failed_attempts(tmp_path, fallback_field="salary_min")
    report = audit_llm_calls(tmp_path)
    summary = manifest_summary(tmp_path)

    for key in MANIFEST_SUMMARY_KEYS:
        assert summary[key] == report[key], key
    assert summary["rates"]["final_fallback_call_rate"] == 1.0
    assert summary["counts"]["http_attempts"] == 4
    # The distribution travels with it, because "1% of calls" is acceptable only when the
    # failures are spread; the same 1% concentrated on one field is a broken field.
    assert summary["fallback_distribution"]["by_field"] == {"salary_min": 1}
    # The per-occurrence list does NOT, so a batch with thousands stays readable by hand.
    assert "fallback_occurrences" not in summary


def test_a_batch_with_no_model_calls_gets_no_summary_rather_than_zeros(tmp_path):
    """A deterministic run has no denominator; 0% fallback would be a fabricated measurement."""
    from jobrec.evaluation.llm_call_audit import manifest_summary

    run_dir = tmp_path / "full" / "SC-X-01" / "0"
    run_dir.mkdir(parents=True)
    (run_dir / "dialogue_state.json").write_text(
        json.dumps({"turns": [{"speaker": "candidate", "turn_id": "t0"}]}), encoding="utf-8")

    assert manifest_summary(tmp_path) == {}


def test_the_runner_puts_the_summary_in_the_experiment_manifest(tmp_path):
    """On the OFFICIAL path, beside the retry policy it is meant to be read against.

    A fallback rate that lives only in a separate audit script is a number nobody re-derives,
    and a batch whose model calls silently degraded would look identical to a clean one in its
    own manifest. Deterministic here, so the assertion is that the key EXISTS and is empty
    rather than zero-filled: a run with no model calls has no denominator, and a stated 0%
    would be a measurement that was never made.
    """
    from jobrec.config import load_config
    from jobrec.evaluation.experiment_identity import EXPERIMENT_MANIFEST_FILENAME
    from jobrec.evaluation.experiment_runner import ExperimentRunner

    scenarios = [json.loads(line) for line
                 in open("evaluation/data/scenarios_subset.jsonl", encoding="utf-8")
                 if line.strip()][:1]
    path = tmp_path / "s.jsonl"
    path.write_text(json.dumps(scenarios[0], default=str) + "\n", encoding="utf-8")

    config = load_config("configs/experiment_full.yaml", base_dir="configs")
    config.experiment.repeat_count = 1
    runner = ExperimentRunner(config, "data/processed/jobs.jsonl", str(path),
                              out_dir=str(tmp_path / "out"))
    manifest = runner.run(["full"])

    written = json.loads(
        (Path(manifest["experiment_dir"]) / EXPERIMENT_MANIFEST_FILENAME).read_text(
            encoding="utf-8"))
    assert "llm_call_summary" in written, "the manifest states a policy but not its outcome"
    assert written["llm_call_summary"] == {}
    # The policy it is read against is still there.
    assert set(written["retry_policy"]) == {"max_retries", "backoff_seconds",
                                            "timeout_seconds"}


# ---------------------------------------------------------------------------
# A fallback that is NOT a substituted preference still counts as an affected turn and run.
#
# ``final_fallback_turns`` / ``final_fallback_runs`` were read from the extraction snapshots
# alone. An exhausted CLARIFICATION call leaves the template question in place and changes no
# preference, so it registered nowhere: on a 126-run hybrid pilot the affected-run rate was
# reported as 1/126 = 0.79% where 4 runs had actually degraded (3.17%). The affected-run rate
# is a release threshold, so a fallback invisible to it is a fallback silently excluded from
# the metric.
# ---------------------------------------------------------------------------
def _exhausted_call(root, run: str, purpose: str, *, turn_index: int = 0):
    """One logical call of ``purpose`` that failed every attempt, in its own run dir."""
    variant, scenario, index = run.split("/")
    run_dir = root / variant / scenario / index
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "model_calls.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "call_id": f"call-{purpose}", "purpose": purpose,
            "parsed_ok": False, "raw_response": "",
            "turn_run_id": f"{scenario}-run", "turn_index": turn_index,
            "response_metadata": {"fell_back": True, "error": "LLMError", "attempts": 1},
        }) + "\n")
    (run_dir / "dialogue_state.json").write_text(
        json.dumps({"turns": [{"speaker": "candidate", "turn_id": f"t{i}",
                               "extraction_snapshot": {"preferences": []}}
                              for i in range(turn_index + 1)]}), encoding="utf-8")
    return run_dir


def test_an_exhausted_clarification_call_counts_as_an_affected_turn_and_run(tmp_path):
    from jobrec.evaluation.llm_call_audit import audit_llm_calls

    _exhausted_call(tmp_path, "full/SC-B-02/0", "clarification")
    _exhausted_call(tmp_path, "no_context/SC-B-01/0", "clarification")
    _exhausted_call(tmp_path, "no_memory/SC-D-12/0", "clarification", turn_index=2)
    # A fourth run that is clean, so the denominator is not all failures.
    clean = tmp_path / "full" / "SC-A-01" / "0"
    clean.mkdir(parents=True)
    (clean / "dialogue_state.json").write_text(
        json.dumps({"turns": [{"speaker": "candidate", "turn_id": "t0",
                               "extraction_snapshot": {"preferences": []}}]}),
        encoding="utf-8")

    report = audit_llm_calls(tmp_path)
    counts, rates = report["counts"], report["rates"]

    assert counts["total_runs"] == 4
    assert counts["failed_logical_calls"] == 3
    # The point of the test: these are NOT zero.
    assert counts["final_fallback_turns"] == 3
    assert counts["final_fallback_runs"] == 3
    assert rates["final_fallback_run_rate"] == 0.75
    # Attributed to the purpose, so a clarification fallback is not mistaken for a field one.
    assert report["fallback_distribution"]["by_field"] == {"<clarification>": 3}
    assert report["fallback_distribution"]["by_turn_index"] == {"0": 2, "2": 1}


def test_a_clarification_call_that_recovered_is_not_an_affected_run(tmp_path):
    """A rescued call got what it asked for; counting it would conflate the two."""
    from jobrec.evaluation.llm_call_audit import audit_llm_calls

    run_dir = _exhausted_call(tmp_path, "full/SC-B-02/0", "clarification")
    # A second attempt of the SAME logical call, this time successful.
    with (run_dir / "model_calls.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({
            "call_id": "call-clarification", "purpose": "clarification",
            "parsed_ok": True, "raw_response": "PHRASED",
            "turn_run_id": "SC-B-02-run", "turn_index": 0,
            "response_metadata": {"attempts": 1, "response_id": "resp-1"},
        }) + "\n")

    counts = audit_llm_calls(tmp_path)["counts"]

    assert counts["logical_calls"] == 1
    assert counts["failed_logical_calls"] == 0
    assert counts["recovered_after_retry"] == 1
    assert counts["final_fallback_turns"] == 0
    assert counts["final_fallback_runs"] == 0
