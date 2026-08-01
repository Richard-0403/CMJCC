"""HTTP attempts are counted, not indexed.

``_failed_call_record`` stored the retry's 1-based INDEX under the key ``attempts``, so four
consecutive failures recorded 1, 2, 3, 4 and any sum over them reported TEN attempts for four
requests. That number is the denominator the endpoint fitness decision rests on -- fallback
rate per attempt -- so inflating it made the endpoint look better per-attempt than it is.

The index is still useful (it says which retry a record belongs to) and is kept under its own
name, so nothing is lost by making ``attempts`` mean what it says.
"""

from __future__ import annotations

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


def test_the_diagnosis_reports_four_attempts_for_four_failures(tmp_path):
    """End to end through the report that feeds the endpoint decision."""
    import json

    from scripts.diagnose_llm_fallbacks import diagnose

    run_dir = tmp_path / "full" / "SC-X-01" / "0"
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
    (run_dir / "dialogue_state.json").write_text(
        json.dumps({"turns": [{"speaker": "candidate", "turn_id": "t0",
                               "extraction_snapshot": None}]}), encoding="utf-8")

    counts = diagnose(tmp_path)["counts"]
    assert counts["http_attempts"] == 4, counts
    # Four records, one logical call (same run, turn and purpose), which never succeeded.
    assert counts["call_records"] == 4
    assert counts["logical_calls"] == 1
    assert counts["failed_logical_calls"] == 1
    assert counts["retry_attempts"] == 3
