"""Unit tests for structured JSON logging and the per-run log trace (R27).

Covers the three things Requirement 27 asks for, against the real logger, the real
orchestrator and the real bundle writer (no mocks):

* **R27.1** every record carries ``{run_id, session_id, scenario_id, variant,
  component, event, severity}``, and the JSON formatter renders it as one parseable
  JSON object per line;
* **R27.2** the ``warning`` / ``validation_error`` / ``system_failure`` severities are
  distinguished, both in the record and in the stdlib level they are emitted at;
* **R27.3** a real turn's records are exported as ``log_trace.jsonl`` in the run
  bundle, including the ``system_failure`` record of a failed run.

Redaction is asserted through :mod:`jobrec.utils.redaction`: credentials never reach a
record, and ``config.logging.redact_candidate_text`` suppresses candidate free text.
"""

from __future__ import annotations

import io
import json
import logging
from pathlib import Path

import pytest

from jobrec.app_service import AppService
from jobrec.config import AppConfig, load_config
from jobrec.evaluation.exporters import write_run_bundle
from jobrec.retrieval.base import RetrievalOutcome
from jobrec.utils.observability import (
    DIAGNOSTIC_SEVERITIES,
    LOG_TRACE_FILENAME,
    LOGGER_NAME,
    RECORD_FIELDS,
    SEVERITY_SYSTEM_FAILURE,
    SEVERITY_WARNING,
    JsonFormatter,
    LogContext,
    RunTrace,
    configure_logging,
    run_trace,
)
from jobrec.utils.redaction import REDACTED, REDACTED_KEY
from tests.conftest import CATALOG_PATH

UTTERANCE = "I want a data analyst role in Kuala Lumpur, hybrid, at least RM4000."


def _context() -> LogContext:
    return LogContext(
        run_id="run-r27", session_id="sess-r27", scenario_id="SC-R27", variant="full"
    )


# ------------------------------------------------------------------ record shape
def test_every_record_carries_the_required_identifying_fields() -> None:
    """R27.1: run/session/scenario/variant/component/event/severity on every record."""
    trace = RunTrace(_context())

    trace.info("orchestrator", "turn_started")
    trace.warning("hybrid_retriever", "empty_recall_fallback", catalog_size=6)
    trace.validation_error("candidate_understanding", "extraction_field_validation_failed")
    trace.system_failure("orchestrator", "run_failed")

    records = trace.records
    assert len(records) == 4
    for record in records:
        assert set(RECORD_FIELDS) <= set(record)
        assert record["run_id"] == "run-r27"
        assert record["session_id"] == "sess-r27"
        assert record["scenario_id"] == "SC-R27"
        assert record["variant"] == "full"
        assert record["timestamp"]
    assert [r["severity"] for r in records[1:]] == list(DIAGNOSTIC_SEVERITIES)
    assert records[1]["detail"] == {"catalog_size": 6}


def test_severities_map_onto_distinct_log_levels_and_render_as_json() -> None:
    """R27.1/27.2: records reach the logger as JSON, one severity per level."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    levels: list[int] = []
    handler.addFilter(lambda record: bool(levels.append(record.levelno)) or True)
    logger = logging.getLogger("jobrec.trace.test-severities")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    try:
        trace = RunTrace(_context(), logger=logger)
        trace.warning("retrieval", "empty_recall_fallback")
        trace.validation_error("candidate_understanding", "field_invalid")
        trace.system_failure("orchestrator", "run_failed")
    finally:
        logger.removeHandler(handler)

    lines = [line for line in stream.getvalue().splitlines() if line.strip()]
    emitted = [json.loads(line) for line in lines]
    assert [r["severity"] for r in emitted] == list(DIAGNOSTIC_SEVERITIES)
    assert [r["event"] for r in emitted] == [
        "empty_recall_fallback", "field_invalid", "run_failed"
    ]
    # Distinguishable without parsing the payload, too: one level per severity.
    assert levels == [logging.WARNING, logging.ERROR, logging.CRITICAL]


def test_an_unknown_severity_is_rejected() -> None:
    """A severity outside the documented set is a programming error, not a downgrade."""
    with pytest.raises(ValueError, match="unknown severity"):
        RunTrace(_context()).emit("orchestrator", "whatever", "catastrophe")


def test_configure_logging_installs_one_json_handler() -> None:
    """The handler is installed on the package logger and never stacked twice (R27.1)."""
    logger = logging.getLogger("jobrec")
    before = list(logger.handlers)
    try:
        stream = io.StringIO()
        configure_logging(AppConfig(), stream=stream, force=True)
        installed = [h for h in logger.handlers if h not in before]
        assert len(installed) == 1
        configure_logging(AppConfig())
        assert len([h for h in logger.handlers if h not in before]) == 1

        logging.getLogger(LOGGER_NAME).warning("plain component message")
        record = json.loads(stream.getvalue().splitlines()[0])
        assert record["severity"] == SEVERITY_WARNING
        assert record["message"] == "plain component message"
    finally:
        for handler in [h for h in logger.handlers if h not in before]:
            logger.removeHandler(handler)


# --------------------------------------------------------------------- redaction
def test_records_never_carry_credentials_or_redacted_candidate_text() -> None:
    """Credentials always stripped; candidate free text suppressed on request (R26.1/R12.3)."""
    trace = RunTrace(_context())

    record = trace.warning(
        "llm", "provider_error", "call failed with api_key=sk-abcd1234efgh",
        request={"authorization": "Bearer sk-abcd1234efgh5678"},
    )

    assert "sk-abcd1234efgh" not in json.dumps(record)
    assert REDACTED_KEY in record["message"]

    config = AppConfig()
    config.logging.redact_candidate_text = True
    redacting = run_trace(config, run_id="run-r27", session_id="sess-r27")
    redacted = redacting.warning(
        "candidate_understanding", "extraction_field_unconfirmed",
        "salary_min: value could not be normalized",
        utterance="I am Aisyah and I want RM4000",
    )

    assert redacted["message"] == REDACTED
    assert redacted["detail"]["utterance"] == REDACTED
    # Structural fields survive so the trace is still filterable.
    assert redacted["event"] == "extraction_field_unconfirmed"
    assert redacted["variant"] == "full"


# ---------------------------------------------------------- end-to-end artifact
def test_a_real_turn_exports_a_well_formed_log_trace(tmp_path: Path) -> None:
    """R27.3: a real run's records land in ``log_trace.jsonl`` in the bundle."""
    config = load_config("configs/experiment_full.yaml", base_dir="configs")
    service = AppService(config, CATALOG_PATH)
    candidate = service.create_candidate(
        {"candidate_id": "cand-r27", "skills": ["Python", "SQL"], "years_experience": 3}
    )
    session_id = service.create_session(candidate.candidate_id, "full")

    result = service.process_turn(session_id, UTTERANCE, scenario_id="SC-R27")
    out = write_run_bundle(result, tmp_path / "run", config)

    lines = [
        line
        for line in (out / LOG_TRACE_FILENAME).read_text().splitlines()
        if line.strip()
    ]
    records = [json.loads(line) for line in lines]
    assert records, "the run bundle carries no structured log records"
    for record in records:
        assert set(RECORD_FIELDS) <= set(record)
        assert record["run_id"] == result.run_record.run_id
        assert record["session_id"] == session_id
        assert record["scenario_id"] == "SC-R27"
        assert record["variant"] == "full"
    events = [record["event"] for record in records]
    assert events[0] == "turn_started"
    assert events[-1] == "turn_completed"
    assert records[-1]["detail"]["success"] is True


def test_a_failed_turn_records_a_system_failure(tmp_path: Path, monkeypatch) -> None:
    """R27.2/27.3: a run converted into a failure carries a ``system_failure`` record."""
    config = load_config("configs/experiment_full.yaml", base_dir="configs")
    service = AppService(config, CATALOG_PATH)
    candidate = service.create_candidate(
        {"candidate_id": "cand-r27-fail", "skills": ["Python"], "years_experience": 2}
    )
    session_id = service.create_session(candidate.candidate_id, "full")
    orchestrator, _store = service._orchestrator_for(session_id, "full")

    def boom(*_args, **_kwargs):
        raise RuntimeError("ranking exploded")

    monkeypatch.setattr(orchestrator.ranking, "rank", boom)

    result = service.process_turn(session_id, UTTERANCE, scenario_id="SC-R27-fail")
    out = write_run_bundle(result, tmp_path / "run", config)

    assert result.run_record.success is False
    records = [
        json.loads(line)
        for line in (out / LOG_TRACE_FILENAME).read_text().splitlines()
        if line.strip()
    ]
    failures = [r for r in records if r["severity"] == SEVERITY_SYSTEM_FAILURE]
    assert len(failures) == 1
    assert failures[0]["event"] == "run_failed"
    assert failures[0]["detail"]["error_type"] == "RuntimeError"
    assert records[-1]["detail"]["success"] is False


def test_a_recovered_condition_is_recorded_as_a_warning(monkeypatch) -> None:
    """R27.2: the empty-recall fallback is a ``warning``; the run still succeeds."""
    config = load_config("configs/experiment_full.yaml", base_dir="configs")
    service = AppService(config, CATALOG_PATH)
    candidate = service.create_candidate(
        {"candidate_id": "cand-r27-recall", "skills": ["Python"], "years_experience": 2}
    )
    session_id = service.create_session(candidate.candidate_id, "full")
    orchestrator, _store = service._orchestrator_for(session_id, "full")

    def recall_nothing(_query, _jobs, _pool_size):
        return RetrievalOutcome(retrieved=[], initial_pool_size=0)

    monkeypatch.setattr(orchestrator.retriever, "retrieve", recall_nothing)

    result = service.process_turn(session_id, UTTERANCE, scenario_id="SC-R27-recall")

    warnings = [r for r in result.log_trace if r["severity"] == SEVERITY_WARNING]
    assert [r["event"] for r in warnings] == ["empty_recall_fallback"]
    assert result.run_record.success is True
    assert not [r for r in result.log_trace if r["severity"] == SEVERITY_SYSTEM_FAILURE]
