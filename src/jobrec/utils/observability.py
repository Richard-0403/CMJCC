"""Structured JSON logging and per-run log traces (R27).

Every diagnostic the system emits is a *structured record* carrying the same
identifying fields — ``{run_id, session_id, scenario_id, variant, component,
event, severity}`` (R27.1) — so logs can be filtered by run, session, scenario,
variant, component or event without parsing prose. Three diagnostic severities
are distinguished (R27.2):

``warning``
    A recovered or non-fatal condition (retrieval fell back to the full catalog,
    a model call was retried, a value was repaired).
``validation_error``
    A contract/schema violation: a field failed validation, a stated value could
    not be normalized and had to fall back.
``system_failure``
    The turn could not be completed and was converted into a failed run.

``info`` is also available for lifecycle bookkeeping (turn started/completed) so
a successful run still produces an inspectable trace; it is not one of the three
diagnostic severities.

Two consumers share one record builder:

* :class:`RunTrace` collects the records of a single turn/run in memory. The
  orchestrator attaches them to its ``TurnResult`` and
  :func:`jobrec.evaluation.exporters.write_run_bundle` exports them as
  ``log_trace.jsonl`` in the run bundle (R27.3) via :func:`write_log_trace`.
* the same records are mirrored to the stdlib :mod:`logging` tree under
  :data:`LOGGER_NAME`, where :class:`JsonFormatter` renders one JSON object per
  line. :func:`configure_logging` installs that formatter.

Records never carry credentials or PII: every message and every detail value
passes through :mod:`jobrec.utils.redaction`, honouring
``config.logging.redact_candidate_text`` exactly as the run-detail API does, so
candidate free text can be suppressed while the structural fields (ids, counts,
event names) stay inspectable.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .redaction import install_secret_log_filter, redact, redact_payload, secret_values
from .time import to_iso, utcnow

if TYPE_CHECKING:  # pragma: no cover - typing only, keeps utils import-light
    from ..config import AppConfig

__all__ = [
    "CONTEXT_FIELDS",
    "DIAGNOSTIC_SEVERITIES",
    "LOGGER_NAME",
    "LOG_TRACE_FILENAME",
    "RECORD_FIELDS",
    "SEVERITIES",
    "SEVERITY_INFO",
    "SEVERITY_SYSTEM_FAILURE",
    "SEVERITY_VALIDATION_ERROR",
    "SEVERITY_WARNING",
    "JsonFormatter",
    "LogContext",
    "RunTrace",
    "configure_logging",
    "run_trace",
    "write_log_trace",
]

#: Logger the structured records are mirrored to. Deliberately distinct from the
#: component loggers (e.g. ``jobrec.orchestration.orchestrator``) so structured
#: emission never interleaves with a component's own human-readable log stream.
LOGGER_NAME = "jobrec.trace"

#: Package logger :func:`configure_logging` attaches the JSON handler to.
ROOT_LOGGER_NAME = "jobrec"

#: Name of the per-run trace artifact inside a run bundle (R27.3).
LOG_TRACE_FILENAME = "log_trace.jsonl"

SEVERITY_INFO = "info"
SEVERITY_WARNING = "warning"
SEVERITY_VALIDATION_ERROR = "validation_error"
SEVERITY_SYSTEM_FAILURE = "system_failure"

#: The three severities R27.2 requires the system to distinguish.
DIAGNOSTIC_SEVERITIES: tuple[str, ...] = (
    SEVERITY_WARNING,
    SEVERITY_VALIDATION_ERROR,
    SEVERITY_SYSTEM_FAILURE,
)

#: Every accepted severity (the diagnostic three plus lifecycle ``info``).
SEVERITIES: tuple[str, ...] = (SEVERITY_INFO, *DIAGNOSTIC_SEVERITIES)

#: Identifying fields carried by every record (R27.1).
CONTEXT_FIELDS: tuple[str, ...] = ("run_id", "session_id", "scenario_id", "variant")

#: The full required field set of a record (R27.1/27.2).
RECORD_FIELDS: tuple[str, ...] = (*CONTEXT_FIELDS, "component", "event", "severity")

_SEVERITY_LEVELS: dict[str, int] = {
    SEVERITY_INFO: logging.INFO,
    SEVERITY_WARNING: logging.WARNING,
    SEVERITY_VALIDATION_ERROR: logging.ERROR,
    SEVERITY_SYSTEM_FAILURE: logging.CRITICAL,
}

# Reverse mapping used to give records that were NOT emitted through RunTrace
# (a plain ``logger.warning`` in a component) a severity in the JSON output.
_LEVEL_SEVERITIES: tuple[tuple[int, str], ...] = (
    (logging.CRITICAL, SEVERITY_SYSTEM_FAILURE),
    (logging.ERROR, SEVERITY_VALIDATION_ERROR),
    (logging.WARNING, SEVERITY_WARNING),
    (logging.NOTSET, SEVERITY_INFO),
)

#: Marker attribute used to recognise (and not duplicate) our own handler.
_HANDLER_MARKER = "_jobrec_json_handler"


def severity_for_level(level: int) -> str:
    """Map a stdlib logging level onto the closest structured severity."""
    for threshold, severity in _LEVEL_SEVERITIES:
        if level >= threshold:
            return severity
    return SEVERITY_INFO


@dataclass(frozen=True)
class LogContext:
    """The identifying fields shared by every record of one run (R27.1)."""

    run_id: str | None = None
    session_id: str | None = None
    scenario_id: str | None = None
    variant: str | None = None

    def as_dict(self) -> dict[str, str | None]:
        return asdict(self)


class JsonFormatter(logging.Formatter):
    """Render a log record as a single-line JSON object (R27.1).

    Records emitted through :class:`RunTrace` carry the already-built structured
    payload on ``record.structured`` and are rendered verbatim. Records emitted
    by a plain component logger are still rendered as JSON on a best-effort
    basis: the logger name becomes ``component``, ``module.function`` becomes
    ``event``, the level maps onto a severity, and the identifying fields are
    ``None`` because that call site did not supply them.
    """

    def format(self, record: logging.LogRecord) -> str:  # noqa: A003 - stdlib API
        structured = getattr(record, "structured", None)
        if isinstance(structured, Mapping):
            payload: dict[str, Any] = dict(structured)
        else:
            payload = {
                "timestamp": to_iso(datetime.fromtimestamp(record.created, UTC)),
                **dict.fromkeys(CONTEXT_FIELDS),
                "component": record.name,
                "event": f"{record.module}.{record.funcName}",
                "severity": severity_for_level(record.levelno),
                "message": record.getMessage(),
                "detail": {},
            }
        payload.setdefault("logger", record.name)
        if record.exc_info:
            # A traceback can quote a request body or an env dump, so it goes
            # through the same redactor as the message (R26.1).
            payload["exception"] = redact(
                self.formatException(record.exc_info), secrets=secret_values()
            )
        return json.dumps(payload, default=str)


def configure_logging(
    config: AppConfig | None = None, *, stream: Any = None, force: bool = False
) -> logging.Logger:
    """Install the JSON log handler on the ``jobrec`` package logger (R27.1).

    Reads the level and format from ``config.logging`` (defaults ``INFO``/``json``);
    a non-``json`` format keeps a plain text formatter so a developer can opt out.
    Idempotent: calling it repeatedly reuses the handler it installed rather than
    stacking duplicates, unless ``force`` replaces it (e.g. to retarget ``stream``).

    The handler always carries a
    :class:`~jobrec.utils.redaction.SecretLogFilter`, so nothing formatted by it
    can contain an API key (R26.1).
    """
    settings = getattr(config, "logging", None)
    level = str(getattr(settings, "level", "INFO") or "INFO").upper()
    fmt = str(getattr(settings, "format", "json") or "json").lower()

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    logger.setLevel(level)

    existing = next(
        (h for h in logger.handlers if getattr(h, _HANDLER_MARKER, False)), None
    )
    if existing is not None:
        if not force:
            existing.setFormatter(JsonFormatter() if fmt == "json" else logging.Formatter())
            existing.setLevel(level)
            install_secret_log_filter(existing)
            return logger
        logger.removeHandler(existing)

    handler = logging.StreamHandler(stream) if stream is not None else logging.StreamHandler()
    setattr(handler, _HANDLER_MARKER, True)
    handler.setLevel(level)
    install_secret_log_filter(handler)
    handler.setFormatter(
        JsonFormatter()
        if fmt == "json"
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    logger.addHandler(handler)
    return logger


class RunTrace:
    """Structured-record collector for a single run (R27.1–27.3).

    Each :meth:`emit` builds one record, redacts it, appends it to the in-memory
    trace and mirrors it to the :data:`LOGGER_NAME` logger. ``records`` is the
    per-run trace that gets exported as ``log_trace.jsonl``.

    A trace built without a context is still usable (all identifying fields are
    ``None``); that is what direct unit-level calls into a component get.
    """

    def __init__(
        self,
        context: LogContext | None = None,
        *,
        redact_candidate_text: bool = False,
        logger: logging.Logger | None = None,
        emit_to_logger: bool = True,
    ) -> None:
        self.context = context or LogContext()
        self.redact_candidate_text = redact_candidate_text
        self._logger = logger or logging.getLogger(LOGGER_NAME)
        self._emit_to_logger = emit_to_logger
        self._records: list[dict[str, Any]] = []

    # ------------------------------------------------------------- emission
    def emit(
        self,
        component: str,
        event: str,
        severity: str,
        message: str | None = None,
        **detail: Any,
    ) -> dict[str, Any]:
        """Record one structured event and return it.

        ``severity`` must be one of :data:`SEVERITIES`; an unknown severity is a
        programming error and raises ``ValueError`` rather than being silently
        downgraded, so no record can escape with an uninterpretable severity.
        """
        if severity not in SEVERITIES:
            raise ValueError(
                f"unknown severity {severity!r}; expected one of {list(SEVERITIES)}"
            )
        # The exported trace is a log artifact too, so literal API-key values are
        # masked here as well as in the handler filter (R26.1).
        secrets = secret_values()
        record: dict[str, Any] = {
            "timestamp": to_iso(utcnow()),
            **self.context.as_dict(),
            "component": component,
            "event": event,
            "severity": severity,
            "message": (
                redact(
                    message,
                    redact_candidate_text=self.redact_candidate_text,
                    secrets=secrets,
                )
                if message
                else None
            ),
            "detail": redact_payload(
                dict(detail),
                redact_candidate_text=self.redact_candidate_text,
                secrets=secrets,
            ),
        }
        self._records.append(record)
        if self._emit_to_logger:
            self._logger.log(
                _SEVERITY_LEVELS[severity],
                "%s.%s",
                component,
                event,
                extra={"structured": record},
            )
        return record

    def info(self, component: str, event: str, message: str | None = None, **detail: Any):
        """Lifecycle bookkeeping (turn started/completed)."""
        return self.emit(component, event, SEVERITY_INFO, message, **detail)

    def warning(self, component: str, event: str, message: str | None = None, **detail: Any):
        """A recovered or non-fatal condition (R27.2)."""
        return self.emit(component, event, SEVERITY_WARNING, message, **detail)

    def validation_error(
        self, component: str, event: str, message: str | None = None, **detail: Any
    ):
        """A schema/contract violation on a field or payload (R27.2)."""
        return self.emit(component, event, SEVERITY_VALIDATION_ERROR, message, **detail)

    def system_failure(
        self, component: str, event: str, message: str | None = None, **detail: Any
    ):
        """The run could not be completed (R27.2)."""
        return self.emit(component, event, SEVERITY_SYSTEM_FAILURE, message, **detail)

    # -------------------------------------------------------------- readout
    @property
    def records(self) -> list[dict[str, Any]]:
        """A copy of the trace, in emission order."""
        return [dict(record) for record in self._records]

    def extend(self, records: Iterable[Mapping[str, Any]]) -> None:
        """Append already-built records (e.g. merging another turn's trace)."""
        self._records.extend(dict(record) for record in records)

    def __len__(self) -> int:
        return len(self._records)


def run_trace(
    config: AppConfig | None = None,
    *,
    run_id: str | None = None,
    session_id: str | None = None,
    scenario_id: str | None = None,
    variant: str | None = None,
    logger: logging.Logger | None = None,
) -> RunTrace:
    """Build a :class:`RunTrace` for one run from the resolved config.

    The variant defaults to ``config.experiment.variant`` and the redaction
    behaviour to ``config.logging.redact_candidate_text``, so callers only pass
    the run/session/scenario identifiers.
    """
    if variant is None:
        resolved = getattr(getattr(config, "experiment", None), "variant", None)
        variant = getattr(resolved, "value", resolved)
    redact_text = bool(
        getattr(getattr(config, "logging", None), "redact_candidate_text", False)
    )
    return RunTrace(
        LogContext(
            run_id=run_id,
            session_id=session_id,
            scenario_id=scenario_id,
            variant=variant,
        ),
        redact_candidate_text=redact_text,
        logger=logger,
    )


def write_log_trace(out_dir: str | Path, records: Iterable[Mapping[str, Any]]) -> Path:
    """Write the per-run trace as ``log_trace.jsonl`` and return its path (R27.3).

    One JSON object per line, in emission order, so the artifact streams and diffs
    like the bundle's other ``*.jsonl`` files.
    """
    path = Path(out_dir) / LOG_TRACE_FILENAME
    with path.open("w") as fh:
        for record in records or ():
            fh.write(json.dumps(dict(record), default=str))
            fh.write("\n")
    return path
