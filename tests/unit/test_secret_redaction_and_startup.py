"""Unit tests for secret redaction in logs and startup validation (R26.1/26.3/26.4).

R26.1: keys are read from the environment and never written to any log. The tests
here exercise the machinery that makes that true rather than the credential-shape
patterns (already covered by ``tests/unit/test_run_detail_redaction.py``):

* :func:`~jobrec.utils.redaction.secret_env_names` /
  :func:`~jobrec.utils.redaction.secret_values` discover secret-shaped variables and
  their literal values, subject to the :data:`MIN_SECRET_LENGTH` floor and a
  longest-first masking order;
* :class:`~jobrec.utils.redaction.SecretLogFilter` stops a key from reaching a
  handler's *output*, whether the key arrives in the message, in ``record.args``,
  in a structured payload or in a formatted traceback — without mutating the
  caller's payload;
* :class:`~jobrec.llm.remote_provider.RemoteLLMProvider` reads the key from the
  environment only and never exposes it through ``repr``, its manifest or a raised
  transport error.

R26.3/26.4: :func:`~jobrec.app_service.validate_startup` validates the resolved
configuration (and the API environment a hybrid/remote run needs) at startup and
fails fast with a single ``RuntimeError(ErrorCode.CONFIG_INVALID, ...)`` listing
every problem, before the database is touched.

Every key used below is obviously fake and is injected through ``monkeypatch`` so no
environment state leaks between tests.
"""

from __future__ import annotations

import io
import json
import logging
from collections.abc import Callable

import httpx
import pytest

from jobrec import app_service as app_service_module
from jobrec.app_service import build_default_service, validate_startup
from jobrec.config import AppConfig, load_config
from jobrec.domain.enums import ErrorCode, RunMode
from jobrec.llm import remote_provider as remote_provider_module
from jobrec.llm.provider import LLMError, LLMTimeout
from jobrec.llm.remote_provider import API_KEY_ENV, RemoteLLMProvider
from jobrec.storage import db as db_module
from jobrec.utils.observability import JsonFormatter
from jobrec.utils.redaction import (
    MIN_SECRET_LENGTH,
    REDACTED_KEY,
    SecretLogFilter,
    install_secret_log_filter,
    redact,
    secret_env_names,
    secret_values,
)
from tests.conftest import CATALOG_PATH

#: Obvious fakes. Deliberately carry no credential shape (no ``sk-``/``Bearer``
#: prefix), so a test that shows them redacted proves the *env-value* masking
#: worked rather than a pattern match.
FAKE_KEY = "zzfakekeyvaluezz00000001"
FAKE_KEY_PREFIX = "zzfakekeyvaluezz"
FAKE_KEY_ENV = "JOBREC_TEST_FAKE_API_KEY"

SHORT_VALUE = "s" * (MIN_SECRET_LENGTH - 1)
FLOOR_VALUE = "f" * MIN_SECRET_LENGTH


# ----------------------------------------------------------- env-secret discovery
def test_secret_env_names_matches_secret_shaped_names_only() -> None:
    """Names are matched on their suffix, so provider-specific names are covered."""
    env = {
        "JOBREC_LLM_API_KEY": FAKE_KEY,
        "OPENAI_API_KEY": FAKE_KEY,
        "PGPASSWORD": FAKE_KEY,
        "SERVICE_AUTH_TOKEN": FAKE_KEY,
        "CLIENT_SECRET": FAKE_KEY,
        # Not secrets: a URL, a model name, and a variable naming *another*
        # variable (it holds a name, not a value).
        "JOBREC_LLM_BASE_URL": "https://example.invalid/v1",
        "JOBREC_LLM_MODEL": "gpt-4o-mini",
        "JOBREC_API_KEY_ENV": "JOBREC_LLM_API_KEY",
    }

    assert secret_env_names(env) == (
        "CLIENT_SECRET",
        "JOBREC_LLM_API_KEY",
        "OPENAI_API_KEY",
        "PGPASSWORD",
        "SERVICE_AUTH_TOKEN",
    )


def test_secret_env_names_and_values_default_to_the_process_environment(monkeypatch) -> None:
    """With no explicit mapping the live environment is consulted."""
    monkeypatch.setenv(FAKE_KEY_ENV, FAKE_KEY)

    assert FAKE_KEY_ENV in secret_env_names()
    assert FAKE_KEY in secret_values()


def test_values_below_the_length_floor_are_not_masked() -> None:
    """A short, word-like value must not blank unrelated substrings in every line."""
    env = {"DEV_PASSWORD": SHORT_VALUE, "DEV_API_KEY": FLOOR_VALUE}

    assert secret_values(env) == (FLOOR_VALUE,)

    text = f"local value {SHORT_VALUE} and key {FLOOR_VALUE}"
    cleaned = redact(text, secrets=secret_values(env))
    assert SHORT_VALUE in cleaned
    assert FLOOR_VALUE not in cleaned


def test_secret_values_are_masked_longest_first() -> None:
    """When one value is a prefix of another, the longer one is masked first."""
    env = {"A_API_KEY": FAKE_KEY_PREFIX, "B_API_KEY": FAKE_KEY}
    values = secret_values(env)

    assert values == (FAKE_KEY, FAKE_KEY_PREFIX)
    # Shortest-first would have left the "00000001" tail behind.
    assert redact(f"sent {FAKE_KEY} upstream", secrets=values) == f"sent {REDACTED_KEY} upstream"
    # The value carries no credential shape: without the env values it survives,
    # which is exactly why the env machinery exists.
    assert FAKE_KEY in redact(f"sent {FAKE_KEY} upstream")


def test_secret_values_can_be_restricted_to_explicit_names() -> None:
    """Explicit ``names`` win over discovery, including for a non-secret-shaped name."""
    env = {"A_API_KEY": FAKE_KEY, "B_API_KEY": FLOOR_VALUE, "CUSTOM_CREDENTIAL": FAKE_KEY_PREFIX}

    assert secret_values(env, ["A_API_KEY"]) == (FAKE_KEY,)
    assert secret_values(env, ["CUSTOM_CREDENTIAL"]) == (FAKE_KEY_PREFIX,)
    assert secret_values(env, ["MISSING_API_KEY"]) == ()


# -------------------------------------------------------------- SecretLogFilter
@pytest.fixture()
def json_log(monkeypatch):
    """A logger whose handler renders JSON and carries the secret filter."""
    monkeypatch.setenv(FAKE_KEY_ENV, FAKE_KEY)
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    install_secret_log_filter(handler)
    logger = logging.getLogger("jobrec.tests.secret_filter")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    try:
        yield logger, stream
    finally:
        logger.removeHandler(handler)


def _records(stream: io.StringIO) -> list[dict]:
    return [json.loads(line) for line in stream.getvalue().splitlines() if line.strip()]


def test_a_key_in_the_message_never_reaches_the_handler_output(json_log) -> None:
    """R26.1: the filter rewrites the record before any handler can format it."""
    logger, stream = json_log

    logger.warning(f"calling upstream with {FAKE_KEY}")

    output = stream.getvalue()
    assert FAKE_KEY not in output
    assert REDACTED_KEY in _records(stream)[0]["message"]


def test_a_key_supplied_through_record_args_is_masked(json_log) -> None:
    """Lazy ``%s`` formatting must not smuggle the value past the filter."""
    logger, stream = json_log

    logger.error("upstream rejected credential %s for model %s", FAKE_KEY, "gpt-4o-mini")

    output = stream.getvalue()
    assert FAKE_KEY not in output
    message = _records(stream)[0]["message"]
    assert REDACTED_KEY in message
    # Non-secret arguments are still interpolated, so the message stays useful.
    assert "gpt-4o-mini" in message


def test_a_key_in_a_structured_payload_is_masked_without_mutating_the_caller(
    json_log,
) -> None:
    """The record gets a redacted *copy*; the caller's in-memory trace is untouched."""
    logger, stream = json_log
    structured = {
        "run_id": "run-r26",
        "component": "llm",
        "event": "provider_error",
        "severity": "warning",
        "message": "call failed",
        "detail": {"authorization": FAKE_KEY, "attempt": 2},
    }

    logger.warning("%s.%s", "llm", "provider_error", extra={"structured": structured})

    output = stream.getvalue()
    assert FAKE_KEY not in output
    emitted = _records(stream)[0]
    assert emitted["detail"]["authorization"] == REDACTED_KEY
    # Structural fields survive, so the record is still filterable.
    assert emitted["detail"]["attempt"] == 2
    assert emitted["run_id"] == "run-r26"
    # The dict the caller still owns was not rewritten behind its back.
    assert structured["detail"]["authorization"] == FAKE_KEY


def test_a_key_quoted_in_a_traceback_is_masked(json_log) -> None:
    """A traceback can quote a request body, so it goes through the same redactor."""
    logger, stream = json_log

    try:
        raise RuntimeError(f"upstream refused credential {FAKE_KEY}")
    except RuntimeError:
        logger.exception("remote call failed")

    output = stream.getvalue()
    assert FAKE_KEY not in output
    assert REDACTED_KEY in _records(stream)[0]["exception"]


def test_a_key_exported_after_logging_was_configured_is_still_masked(monkeypatch) -> None:
    """The environment is read per record, not once at install time."""
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(JsonFormatter())
    log_filter = install_secret_log_filter(handler)
    assert FAKE_KEY not in log_filter.secrets()

    logger = logging.getLogger("jobrec.tests.secret_filter_late_env")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    try:
        monkeypatch.setenv(FAKE_KEY_ENV, FAKE_KEY)
        logger.warning("late export %s", FAKE_KEY)
    finally:
        logger.removeHandler(handler)

    assert FAKE_KEY not in stream.getvalue()
    assert REDACTED_KEY in _records(stream)[0]["message"]


def test_install_secret_log_filter_is_idempotent() -> None:
    """Repeated installs reuse one filter instead of stacking duplicates."""
    logger = logging.getLogger("jobrec.tests.secret_filter_idempotent")
    first = install_secret_log_filter(logger)
    try:
        second = install_secret_log_filter(logger)
        assert first is second
        assert [f for f in logger.filters if isinstance(f, SecretLogFilter)] == [first]
    finally:
        logger.removeFilter(first)


def test_the_filter_never_drops_a_record(json_log) -> None:
    """Redaction rewrites records; it must not silently lose diagnostics."""
    logger, stream = json_log

    logger.info("nothing secret here")
    logger.warning("but this has %s", FAKE_KEY)

    assert len(_records(stream)) == 2


# --------------------------------------------------------------- remote provider
def test_remote_provider_reads_the_key_from_the_environment_only(monkeypatch) -> None:
    """R26.1: env-only key, absent from ``repr`` and from the persisted manifest."""
    monkeypatch.setenv(API_KEY_ENV, FAKE_KEY)

    provider = RemoteLLMProvider()

    assert provider.has_api_key is True
    assert not hasattr(provider, "api_key")
    assert FAKE_KEY not in repr(provider)
    manifest = provider.manifest()
    assert FAKE_KEY not in json.dumps(manifest)
    # The manifest records where the key came FROM, never the value.
    assert manifest["api_key_env"] == API_KEY_ENV
    assert manifest["api_key_present"] is True


def test_remote_provider_without_a_key_reports_the_variable_name(monkeypatch) -> None:
    """A missing key is an explicit, actionable failure naming the env variable."""
    monkeypatch.delenv(API_KEY_ENV, raising=False)

    provider = RemoteLLMProvider()

    assert provider.has_api_key is False
    assert provider.manifest()["api_key_present"] is False
    with pytest.raises(LLMTimeout) as excinfo:
        provider.complete_json("hello", purpose="extraction")
    assert API_KEY_ENV in str(excinfo.value)


def test_transport_errors_are_scrubbed_before_being_raised(monkeypatch) -> None:
    """A transport error that quoted the request must not carry the key upward."""
    monkeypatch.setenv(API_KEY_ENV, FAKE_KEY)
    provider = RemoteLLMProvider()

    def boom(_body: dict) -> dict:
        raise httpx.ConnectError(f"connection refused (Authorization: Bearer {FAKE_KEY})")

    monkeypatch.setattr(provider, "_post", boom)

    with pytest.raises(LLMError) as excinfo:
        provider.complete_json("hello", purpose="extraction")

    assert FAKE_KEY not in str(excinfo.value)
    assert REDACTED_KEY in str(excinfo.value)


def test_timeout_errors_are_scrubbed_before_being_raised(monkeypatch) -> None:
    """The timeout branch scrubs too: an httpx timeout quoting the key must not escape.

    ``httpx.TimeoutException`` is caught separately from the other transport errors and
    re-raised as :class:`LLMTimeout`, so it needs its own case - the exception has to be
    rendered to text before it is scrubbed, otherwise the key would ride along untouched.
    """
    monkeypatch.setenv(API_KEY_ENV, FAKE_KEY)
    provider = RemoteLLMProvider()

    def boom(_body: dict) -> dict:
        raise httpx.ReadTimeout(f"timed out (Authorization: Bearer {FAKE_KEY})")

    monkeypatch.setattr(provider, "_post", boom)

    with pytest.raises(LLMTimeout) as excinfo:
        provider.complete_json("hello", purpose="extraction")

    assert FAKE_KEY not in str(excinfo.value)
    assert REDACTED_KEY in str(excinfo.value)


def test_the_provider_logger_scrubs_its_own_records(monkeypatch) -> None:
    """The module logger carries the filter, whatever handler is attached to it."""
    monkeypatch.setenv(API_KEY_ENV, FAKE_KEY)
    logger = remote_provider_module.logger
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    propagate = logger.propagate
    logger.addHandler(handler)
    logger.propagate = False
    try:
        logger.warning("posting with %s", FAKE_KEY)
    finally:
        logger.removeHandler(handler)
        logger.propagate = propagate

    assert FAKE_KEY not in stream.getvalue()
    assert REDACTED_KEY in stream.getvalue()


# ------------------------------------------------------------- startup validation
def _hybrid_remote(config: AppConfig) -> None:
    config.llm.mode = RunMode.HYBRID
    config.llm.provider = "remote"


#: One invalid config per class of problem, with a fragment of its expected report.
INVALID_CONFIGS: list[tuple[str, Callable[[AppConfig], None], str]] = [
    ("reference_date", lambda c: setattr(c.project, "reference_date", "01-01-2026"),
     "project.reference_date"),
    ("top_k", lambda c: setattr(c.experiment, "top_k", 0), "experiment.top_k"),
    ("pool_smaller_than_top_k",
     lambda c: setattr(c.experiment, "retrieval_pool_size", 1),
     "experiment.retrieval_pool_size"),
    ("repeat_count", lambda c: setattr(c.experiment, "repeat_count", 0),
     "experiment.repeat_count"),
    ("max_dialogue_turns", lambda c: setattr(c.experiment, "max_dialogue_turns", 0),
     "experiment.max_dialogue_turns"),
    ("clarification_threshold",
     lambda c: setattr(c.memory, "clarification_confidence_threshold", 1.5),
     "memory.clarification_confidence_threshold"),
    ("empty_weights", lambda c: setattr(c.ranking, "weights", {}), "ranking.weights"),
    ("negative_weight",
     lambda c: setattr(c.ranking, "weights", {"role_match": -0.5}), "non-negative"),
    ("salary_scale", lambda c: setattr(c.ranking, "salary_scale", 0.0),
     "ranking.salary_scale"),
    ("retrieval_weights", lambda c: [setattr(c.retrieval, name, 0.0) for name in
                                     ("lexical_weight", "semantic_weight", "structured_weight")],
     "retrieval weights"),
    ("llm_timeout", lambda c: setattr(c.llm, "timeout_seconds", 0),
     "llm.timeout_seconds"),
    ("llm_max_retries", lambda c: setattr(c.llm, "max_retries", -1), "llm.max_retries"),
    ("logging_level", lambda c: setattr(c.logging, "level", "CHATTY"), "logging.level"),
    ("hybrid_remote_without_key", _hybrid_remote, API_KEY_ENV),
]


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [(mutate, fragment) for _id, mutate, fragment in INVALID_CONFIGS],
    ids=[case_id for case_id, _mutate, _fragment in INVALID_CONFIGS],
)
def test_each_invalid_config_fails_fast_with_config_invalid(mutate, fragment) -> None:
    """R26.3/26.4: every class of missing/invalid configuration is caught at startup."""
    config = AppConfig()
    mutate(config)

    with pytest.raises(RuntimeError) as excinfo:
        validate_startup(config, env={})

    assert excinfo.value.args[0] is ErrorCode.CONFIG_INVALID
    assert "CONFIG_INVALID" in str(excinfo.value)
    assert fragment in str(excinfo.value)


@pytest.mark.parametrize(
    ("mode", "provider"),
    [
        (RunMode.DETERMINISTIC, "mock"),
        (RunMode.REPLAY, "replay"),
        # Hybrid through a local provider still needs no key: that is what keeps CI
        # key-free.
        (RunMode.HYBRID, "mock"),
    ],
)
def test_a_valid_config_passes_without_any_api_key(mode, provider) -> None:
    """Only a hybrid run through the remote provider requires an API key."""
    config = AppConfig()
    config.llm.mode = mode
    config.llm.provider = provider

    assert validate_startup(config, env={}) is config


def test_the_shipped_config_templates_validate() -> None:
    """R26.2/26.3: the deterministic template validates against the real catalog."""
    config = load_config("configs/deterministic.yaml", base_dir="configs")

    assert validate_startup(config, catalog_path=CATALOG_PATH, env={}) is config


def test_hybrid_remote_passes_once_the_key_is_in_the_environment() -> None:
    """The key is looked up by name in the environment, never in configuration."""
    config = AppConfig()
    _hybrid_remote(config)

    assert validate_startup(config, env={API_KEY_ENV: FAKE_KEY}) is config


def test_a_blank_key_counts_as_missing_and_no_value_is_echoed() -> None:
    """A whitespace-only key fails fast, and no secret value appears in the report."""
    config = AppConfig()
    _hybrid_remote(config)
    env = {API_KEY_ENV: "   ", "OTHER_API_KEY": FAKE_KEY}

    with pytest.raises(RuntimeError) as excinfo:
        validate_startup(config, env=env)

    message = str(excinfo.value)
    assert excinfo.value.args[0] is ErrorCode.CONFIG_INVALID
    assert API_KEY_ENV in message
    assert FAKE_KEY not in message


def test_every_problem_is_reported_in_one_error() -> None:
    """R26.4: one explicit error lists all problems, not just the first one found."""
    config = AppConfig()
    config.experiment.top_k = 0
    config.logging.level = "CHATTY"
    _hybrid_remote(config)

    with pytest.raises(RuntimeError) as excinfo:
        validate_startup(config, env={})

    message = str(excinfo.value)
    assert "3 problem(s)" in message
    assert "experiment.top_k" in message
    assert "logging.level" in message
    assert API_KEY_ENV in message


def test_a_missing_catalog_file_is_reported(tmp_path) -> None:
    """A catalog path that does not exist is a startup problem, not a mid-run surprise."""
    missing = tmp_path / "jobs.jsonl"

    with pytest.raises(RuntimeError) as excinfo:
        validate_startup(AppConfig(), catalog_path=str(missing), env={})

    assert excinfo.value.args[0] is ErrorCode.CONFIG_INVALID
    assert "catalog file not found" in str(excinfo.value)


def test_an_unresolved_config_object_is_rejected() -> None:
    """Startup validation insists on a resolved ``AppConfig``."""
    with pytest.raises(RuntimeError) as excinfo:
        validate_startup({"project": {"name": "jobrec"}})  # type: ignore[arg-type]

    assert excinfo.value.args[0] is ErrorCode.CONFIG_INVALID


def test_the_service_builder_validates_config_before_touching_the_database(
    monkeypatch,
) -> None:
    """A config problem is reported as CONFIG_INVALID, never as DB_UNAVAILABLE."""
    checked: list[str] = []

    def _spy(url=None):  # pragma: no cover - must not be reached
        checked.append("db")
        return False

    monkeypatch.delenv("JOBREC_REQUIRE_DB", raising=False)
    monkeypatch.setattr(db_module, "is_database_available", _spy)
    monkeypatch.setattr(app_service_module, "_sql_repo", lambda: None)
    config = AppConfig()
    config.project.environment = "experiment"
    config.experiment.top_k = 0

    with pytest.raises(RuntimeError) as excinfo:
        build_default_service(config, catalog_path=CATALOG_PATH)

    assert excinfo.value.args[0] is ErrorCode.CONFIG_INVALID
    assert checked == []
