"""Unit tests for the experiment-mode database fail-fast guard (R9.6).

R9.6: WHILE running in production (experiment) mode, IF the database is
unavailable, THEN the System SHALL fail fast and SHALL NOT switch silently to
in-memory storage. ``build_default_service`` therefore raises
``RuntimeError(ErrorCode.DB_UNAVAILABLE, ...)`` whenever the database is
REQUIRED (``JOBREC_REQUIRE_DB=1`` or ``config.project.environment`` in
``{experiment, production}``) but unreachable, and never constructs an
``InMemoryRepository`` on that path. Deterministic unit runs opt out with
``require_db=False``.
"""

from __future__ import annotations

import pytest

from jobrec import app_service as app_service_module
from jobrec.app_service import AppService, build_default_service
from jobrec.config import AppConfig
from jobrec.domain.enums import ErrorCode
from jobrec.storage import db as db_module
from jobrec.storage.repositories import InMemoryRepository

CATALOG_PATH = "data/processed/jobs.jsonl"

#: Environments that make a live database mandatory (no silent degradation).
DB_REQUIRED_ENVIRONMENTS = ["experiment", "production"]


def _config(environment: str) -> AppConfig:
    cfg = AppConfig()
    cfg.project.environment = environment
    return cfg


@pytest.fixture()
def repo_spy(monkeypatch):
    """Record every repository the builder constructs, so a silent fallback shows up.

    ``JOBREC_REQUIRE_DB`` is cleared first so each test states its own intent and
    nothing leaks in from the ambient environment.
    """
    built: list[str] = []

    class _SpyInMemoryRepository(InMemoryRepository):
        def __init__(self) -> None:
            built.append("in_memory")
            super().__init__()

    def _spy_sql_repo():
        built.append("sql")
        return InMemoryRepository()  # stand-in: never touched by the fail-fast path

    monkeypatch.delenv("JOBREC_REQUIRE_DB", raising=False)
    monkeypatch.setattr(app_service_module, "InMemoryRepository", _SpyInMemoryRepository)
    monkeypatch.setattr(app_service_module, "_sql_repo", _spy_sql_repo)
    return built


def _set_database_reachable(monkeypatch, reachable: bool) -> None:
    monkeypatch.setattr(db_module, "is_database_available", lambda url=None: reachable)


@pytest.mark.parametrize("environment", DB_REQUIRED_ENVIRONMENTS)
def test_experiment_mode_without_database_raises_db_unavailable(
    monkeypatch, repo_spy, environment
):
    """Experiment/production mode with an unreachable DB fails fast, never falls back."""
    _set_database_reachable(monkeypatch, False)

    with pytest.raises(RuntimeError) as excinfo:
        build_default_service(_config(environment), catalog_path=CATALOG_PATH)

    # The failure is explicit and machine-readable.
    assert excinfo.value.args[0] is ErrorCode.DB_UNAVAILABLE
    assert "DB_UNAVAILABLE" in str(excinfo.value)
    # No repository at all was built: no silent in-memory degradation.
    assert repo_spy == []


def test_require_db_env_var_triggers_fail_fast_in_any_environment(monkeypatch, repo_spy):
    """``JOBREC_REQUIRE_DB=1`` makes even a local run fail fast on an unreachable DB."""
    _set_database_reachable(monkeypatch, False)
    monkeypatch.setenv("JOBREC_REQUIRE_DB", "1")

    with pytest.raises(RuntimeError) as excinfo:
        build_default_service(_config("local"), catalog_path=CATALOG_PATH)

    assert excinfo.value.args[0] is ErrorCode.DB_UNAVAILABLE
    assert repo_spy == []


def test_use_database_false_cannot_override_experiment_mode_fail_fast(monkeypatch, repo_spy):
    """An explicit ``use_database=False`` never buys an in-memory experiment run."""
    _set_database_reachable(monkeypatch, False)

    with pytest.raises(RuntimeError) as excinfo:
        build_default_service(
            _config("experiment"), catalog_path=CATALOG_PATH, use_database=False
        )

    assert excinfo.value.args[0] is ErrorCode.DB_UNAVAILABLE
    assert repo_spy == []


@pytest.mark.parametrize("environment", DB_REQUIRED_ENVIRONMENTS)
def test_experiment_mode_with_reachable_database_uses_sql_repository(
    monkeypatch, repo_spy, environment
):
    """The guard only fires on an unreachable DB: a reachable one takes the SQL path."""
    _set_database_reachable(monkeypatch, True)

    service = build_default_service(_config(environment), catalog_path=CATALOG_PATH)

    assert isinstance(service, AppService)
    assert repo_spy == ["sql"]


def test_local_environment_without_database_builds_in_memory(monkeypatch, repo_spy):
    """A dev/local run may still degrade to in-memory, explicitly and without error."""
    _set_database_reachable(monkeypatch, False)

    service = build_default_service(_config("local"), catalog_path=CATALOG_PATH)

    assert isinstance(service, AppService)
    assert isinstance(service.repo, InMemoryRepository)
    assert repo_spy == ["in_memory"]


def test_require_db_false_opts_deterministic_tests_out_of_fail_fast(monkeypatch, repo_spy):
    """``require_db=False`` keeps deterministic runs on the in-memory path in any mode."""
    _set_database_reachable(monkeypatch, False)
    monkeypatch.setenv("JOBREC_REQUIRE_DB", "1")

    service = build_default_service(
        _config("experiment"), catalog_path=CATALOG_PATH, require_db=False
    )

    assert isinstance(service.repo, InMemoryRepository)
    assert repo_spy == ["in_memory"]
