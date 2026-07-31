"""Application service shared by the API and CLI.

CLI and API must call the same service, never duplicate business logic. The
service loads the catalog once, manages per-session orchestrators (each with a
persistent evidence store so cross-turn claims resolve), and persists run
bundles through a repository (PostgreSQL or in-memory).
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from datetime import date
from pathlib import Path

from .agents.memory_agent import MemoryAgent
from .catalog import catalog_hash, load_catalog
from .config import AppConfig
from .domain.candidate import CandidateState
from .domain.dialogue import DialogueState
from .domain.enums import ErrorCode, ExperimentVariant, RunMode
from .evidence_store import EvidenceStore
from .orchestration.orchestrator import ConversationOrchestrator, TurnResult, make_provider
from .storage.repositories import InMemoryRepository, Repository
from .utils.hashing import random_id


class AppService:
    """High-level operations: candidates, sessions, turns, run retrieval."""

    def __init__(
        self,
        config: AppConfig,
        catalog_path: str,
        repository: Repository | None = None,
        replay_dir: str | None = None,
    ) -> None:
        self.config = config
        self.jobs = load_catalog(catalog_path)
        self.catalog_hash = catalog_hash(self.jobs)
        self.catalog_snapshot_id = self._snapshot_id(catalog_path)
        self.repo: Repository = repository or InMemoryRepository()
        # Repositories redact run-detail output per config (R12.3); inject the
        # resolved config when the repository was built without one.
        if getattr(self.repo, "config", None) is None:
            try:
                self.repo.config = config  # type: ignore[attr-defined]
            except AttributeError:  # a custom repository may not accept one
                pass
        self.replay_dir = replay_dir
        # per-session orchestrator + evidence store cache
        self._sessions: dict[str, tuple[ConversationOrchestrator, EvidenceStore]] = {}

    def _snapshot_id(self, catalog_path: str) -> str:
        manifest = Path(catalog_path).parent / "catalog_manifest.json"
        if manifest.exists():
            return json.loads(manifest.read_text()).get("catalog_snapshot_id", "catalog-unknown")
        return "catalog-unknown"

    # ------------------------------------------------------------ candidates
    def create_candidate(self, profile: dict) -> CandidateState:
        store = EvidenceStore()
        state = MemoryAgent(store, self.config).create_candidate_state(profile)
        self.repo.upsert_candidate_state(state)
        return state

    def get_candidate(self, candidate_id: str, version: int | None = None) -> CandidateState | None:
        return self.repo.get_candidate_state(candidate_id, version)

    # -------------------------------------------------------------- sessions
    def create_session(self, candidate_id: str, variant: str = "full",
                       session_id: str | None = None) -> str:
        """Open a session. Random by default; ``session_id`` supplies a chosen one.

        The default stays RANDOM because that is correct for real interaction: two live
        conversations must never share an id, and a caller cannot be trusted to have made one
        unique. Only :class:`~jobrec.evaluation.experiment_runner.ExperimentRunner` passes an
        id, because an experiment is the opposite case -- a frozen input set replayed on
        purpose, where a random id makes every ``run_id`` differ between two otherwise
        identical batches and so makes their bundles undiffable.
        """
        session_id = session_id or random_id("sess")
        self.repo.create_session(session_id, candidate_id, variant)
        return session_id

    def _config_for_variant(self, variant: str) -> AppConfig:
        cfg = self.config.model_copy(deep=True)
        cfg.experiment.variant = ExperimentVariant(variant)
        return cfg

    def _orchestrator_for(self, session_id: str, variant: str):
        if session_id in self._sessions:
            return self._sessions[session_id]
        cfg = self._config_for_variant(variant)
        store = EvidenceStore()
        replay_path = f"{self.replay_dir}/model_calls.jsonl" if self.replay_dir else None
        provider = make_provider(cfg, replay_path=replay_path)
        orch = ConversationOrchestrator(
            cfg, self.jobs, self.catalog_snapshot_id, self.catalog_hash,
            provider=provider, store=store,
        )
        self._sessions[session_id] = (orch, store)
        return orch, store

    # ----------------------------------------------------------------- turns
    def process_turn(self, session_id: str, text: str, scenario_id: str | None = None) -> TurnResult:
        session = self.repo.get_session(session_id)
        if session is None:
            raise KeyError(f"unknown session {session_id}")
        candidate_state = self.repo.get_candidate_state(session["candidate_id"])
        if candidate_state is None:
            raise KeyError(f"unknown candidate {session['candidate_id']}")
        dialogue_state = self.repo.get_latest_dialogue_state(session_id) or DialogueState(
            session_id=session_id, candidate_id=candidate_state.candidate_id, version=1, turns=[],
        )
        orch, store = self._orchestrator_for(session_id, session["experiment_variant"])
        # Ensure profile evidence resolves in this session's store (idempotent).
        MemoryAgent(store, self.config).register_profile_evidence(candidate_state)
        # Record db/migration versions on the run when backed by a SQL repository.
        # In-memory runs (no ``versions`` capability) leave these as None.
        versions = self.repo.versions() if hasattr(self.repo, "versions") else None
        result = orch.process_turn(
            candidate_state, dialogue_state, text, scenario_id=scenario_id, versions=versions,
        )
        self.repo.save_turn(result, store.all(), result.model_calls)
        return result

    def get_run(self, run_id: str, **flags: bool) -> dict | None:
        return self.repo.get_run(run_id, **flags)

    def retriever_ready(self) -> bool:
        """Readiness probe: catalog is loaded and a retriever can be built."""
        try:
            from .retrieval.hybrid import make_retriever

            make_retriever(self.jobs, self.config)
            return len(self.jobs) > 0
        except Exception:
            return False


#: Accepted ``logging.level`` names (mirrors the stdlib level names).
_LOG_LEVELS = frozenset(
    {"CRITICAL", "FATAL", "ERROR", "WARN", "WARNING", "INFO", "DEBUG", "NOTSET"}
)


def _api_env_problems(config: AppConfig, env: Mapping[str, str]) -> list[str]:
    """Report missing API environment for a run mode that needs a live model (R26.4).

    Only a ``hybrid`` run through the ``remote`` provider talks to a model API, so
    only that combination requires a key. Deterministic and replay runs need none,
    which is what keeps CI key-free. The key is looked up by NAME
    (:data:`~jobrec.llm.remote_provider.API_KEY_ENV`); its value is never echoed.
    """
    from .llm.remote_provider import API_KEY_ENV, BASE_URL_ENV, MODEL_ENV

    if config.llm.mode != RunMode.HYBRID or config.llm.provider != "remote":
        return []
    if (env.get(API_KEY_ENV) or "").strip():
        return []
    return [
        f"llm.mode=hybrid with llm.provider=remote requires the API key in the "
        f"environment variable {API_KEY_ENV} (optionally {BASE_URL_ENV} and "
        f"{MODEL_ENV}); keys are read from the environment only and are never "
        f"read from configuration files"
    ]


def _startup_problems(
    config: AppConfig, catalog_path: str | None, env: Mapping[str, str]
) -> list[str]:
    """Collect every startup problem so one error reports all of them at once."""
    problems: list[str] = []

    try:
        date.fromisoformat(config.project.reference_date)
    except (TypeError, ValueError):
        problems.append(
            f"project.reference_date must be an ISO date (YYYY-MM-DD), got "
            f"{config.project.reference_date!r}"
        )

    exp = config.experiment
    if exp.top_k < 1:
        problems.append(f"experiment.top_k must be >= 1, got {exp.top_k}")
    if exp.retrieval_pool_size < exp.top_k:
        problems.append(
            f"experiment.retrieval_pool_size ({exp.retrieval_pool_size}) must be >= "
            f"experiment.top_k ({exp.top_k})"
        )
    if exp.repeat_count < 1:
        problems.append(f"experiment.repeat_count must be >= 1, got {exp.repeat_count}")
    if exp.max_dialogue_turns < 1:
        problems.append(
            f"experiment.max_dialogue_turns must be >= 1, got {exp.max_dialogue_turns}"
        )

    threshold = config.memory.clarification_confidence_threshold
    if not 0.0 <= threshold <= 1.0:
        problems.append(
            f"memory.clarification_confidence_threshold must be within [0, 1], got {threshold}"
        )

    weights = config.ranking.weights
    if not weights:
        problems.append("ranking.weights must define at least one feature weight")
    else:
        negative = sorted(name for name, value in weights.items() if value < 0)
        if negative:
            problems.append(f"ranking.weights must be non-negative; negative: {negative}")
        elif sum(weights.values()) <= 0:
            problems.append("ranking.weights must sum to a positive total")
    if config.ranking.salary_scale <= 0:
        problems.append(
            f"ranking.salary_scale must be > 0, got {config.ranking.salary_scale}"
        )

    retrieval_total = (
        config.retrieval.lexical_weight
        + config.retrieval.semantic_weight
        + config.retrieval.structured_weight
    )
    if retrieval_total <= 0:
        problems.append("retrieval weights must sum to a positive total")

    if config.llm.timeout_seconds <= 0:
        problems.append(
            f"llm.timeout_seconds must be > 0, got {config.llm.timeout_seconds}"
        )
    if config.llm.max_retries < 0:
        problems.append(f"llm.max_retries must be >= 0, got {config.llm.max_retries}")

    level = str(config.logging.level or "").upper()
    if level not in _LOG_LEVELS:
        problems.append(
            f"logging.level must be one of {sorted(_LOG_LEVELS)}, got {config.logging.level!r}"
        )

    if catalog_path is not None and not Path(catalog_path).exists():
        problems.append(f"catalog file not found: {catalog_path}")

    problems.extend(_api_env_problems(config, env))
    return problems


def validate_startup(
    config: AppConfig,
    *,
    catalog_path: str | None = None,
    env: Mapping[str, str] | None = None,
) -> AppConfig:
    """Validate that required configuration and API environment are present (R26.3).

    Called at startup by the CLI and by :func:`build_default_service`, so both
    entry points refuse to run on a config that cannot produce a valid experiment.
    Every problem found is reported in a single explicit
    ``RuntimeError(ErrorCode.CONFIG_INVALID, ...)`` (R26.4) rather than surfacing
    later as an obscure failure mid-run; no value of a secret env var appears in
    the message.

    This complements — and never duplicates — the database guard in
    :func:`build_default_service`: reachability of PostgreSQL stays that guard's
    responsibility and keeps reporting ``ErrorCode.DB_UNAVAILABLE``.

    Returns ``config`` so callers can validate inline.
    """
    if not isinstance(config, AppConfig):
        raise RuntimeError(
            ErrorCode.CONFIG_INVALID,
            f"startup validation requires a resolved AppConfig, got {type(config).__name__}",
        )
    problems = _startup_problems(config, catalog_path, os.environ if env is None else env)
    if problems:
        detail = "; ".join(problems)
        raise RuntimeError(
            ErrorCode.CONFIG_INVALID,
            f"invalid startup configuration ({len(problems)} problem(s)): {detail}",
        )
    return config


def _resolve_require_db(config: AppConfig) -> bool:
    """Resolve whether a live database is REQUIRED for this run.

    The DB is required when ``JOBREC_REQUIRE_DB=1`` is set in the environment, or
    when the resolved project environment is an experiment/production run. In
    those modes there is NO silent in-memory fallback: an unreachable database is
    a fatal, explicit error. Deterministic unit tests keep using in-memory/SQLite
    by passing ``require_db=False`` explicitly.
    """
    if os.environ.get("JOBREC_REQUIRE_DB") == "1":
        return True
    return config.project.environment in {"experiment", "production"}


def build_default_service(
    config: AppConfig,
    catalog_path: str = "data/processed/jobs.jsonl",
    use_database: bool | None = None,
    require_db: bool | None = None,
) -> AppService:
    """Build an AppService, using PostgreSQL when available (or requested).

    When the database is REQUIRED (``require_db`` resolves to True via
    ``JOBREC_REQUIRE_DB=1`` or ``config.project.environment`` in
    ``{experiment, production}``) but no database is reachable, this fails fast
    with ``RuntimeError(ErrorCode.DB_UNAVAILABLE, ...)`` instead of silently
    degrading to an in-memory repository. Deterministic unit tests should pass
    ``require_db=False`` to keep using the in-memory/SQLite path.

    Configuration and API environment are validated first (:func:`validate_startup`,
    R26.3/26.4); database reachability is checked after, so a config problem is
    reported as ``CONFIG_INVALID`` and an unreachable database as ``DB_UNAVAILABLE``.
    """
    from .storage.db import is_database_available

    validate_startup(config, catalog_path=catalog_path)

    if require_db is None:
        require_db = _resolve_require_db(config)

    available = is_database_available()

    if require_db and not available:
        raise RuntimeError(
            ErrorCode.DB_UNAVAILABLE,
            "experiment mode requires a reachable PostgreSQL database via "
            "DATABASE_URL; no in-memory fallback is permitted when the database "
            "is required",
        )

    repo: Repository
    if require_db:
        want_db = True
    elif use_database is not None:
        want_db = use_database
    else:
        want_db = available
    if want_db:
        repo = _sql_repo()
    else:
        repo = InMemoryRepository()
    return AppService(config, catalog_path, repository=repo)


def _sql_repo():
    from .storage.db import create_all, make_engine, make_session_factory
    from .storage.repositories import SqlRepository

    engine = make_engine()
    create_all(engine)
    return SqlRepository(make_session_factory(engine))
