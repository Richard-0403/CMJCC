"""Application service shared by the API and CLI.

CLI and API must call the same service, never duplicate business logic. The
service loads the catalog once, manages per-session orchestrators (each with a
persistent evidence store so cross-turn claims resolve), and persists run
bundles through a repository (PostgreSQL or in-memory).
"""

from __future__ import annotations

import json
from pathlib import Path

from .catalog import catalog_hash, load_catalog
from .config import AppConfig
from .domain.candidate import CandidateState
from .domain.dialogue import DialogueState
from .domain.enums import ExperimentVariant
from .evidence_store import EvidenceStore
from .orchestration.orchestrator import ConversationOrchestrator, TurnResult, make_provider
from .agents.memory_agent import MemoryAgent
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
    def create_session(self, candidate_id: str, variant: str = "full") -> str:
        session_id = random_id("sess")
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
        result = orch.process_turn(candidate_state, dialogue_state, text, scenario_id=scenario_id)
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


def build_default_service(
    config: AppConfig,
    catalog_path: str = "data/processed/jobs.jsonl",
    use_database: bool | None = None,
) -> AppService:
    """Build an AppService, using PostgreSQL when available (or requested)."""
    from .storage.db import is_database_available, make_session_factory

    repo: Repository
    want_db = use_database if use_database is not None else is_database_available()
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
