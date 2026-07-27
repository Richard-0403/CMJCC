"""PostgreSQL persistence and restart-recovery integration tests (R9.1-R9.5, R9.9).

Complements ``test_postgres.py`` (which covers the basic persist-then-reload of a
single run) by exercising *restart* recovery against a real database: every state
record is reloaded through freshly built connections, the same session continues
in a rebuilt service, persisted claims keep resolvable evidence links, and the
long-term ``CandidateState`` version history survives.

"Restart" is simulated the only way it can be inside one test process: the
service and repository are rebuilt from ``DATABASE_URL`` with a new engine,
session factory and (for the service) an empty per-session orchestrator cache, so
nothing is read from in-process memory.

Skipped automatically when no database is reachable (``DATABASE_URL``). Run a
local instance (see ``scripts/pg_local.sh``) or ``make test-postgres``.
"""

from __future__ import annotations

import pytest

from jobrec.config import load_config
from jobrec.storage.db import is_database_available

pytestmark = pytest.mark.postgres

if not is_database_available():
    pytest.skip("no PostgreSQL reachable; set DATABASE_URL to run", allow_module_level=True)

CATALOG_PATH = "data/processed/jobs.jsonl"
CONFIG_PATH = "configs/experiment_full.yaml"
QUERY = "data analyst in Kuala Lumpur, hybrid ok, at least RM4000"

PROFILE = {
    "skills": ["Python", "SQL"],
    "years_experience": 1,
    "target_roles": ["Data Analyst"],
    "preferred_locations": ["Kuala Lumpur"],
    "work_modes": ["remote"],
}


def _restart_service():
    """Build a service as a restarted process would: DB required, no cached state."""
    from jobrec.app_service import build_default_service

    cfg = load_config(CONFIG_PATH, base_dir="configs")
    return build_default_service(cfg, CATALOG_PATH, require_db=True)


@pytest.fixture()
def session_factory():
    """A session factory on a NEW engine, for post-restart reads of raw rows."""
    from jobrec.storage.db import create_all, make_engine, make_session_factory

    engine = make_engine()
    create_all(engine)
    return make_session_factory(engine)


@pytest.fixture()
def restarted_repo(session_factory):
    """A repository built after the "restart" (fresh connections, no caches)."""
    from jobrec.storage.repositories import SqlRepository

    return SqlRepository(session_factory)


def _seed_turn(svc):
    """Create a candidate + session and run one turn. Returns (candidate_id, session_id, result)."""
    from jobrec.utils.hashing import random_id

    candidate_id = random_id("pgcand")
    svc.create_candidate({"candidate_id": candidate_id, **PROFILE})
    session_id = svc.create_session(candidate_id, "full")
    result = svc.process_turn(session_id, QUERY)
    return candidate_id, session_id, result


def test_all_state_records_are_saved_and_restored(restarted_repo):
    """Every persisted state record reloads after a restart (R9.2).

    Covers CandidateState, DialogueState, RecommendationDecision (which carries the
    per-search ActiveSearchState identity), EvidenceLog and Handoff records.
    """
    svc = _restart_service()
    candidate_id, session_id, res = _seed_turn(svc)
    assert res.decision is not None, "expected a recommendation decision for this query"

    # Session + CandidateState + DialogueState.
    session = restarted_repo.get_session(session_id)
    assert session is not None
    assert session["candidate_id"] == candidate_id
    assert session["experiment_variant"] == "full"

    candidate = restarted_repo.get_candidate_state(candidate_id)
    assert candidate is not None
    assert candidate.candidate_id == candidate_id
    assert candidate.version >= 1

    dialogue = restarted_repo.get_latest_dialogue_state(session_id)
    assert dialogue is not None
    assert dialogue.session_id == session_id
    assert dialogue.candidate_id == candidate_id
    assert dialogue.turns

    # Run bundle: run record, response, decision, evidence log, handoffs.
    bundle = restarted_repo.get_run(
        res.run_record.run_id,
        include_states=True,
        include_evidence=True,
        include_handoffs=True,
    )
    assert bundle is not None
    assert bundle["run_record"]["run_id"] == res.run_record.run_id
    assert bundle["run_record"]["session_id"] == session_id
    assert bundle["response"]["session_id"] == session_id
    assert bundle["decision"]["decision_id"] == res.decision.decision_id
    # ActiveSearchState is per-search and re-derived by CMJCC; its identity is
    # persisted on the decision.
    assert bundle["decision"]["active_search_id"] == res.decision.active_search_id
    assert bundle["evidence_log"], "no EvidenceLog records restored"
    assert bundle["handoffs"], "no Handoff records restored"
    assert {h["handoff_id"] for h in bundle["handoffs"]} == {h.handoff_id for h in res.handoffs}


def test_session_continues_after_restart(restarted_repo):
    """A rebuilt service resumes the same session and appends to its history (R9.3)."""
    first_svc = _restart_service()
    candidate_id, session_id, first = _seed_turn(first_svc)

    restarted_svc = _restart_service()  # new engine/repo, empty orchestrator cache
    second = restarted_svc.process_turn(session_id, "remote is fine too")

    assert second.dialogue_state.session_id == session_id
    assert second.run_record.session_id == session_id
    assert second.run_record.candidate_id == candidate_id
    # The restored dialogue was continued, not restarted from scratch.
    assert second.dialogue_state.version > first.dialogue_state.version
    assert len(second.dialogue_state.turns) > len(first.dialogue_state.turns)
    first_turn_ids = [t.turn_id for t in first.dialogue_state.turns]
    assert [t.turn_id for t in second.dialogue_state.turns][: len(first_turn_ids)] == first_turn_ids
    # Both turns are readable as separate runs from the restarted repository.
    assert restarted_repo.get_run(first.run_record.run_id) is not None
    assert restarted_repo.get_run(second.run_record.run_id) is not None
    latest = restarted_repo.get_latest_dialogue_state(session_id)
    assert latest is not None
    assert latest.version == second.dialogue_state.version


def test_evidence_links_remain_valid_after_restart(restarted_repo, session_factory):
    """Restored claims still resolve to persisted evidence items (R9.4)."""
    from jobrec.storage.models import EvidenceItemRow

    svc = _restart_service()
    _candidate_id, _session_id, res = _seed_turn(svc)

    bundle = restarted_repo.get_run(res.run_record.run_id, include_evidence=True)
    assert bundle is not None
    supported = [c for c in bundle["response"]["claims"] if c["support_status"] == "supported"]
    assert supported, "expected at least one supported claim to verify evidence links"

    with session_factory() as session:
        for claim in supported:
            assert claim["evidence_ids"], f"supported claim {claim['claim_id']} has no evidence id"
            for evidence_id in claim["evidence_ids"]:
                row = session.get(EvidenceItemRow, evidence_id)
                assert row is not None, f"dangling evidence id {evidence_id} after restart"
                assert row.evidence_id == evidence_id


def test_candidate_version_history_is_preserved_after_restart(restarted_repo):
    """Every long-term CandidateState version stays retrievable after a restart (R9.5)."""
    from jobrec.agents.memory_agent import MemoryAgent
    from jobrec.domain.enums import ConfirmationStatus, ConstraintStrength, PersistenceScope
    from jobrec.domain.extraction import ExtractedPreference, ExtractedPreferenceSet
    from jobrec.evidence_store import EvidenceStore
    from jobrec.utils.hashing import random_id

    svc = _restart_service()
    candidate_id = random_id("pgcand")
    v1 = svc.create_candidate({"candidate_id": candidate_id, **PROFILE})
    assert v1.version == 1

    # A durable "from now on prefer hybrid" write-back produces version 2.
    durable = ExtractedPreference(
        field_name="work_modes",
        normalized_value="hybrid",
        raw_text="from now on I prefer hybrid",
        confidence=0.95,
        confirmation_status=ConfirmationStatus.CONFIRMED,
        persistence_scope=PersistenceScope.LONG_TERM,
        proposed_strength=ConstraintStrength.SOFT,
        temporal_scope="long_term",
    )
    v2 = MemoryAgent(EvidenceStore(), svc.config).apply_confirmed_updates(
        v1, ExtractedPreferenceSet(utterance_id="u1", preferences=[durable]), conflicts=[]
    )
    assert v2.version == 2
    svc.repo.upsert_candidate_state(v2)

    latest = restarted_repo.get_candidate_state(candidate_id)
    assert latest is not None
    assert latest.version == 2
    assert "hybrid" in {p.value for p in latest.work_modes if p.is_active}

    # The superseded version is still addressable, so history is not overwritten.
    original = restarted_repo.get_candidate_state(candidate_id, version=1)
    assert original is not None
    assert original.version == 1
    assert {p.value for p in original.work_modes} == {"remote"}
