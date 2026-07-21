"""PostgreSQL persistence integration test.

Skipped automatically when no database is reachable (DATABASE_URL). Run a local
instance (see scripts/pg_local.sh) or docker-compose to exercise it.
"""

from __future__ import annotations

import pytest

from jobrec.config import load_config
from jobrec.storage.db import is_database_available

pytestmark = pytest.mark.postgres

if not is_database_available():
    pytest.skip("no PostgreSQL reachable; set DATABASE_URL to run", allow_module_level=True)


def test_persist_and_reload_run():
    from jobrec.app_service import AppService
    from jobrec.storage.db import create_all, make_engine, make_session_factory
    from jobrec.storage.repositories import SqlRepository

    engine = make_engine()
    create_all(engine)
    repo = SqlRepository(make_session_factory(engine))
    cfg = load_config("configs/experiment_full.yaml", base_dir="configs")
    svc = AppService(cfg, "data/processed/jobs.jsonl", repository=repo)

    svc.create_candidate({"candidate_id": "pgtest", "skills": ["Python", "SQL"],
                          "years_experience": 1, "target_roles": ["Data Analyst"],
                          "preferred_locations": ["Kuala Lumpur"]})
    sid = svc.create_session("pgtest", "full")
    res = svc.process_turn(sid, "data analyst in Kuala Lumpur, hybrid ok, at least RM4000")

    fresh = SqlRepository(make_session_factory(engine))
    run = fresh.get_run(res.run_record.run_id, include_handoffs=True)
    assert run is not None
    assert run["run_record"]["success"]
    assert fresh.get_candidate_state("pgtest").version >= 1
