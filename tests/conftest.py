"""Shared pytest fixtures."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from jobrec.app_service import AppService
from jobrec.catalog import catalog_hash, load_catalog
from jobrec.config import load_config
from jobrec.domain.job import ActiveSearchState
from jobrec.evidence_store import EvidenceStore

CATALOG_PATH = "data/processed/jobs.jsonl"


@pytest.fixture(scope="session")
def jobs():
    return load_catalog(CATALOG_PATH)


@pytest.fixture(scope="session")
def cat_hash(jobs):
    return catalog_hash(jobs)


@pytest.fixture()
def config():
    return load_config("configs/experiment_full.yaml", base_dir="configs")


@pytest.fixture()
def store():
    return EvidenceStore()


@pytest.fixture()
def service(config):
    return AppService(config, CATALOG_PATH)


def make_active(**overrides) -> ActiveSearchState:
    base = dict(
        active_search_id="as-test", session_id="s", candidate_id="c",
        candidate_state_version=1, dialogue_state_version=1,
        target_roles=["data analyst"], skills_have=["python", "sql"],
        preferred_locations=["Kuala Lumpur"], salary_min=4000.0, salary_currency="MYR",
        work_modes=["hybrid"], experience_level="junior", years_experience=1.0,
        employment_types=[], work_authorizations=[], exclusions={},
        hard_constraint_fields=["salary_min", "preferred_locations"],
        soft_preference_fields=["work_modes"], unknown_fields=[],
        clarification_required_fields=[], field_evidence_map={},
        generated_at=datetime.now(UTC),
    )
    base.update(overrides)
    return ActiveSearchState(**base)


@pytest.fixture()
def active():
    return make_active()
