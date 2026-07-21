"""End-to-end API tests using FastAPI TestClient (deterministic, in-memory)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from jobrec.api.app import create_app
from jobrec.app_service import AppService
from jobrec.config import load_config


@pytest.fixture()
def client():
    cfg = load_config("configs/experiment_full.yaml", base_dir="configs")
    svc = AppService(cfg, "data/processed/jobs.jsonl")
    return TestClient(create_app(svc))


def test_health(client):
    assert client.get("/health/live").json()["status"] == "live"
    assert client.get("/health/ready").json()["status"] == "ready"


def test_full_turn_flow(client):
    r = client.post("/v1/candidates", json={
        "candidate_id": "cand-001", "skills": ["Python", "SQL"], "years_experience": 1,
        "preferred_locations": ["Kuala Lumpur"], "salary_min": 4000, "salary_currency": "MYR",
        "work_modes": ["hybrid"]})
    assert r.status_code == 200
    assert len(r.json()["evidence_ids"]) >= 3

    sid = client.post("/v1/sessions", json={"candidate_id": "cand-001",
                                            "experiment_variant": "full"}).json()["session_id"]
    turn = client.post(f"/v1/sessions/{sid}/turns", json={
        "text": "I want a data analyst role in Kuala Lumpur, hybrid is fine, at least RM4000."}).json()
    assert turn["response_type"] == "recommendation"
    assert turn["trace_summary"]["returned"] > 0
    assert turn["recommendations"][0]["features"]  # per-feature breakdown present

    run = client.get(f"/v1/runs/{turn['run_id']}", params={"include_handoffs": True}).json()
    assert "run_record" in run and "handoffs" in run


def test_unknown_session_404(client):
    r = client.post("/v1/sessions/does-not-exist/turns", json={"text": "hi"})
    assert r.status_code == 404
