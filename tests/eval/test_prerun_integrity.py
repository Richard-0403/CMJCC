"""The last round of pre-run gaps, each one an independent way to spend a batch and get
an artifact that cannot answer the question it was built for.

1. The oracle input fingerprint hashed RAW BYTES, so it depended on the CHECKOUT rather than
   the content: the same scenario file arrives with CRLF on a Windows clone and LF elsewhere,
   and a reference frozen on one platform was reported STALE on the other.

2. The frozen oracle was loaded in the ANALYSIS stage, after every run had executed. A stale
   reference is refused, so the refusal arrived with the whole batch already spent.

3. ``response_model`` -- the model the SERVER says answered -- was captured by the provider
   and dropped by the exporter's whitelist, so no bundle could evidence which model produced
   its runs. ``request_params.model`` records only what was ASKED for, and an alias or a
   gateway can make those differ.

4. ``no_match_cause`` states "N of the M jobs in scope" and only N was checked. M could be
   999 against a three-job scope and the verdict stayed ``supported``.

5. An EMPTY blocking diagnosis was treated like a MISSING one, so a diagnosis that ran and
   found no blocking field -- a positive statement -- caused every hard constraint to be
   offered as a reason it had just ruled out.

6. The raw ``base_url`` reached ``RunRecord.model_manifest`` and therefore every run bundle. A
   base URL is the one input that can embed a credential.

Claim/message consistency is asserted in ``test_claims_match_the_reply.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobrec.config import load_config
from jobrec.domain.enums import ConfirmationStatus, EvidenceSource, PersistenceScope
from jobrec.domain.recommendation import ResponseClaim
from jobrec.evidence_store import EvidenceStore

CATALOG = "data/processed/jobs.jsonl"
CONFIG = "configs/experiment_full.yaml"
SCENARIOS = "evaluation/data/scenarios.jsonl"


# ------------------------------------------------------ 1. line endings
def test_crlf_and_lf_scenario_files_share_one_input_fingerprint(tmp_path: Path):
    """Line endings are not part of what a scenario says."""
    from jobrec_eval.oracle_reference import inputs_fingerprint

    lf = tmp_path / "lf.jsonl"
    crlf = tmp_path / "crlf.jsonl"
    body = Path(SCENARIOS).read_bytes().replace(b"\r\n", b"\n")
    lf.write_bytes(body)
    crlf.write_bytes(body.replace(b"\n", b"\r\n"))

    assert lf.read_bytes() != crlf.read_bytes(), "the fixture did not actually differ"
    assert inputs_fingerprint(lf, CATALOG) == inputs_fingerprint(crlf, CATALOG)


def test_a_real_content_edit_still_moves_the_fingerprint(tmp_path: Path):
    """The normalisation is narrow: only CR/CRLF collapse, nothing else is forgiven."""
    from jobrec_eval.oracle_reference import inputs_fingerprint

    base = tmp_path / "a.jsonl"
    edited = tmp_path / "b.jsonl"
    body = Path(SCENARIOS).read_text(encoding="utf-8")
    base.write_text(body, encoding="utf-8", newline="\n")
    edited.write_text(body.replace("SC-A-01", "SC-A-99", 1), encoding="utf-8", newline="\n")

    assert inputs_fingerprint(base, CATALOG) != inputs_fingerprint(edited, CATALOG)


def test_the_live_frozen_oracle_is_fresh():
    """The gate the whole batch depends on, asserted on the real artifact."""
    from jobrec_eval.oracle_reference import (
        frozen_artifact_path,
        inputs_fingerprint,
        load_frozen_references,
    )

    frozen = load_frozen_references(frozen_artifact_path(SCENARIOS))
    assert frozen is not None, "no frozen canonical oracle on disk"
    assert frozen.inputs_fingerprint == inputs_fingerprint(SCENARIOS, CATALOG)


# --------------------------------------------- 2. the gate runs before the runs
def test_a_stale_oracle_stops_the_pipeline_before_any_run(tmp_path: Path, monkeypatch):
    """The refusal must arrive before the first request, not after the last one."""
    from jobrec_eval import cli
    from jobrec_eval.oracle_reference import StaleCanonicalOracleError

    started = []
    monkeypatch.setattr(cli.ExperimentRunner, "run",
                        lambda self, *a, **k: started.append(True) or {})
    monkeypatch.setattr(
        cli, "load_or_build_canonical_references",
        lambda *a, **k: (_ for _ in ()).throw(StaleCanonicalOracleError("stale")))

    with pytest.raises(StaleCanonicalOracleError):
        cli.run_pipeline(CONFIG, SCENARIOS, CATALOG, str(tmp_path / "out"), 1, None, 10, 1)

    assert not started, "the experiment started despite a stale oracle"


def test_the_freshness_gate_does_not_create_the_artifact(tmp_path: Path):
    """``freeze=False``: a check must not satisfy itself by writing the thing it checks."""
    from jobrec_eval.oracle_reference import frozen_artifact_path

    scenarios = tmp_path / "scenarios.jsonl"
    scenarios.write_text(
        Path(SCENARIOS).read_text(encoding="utf-8"), encoding="utf-8", newline="\n")
    target = frozen_artifact_path(scenarios)
    assert not target.exists()

    from jobrec_eval.oracle_reference import load_or_build_canonical_references

    load_or_build_canonical_references(
        scenarios, CATALOG, load_config(CONFIG, base_dir="configs"), freeze=False)
    assert not target.exists(), "the gate froze an artifact as a side effect"


# ------------------------------------------------- 3. server-reported model
def test_the_exporter_keeps_the_model_the_server_reported():
    from jobrec.evaluation.exporters import _RESPONSE_METADATA_KEYS

    for key in ("response_id", "system_fingerprint", "response_model"):
        assert key in _RESPONSE_METADATA_KEYS, key


def test_response_provenance_survives_the_export(tmp_path: Path):
    """End to end: provider metadata -> exported model_calls row."""
    from jobrec.evaluation.exporters import _model_call_row
    from jobrec.llm.provider import LLMCallRecord

    record = LLMCallRecord(
        call_id="c1", purpose="intent_extraction", prompt="p", raw_response="{}",
        parsed_ok=True, latency_ms=1.0, provider="remote", model="gpt-4o-mini",
        metadata={"response_id": "chatcmpl-1", "system_fingerprint": "fp_x",
                  "response_model": "gpt-4o-mini-2024-07-18", "model": "gpt-4o-mini"},
    )
    row = _model_call_row(record)
    meta = row["response_metadata"]
    assert meta["response_model"] == "gpt-4o-mini-2024-07-18"
    assert meta["response_id"] == "chatcmpl-1"
    assert meta["system_fingerprint"] == "fp_x"
    # What was ASKED for stays separately recorded, so the two can be compared.
    assert row["request_params"]["model"] == "gpt-4o-mini"


# ---------------------------------------------------- 4. the denominator
def _stage(store: EvidenceStore, field: str, removed: int, evaluated: int) -> str:
    return store.register_field(
        EvidenceSource.SYSTEM_RULE, "dec-1", f"filtered_by:{field}",
        {"field": field, "filtered_count": removed, "evaluated_jobs": evaluated},
        confidence=1.0, confirmation=ConfirmationStatus.CONFIRMED,
        scope=PersistenceScope.SESSION).evidence_id


def _candidate(store: EvidenceStore, field: str, value) -> str:
    return store.register_field(
        EvidenceSource.DIALOGUE, "cand-1", field, value, confidence=0.9,
        confirmation=ConfirmationStatus.CONFIRMED,
        scope=PersistenceScope.SESSION).evidence_id


@pytest.mark.parametrize(("evaluated", "expected"),
                         [(3, "supported"), (999, "unsupported"), (1, "unsupported")])
def test_the_evaluated_count_is_checked_against_the_stage_record(evaluated, expected):
    """"N of the M jobs in scope": an unchecked M let a true numerator carry a false total."""
    from jobrec.agents.explanation_agent import semantic_status

    store = EvidenceStore()
    claim = ResponseClaim(
        claim_id="c1", claim_type="no_match_cause", text="rendered elsewhere",
        predicate="no_match_cause", field_name="salary_min",
        claim_args={"removed": 2, "evaluated_jobs": evaluated},
        evidence_ids=[_candidate(store, "salary_min", 4000),
                      _stage(store, "salary_min", 2, 3)])
    assert semantic_status(claim, store) == expected


def test_a_record_removing_more_than_it_evaluated_supports_nothing():
    """Arithmetically impossible, so the record is unusable rather than authoritative."""
    from jobrec.agents.explanation_agent import semantic_status

    store = EvidenceStore()
    claim = ResponseClaim(
        claim_id="c1", claim_type="no_match_cause", text="t",
        predicate="no_match_cause", field_name="salary_min",
        claim_args={"removed": 9, "evaluated_jobs": 3},
        evidence_ids=[_candidate(store, "salary_min", 4000),
                      _stage(store, "salary_min", 9, 3)])
    assert semantic_status(claim, store) == "unsupported"


# -------------------------------------- 5. empty diagnosis is not a missing one
def _no_match_decision(diagnosis: dict):
    from jobrec.domain.recommendation import RecommendationDecision
    from jobrec.utils.time import utcnow

    return RecommendationDecision(
        decision_id="dec-1", session_id="s", active_search_id="a", context_id=None,
        experiment_variant="full", no_match=True, no_match_reason_codes=[],
        created_at=utcnow(), scorer_version="t", config_hash="c",
        no_match_diagnosis=diagnosis)


class _Active:
    candidate_id = "cand-1"
    hard_constraint_fields = ["salary_min", "work_modes"]
    salary_min = 4000
    work_modes = ["onsite"]
    field_evidence_map = {"salary_min": ["ev-a"], "work_modes": ["ev-b"]}


def test_an_empty_blocking_diagnosis_invents_no_reasons():
    """A diagnosis that ran and found nothing is a POSITIVE statement, not missing data."""
    from jobrec.agents.explanation_agent import ExplanationAgent

    agent = ExplanationAgent(EvidenceStore(), load_config(CONFIG, base_dir="configs"))
    response, dropped = agent._no_match(
        _no_match_decision({"evaluated_jobs": 3, "eligible_jobs": 0,
                            "blocking_constraints": [], "relaxation_candidates": [],
                            "stage_trace": []}),
        _Active())

    assert {c.field_name for c in [*response.claims, *dropped]} == set(), (
        "reasons were manufactured for fields the diagnosis had ruled out")


def test_a_missing_diagnosis_still_explains_itself():
    """The complementary case: nothing to narrow by, so a reason is better than silence."""
    from jobrec.agents.explanation_agent import ExplanationAgent

    agent = ExplanationAgent(EvidenceStore(), load_config(CONFIG, base_dir="configs"))
    response, dropped = agent._no_match(_no_match_decision({}), _Active())

    named = {c.field_name for c in [*response.claims, *dropped]}
    assert named == {"salary_min", "work_modes"}, named


# ------------------------------------------- 6. no credential in a bundle
def test_the_model_manifest_publishes_a_cleaned_endpoint(monkeypatch):
    """``RunRecord.model_manifest`` is archived in every bundle, so it must carry no secret."""
    from jobrec.llm.remote_provider import (
        API_KEY_ENV,
        BASE_URL_ENV,
        MODEL_ENV,
        RemoteLLMProvider,
    )

    secret = "sk-live-must-never-be-archived"
    monkeypatch.setenv(BASE_URL_ENV, f"https://svc:{secret}@gw.example.com/deploy-a/v1?k={secret}")
    monkeypatch.setenv(MODEL_ENV, "qwen-plus")
    monkeypatch.setenv(API_KEY_ENV, secret)

    manifest = RemoteLLMProvider().manifest()

    assert secret not in json.dumps(manifest)
    assert manifest["endpoint"] == "https://gw.example.com/deploy-a/v1"
    assert "base_url" not in manifest, "the raw base_url is still published"
    # The key's SOURCE is still recorded, which is what makes the run reproducible.
    assert manifest["api_key_env"] == API_KEY_ENV
    assert manifest["api_key_present"] is True
