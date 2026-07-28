"""Tests for the persisted evidence items artifact (``evidence_items.jsonl``).

A claim in ``response_claims.json`` carries only ``evidence_ids``. Without the
evidence ITEMS those ids are opaque, so neither a human annotator nor an offline
audit can check what a claim actually cites. These tests drive a real deterministic
turn through ``AppService`` + ``write_run_bundle`` (no hand-built evidence) and
check that:

- every bundle carries ``evidence_items.jsonl`` with the fields that matter
  (source, source object, field name, raw text, normalized value, scope),
- a claim's ``evidence_ids`` resolve through ``RunBundle.resolve_claim_evidence``
  to items with the expected ``field_name`` / ``normalized_value``,
- a dangling id is REPORTED as unresolvable rather than silently dropped (R10.1),
- a bundle written before the artifact existed still loads (backward compatible),
- adding the artifact moves no reproducibility hash: ``run_record.json`` and the
  ``hashes`` block of ``run_manifest.json`` are identical with and without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobrec.app_service import AppService
from jobrec.config import load_config
from jobrec.evaluation.checksums import compute_checksums
from jobrec.evaluation.exporters import write_run_bundle
from jobrec_eval.loaders import RunBundle, load_bundles

CATALOG_PATH = "data/processed/jobs.jsonl"
UTTERANCE = "I want a data analyst role in Kuala Lumpur, hybrid, at least RM4000."
SCENARIO_ID = "SC-EVID"

#: Fields of an EvidenceItem that make an id resolvable to "field X of Y = Z".
_REQUIRED_FIELDS = (
    "evidence_id",
    "source",
    "source_object_id",
    "field_name",
    "normalized_value",
    "confidence",
    "confirmation_status",
    "persistence_scope",
    "observed_at",
)


@pytest.fixture(scope="module")
def turn():
    """One real deterministic turn (result + resolved config)."""
    cfg = load_config("configs/experiment_full.yaml", base_dir="configs")
    svc = AppService(cfg, CATALOG_PATH)
    svc.create_candidate({
        "candidate_id": "c-evidence", "skills": ["Python", "SQL"],
        "years_experience": 3, "preferred_locations": ["Kuala Lumpur"],
        "work_modes": ["hybrid"],
    })
    session_id = svc.create_session("c-evidence", "full")
    return svc.process_turn(session_id, UTTERANCE, scenario_id=SCENARIO_ID), cfg


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _bundle_from_dir(run_dir: Path) -> RunBundle:
    """A RunBundle backed by the artifacts of one written run directory."""
    bundle = RunBundle(
        variant="full", scenario_id=SCENARIO_ID, run_index=0, path=run_dir,
        run_record=json.loads((run_dir / "run_record.json").read_text()),
        decision=json.loads((run_dir / "recommendation_decision.json").read_text()),
        response=json.loads((run_dir / "response.json").read_text()),
        claims=json.loads((run_dir / "response_claims.json").read_text()),
        handoffs=[], evidence_log=[], latency={}, active_search=None, job_context=None,
    )
    bundle.evidence_items = _read_jsonl(run_dir / "evidence_items.jsonl")
    return bundle


def test_evidence_items_are_written_for_a_real_turn(turn, tmp_path):
    """Every bundle carries the evidence items behind its claims."""
    result, cfg = turn
    out = write_run_bundle(result, tmp_path / "run", cfg)

    items = _read_jsonl(out / "evidence_items.jsonl")
    assert items, "a real turn registered no evidence"
    assert len(items) == len(result.evidence_items)
    for item in items:
        for name in _REQUIRED_FIELDS:
            assert name in item, f"evidence item is missing {name}"
        assert item["field_name"]
        assert item["source_object_id"]

    ids = [item["evidence_id"] for item in items]
    assert len(ids) == len(set(ids)), "evidence ids are not unique in the dump"

    # The items artifact is NOT the per-stage decision log; both are written.
    log = _read_jsonl(out / "evidence_log.jsonl")
    assert "stage" in log[0] and "evidence_id" not in log[0]


def test_claim_evidence_ids_resolve_to_the_cited_field_and_value(turn, tmp_path):
    """A claim's ids resolve to items carrying the field/value the claim rests on."""
    result, cfg = turn
    out = write_run_bundle(result, tmp_path / "run", cfg)
    bundle = _bundle_from_dir(out)

    grounded = [c for c in bundle.claims if c.get("evidence_ids")]
    assert grounded, "no grounded claims to resolve"

    # Ground truth straight from the session store the run used.
    expected = {
        item.evidence_id: (item.field_name, item.normalized_value)
        for item in result.evidence_items
    }

    for claim in grounded:
        resolved = bundle.resolve_claim_evidence(claim)
        assert resolved.claim_id == claim["claim_id"]
        assert resolved.fully_resolved, f"dangling ids {resolved.unresolved_ids}"
        assert resolved.cited_count == len(claim["evidence_ids"])
        assert [i["evidence_id"] for i in resolved.items] == list(claim["evidence_ids"])
        for item in resolved.items:
            field_name, value = expected[item["evidence_id"]]
            assert item["field_name"] == field_name
            if isinstance(value, float):
                assert item["normalized_value"] == pytest.approx(value)
            else:
                assert item["normalized_value"] == value

    index = bundle.evidence_index
    assert set(index) == {item["evidence_id"] for item in bundle.evidence_items}
    assert all(index[i]["evidence_id"] == i for i in index)


def test_dangling_evidence_id_is_reported_not_dropped(turn, tmp_path):
    """An id with no evidence item is surfaced as unresolvable (R10.1)."""
    result, cfg = turn
    out = write_run_bundle(result, tmp_path / "run", cfg)
    bundle = _bundle_from_dir(out)

    real_id = bundle.evidence_items[0]["evidence_id"]
    claim = {"claim_id": "cl-dangling", "evidence_ids": [real_id, "ev-does-not-exist"]}

    resolved = bundle.resolve_claim_evidence(claim)

    assert resolved.fully_resolved is False
    assert resolved.unresolved_ids == ("ev-does-not-exist",)
    assert [i["evidence_id"] for i in resolved.items] == [real_id]
    # The cited count still accounts for the dangling id: nothing is dropped.
    assert resolved.cited_count == 2

    # A bundle with no evidence items at all resolves nothing, rather than
    # reporting a claim as supported by evidence it cannot show.
    empty = _bundle_from_dir(out)
    empty.evidence_items = []
    assert empty.evidence_index == {}
    assert empty.resolve_claim_evidence(claim).unresolved_ids == (real_id, "ev-does-not-exist")


def test_load_bundles_reads_the_artifact_and_older_bundles_still_load(turn, tmp_path):
    """``load_bundles`` picks the artifact up; a bundle without it loads empty."""
    result, cfg = turn
    exp = tmp_path / "exp-evidence"
    write_run_bundle(result, exp / "full" / SCENARIO_ID / "0", cfg)
    legacy = exp / "full" / SCENARIO_ID / "1"
    write_run_bundle(result, legacy, cfg)
    (legacy / "evidence_items.jsonl").unlink()

    bundles = {b.run_index: b for b in load_bundles(exp)}
    assert set(bundles) == {0, 1}

    current = bundles[0]
    assert current.evidence_items
    assert len(current.evidence_items) == len(result.evidence_items)
    assert current.resolve_all_claim_evidence()
    assert all(r.fully_resolved for r in current.resolve_all_claim_evidence())

    older = bundles[1]
    assert older.evidence_items == []
    assert older.evidence_index == {}
    # Its claims still load; their citations are simply reported unresolvable.
    assert older.claims == current.claims
    grounded = [c for c in older.claims if c.get("evidence_ids")]
    for resolution in (older.resolve_claim_evidence(c) for c in grounded):
        assert resolution.items == ()
        assert len(resolution.unresolved_ids) == resolution.cited_count


def test_artifact_moves_no_reproducibility_hash(turn, tmp_path):
    """Adding the artifact changes no hash input (config/catalog/prompt, R11).

    The same result is written twice, once with the evidence items and once with an
    empty list. Only ``evidence_items.jsonl`` differs; ``run_record.json`` (which
    carries the three content hashes) and the ``hashes`` block of
    ``run_manifest.json`` are byte-for-byte identical.
    """
    result, cfg = turn
    with_items = write_run_bundle(result, tmp_path / "with", cfg)

    stripped = result.evidence_items
    result.evidence_items = []
    try:
        without_items = write_run_bundle(result, tmp_path / "without", cfg)
    finally:
        result.evidence_items = stripped

    left = compute_checksums(with_items)
    right = compute_checksums(without_items)
    # run_manifest.json carries a generated_at stamp, so compare its hashes block.
    differing = {name for name in left.keys() | right.keys() if left.get(name) != right.get(name)}
    assert differing == {"evidence_items.jsonl", "run_manifest.json"}

    hashes = [json.loads((d / "run_manifest.json").read_text())["hashes"]
              for d in (with_items, without_items)]
    assert hashes[0] == hashes[1]
    assert set(hashes[0]) == {"config_hash", "catalog_hash", "prompt_hash"}
    assert all(hashes[0].values())

    record = json.loads((with_items / "run_record.json").read_text())
    assert hashes[0] == {
        "config_hash": record["config_hash"],
        "catalog_hash": record["catalog_hash"],
        "prompt_hash": record["prompt_hash"],
    }


def test_same_result_writes_a_byte_identical_artifact(turn, tmp_path):
    """The dump is deterministic for a given run: store order in, same bytes out."""
    result, cfg = turn
    first = write_run_bundle(result, tmp_path / "a", cfg) / "evidence_items.jsonl"
    second = write_run_bundle(result, tmp_path / "b", cfg) / "evidence_items.jsonl"

    assert first.read_bytes() == second.read_bytes()
    # Order follows the store's registration order, which is what makes it stable.
    assert [i["evidence_id"] for i in _read_jsonl(first)] == [
        item.evidence_id for item in result.evidence_items
    ]
