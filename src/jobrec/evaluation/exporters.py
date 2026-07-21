"""Run-bundle exporters (landing-plan section 25).

Writes the full, inspectable set of artifacts for a single run so that a later
evaluation guide can read every intermediate object.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml

from ..config import AppConfig
from ..orchestration.orchestrator import TurnResult


def _dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def _write_jsonl(path: Path, items: list) -> None:
    with path.open("w") as fh:
        for item in items:
            fh.write(json.dumps(_dump(item), default=str))
            fh.write("\n")


def write_run_bundle(result: TurnResult, out_dir: str | Path, config: AppConfig) -> Path:
    """Write all per-run artifacts into ``out_dir`` and return the directory."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    decision = result.decision

    _write_json(out / "run_record.json", _dump(result.run_record))
    _write_json(out / "input_snapshot.json", {
        "session_id": result.dialogue_state.session_id,
        "turns": [t.model_dump(mode="json") for t in result.dialogue_state.turns],
    })
    _write_json(out / "candidate_state_before.json", _dump(result.candidate_state_before))
    _write_json(out / "candidate_state_after.json", _dump(result.candidate_state))
    _write_json(out / "dialogue_state.json", _dump(result.dialogue_state))
    _write_json(out / "extracted_preferences.json", _dump(result.extracted_preferences))
    _write_json(out / "active_search_state.json", _dump(result.active_search_state))
    _write_json(out / "job_context_state.json", _dump(result.job_context_state))
    _write_json(out / "retrieval_results.json",
                {"retrieved_job_ids": decision.retrieved_job_ids} if decision else {})
    _write_json(out / "eligibility_results.json",
                [_dump(e) for e in decision.eligibility_results] if decision else [])
    _write_json(out / "recommendation_decision.json", _dump(decision))
    _write_json(out / "response.json", _dump(result.response))
    _write_json(out / "response_claims.json", [_dump(c) for c in result.response.claims])
    _write_jsonl(out / "handoffs.jsonl", result.handoffs)
    _write_jsonl(out / "evidence_log.jsonl", result.evidence_log)
    _write_json(out / "component_latency.json", result.run_record.component_latency_ms)
    _write_jsonl(out / "model_calls.jsonl",
                 [{"call_id": c.call_id, "purpose": c.purpose, "provider": c.provider,
                   "model": c.model, "latency_ms": c.latency_ms} for c in result.model_calls])
    (out / "resolved_config.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
    return out
