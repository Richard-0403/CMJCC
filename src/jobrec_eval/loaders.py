"""Load run bundles from an experiment directory into normalized tables."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

_VARIANTS = {"full", "profile_only", "one_shot", "no_memory", "no_context"}


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


@dataclass
class RunBundle:
    variant: str
    scenario_id: str
    run_index: int
    path: Path
    run_record: dict
    decision: dict | None
    response: dict | None
    claims: list[dict]
    handoffs: list[dict]
    evidence_log: list[dict]
    latency: dict
    active_search: dict | None
    job_context: dict | None
    clarification: dict | None = None
    dialogue_state: dict | None = None
    dialogue_trace: list[dict] | None = None
    # ExtractedPreferenceSet dump; each preference's ``metadata`` carries
    # ``extraction_method`` and ``extraction_source`` (R13.1).
    extracted_preferences: dict | None = None
    # retrieval_results.json: initial pool + scores, pool size, full-catalog
    # fallback signal and retrieval latency (R14.1).
    retrieval: dict | None = None

    @property
    def run_id(self) -> str:
        return self.run_record.get("run_id", "")

    @property
    def response_turns(self) -> int:
        """Number of dialogue turns for this run (one trace record per turn, R7.8)."""
        return len(self.dialogue_trace or [])

    @property
    def termination_reason(self) -> str | None:
        """Terminal outcome of the dialogue loop, read from the final trace record."""
        if not self.dialogue_trace:
            return None
        return self.dialogue_trace[-1].get("termination_reason")


def discover_experiment_dir(runs_root: str | Path, experiment_id: str | None = None) -> Path:
    root = Path(runs_root)
    if experiment_id:
        return root / experiment_id
    candidates = sorted(root.glob("exp-*"))
    if not candidates:
        raise FileNotFoundError(f"no experiment directories under {root}")
    return candidates[-1]


def load_bundles(experiment_dir: str | Path) -> list[RunBundle]:
    """Walk {variant}/{scenario_id}/{run_index}/ and load every run bundle."""
    exp = Path(experiment_dir)
    bundles: list[RunBundle] = []
    for variant_dir in sorted(exp.iterdir()):
        if not variant_dir.is_dir() or variant_dir.name not in _VARIANTS:
            continue
        for scen_dir in sorted(variant_dir.iterdir()):
            if not scen_dir.is_dir():
                continue
            for run_dir in sorted(scen_dir.iterdir(), key=lambda p: p.name):
                if not run_dir.is_dir():
                    continue
                rr = _read_json(run_dir / "run_record.json")
                if rr is None:
                    continue
                bundles.append(RunBundle(
                    variant=variant_dir.name,
                    scenario_id=scen_dir.name,
                    run_index=int(run_dir.name) if run_dir.name.isdigit() else 0,
                    path=run_dir,
                    run_record=rr,
                    decision=_read_json(run_dir / "recommendation_decision.json"),
                    response=_read_json(run_dir / "response.json"),
                    claims=_read_json(run_dir / "response_claims.json") or [],
                    handoffs=_read_jsonl(run_dir / "handoffs.jsonl"),
                    evidence_log=_read_jsonl(run_dir / "evidence_log.jsonl"),
                    latency=_read_json(run_dir / "component_latency.json") or {},
                    active_search=_read_json(run_dir / "active_search_state.json"),
                    job_context=_read_json(run_dir / "job_context_state.json"),
                    clarification=_read_json(run_dir / "clarification.json"),
                ))
                bundles[-1].dialogue_state = _read_json(run_dir / "dialogue_state.json")
                bundles[-1].dialogue_trace = _read_jsonl(run_dir / "dialogue_trace.jsonl")
                bundles[-1].extracted_preferences = _read_json(
                    run_dir / "extracted_preferences.json")
                bundles[-1].retrieval = _read_json(run_dir / "retrieval_results.json")
    return bundles


# --------------------------------------------------------------- normalized tables
def normalize(bundles: list[RunBundle]) -> dict[str, pd.DataFrame]:
    """Produce the normalized CSV-ready tables described in section 7."""
    runs, recs, checks, claims, handoffs, logs, latency = [], [], [], [], [], [], []

    for b in bundles:
        d = b.decision or {}
        no_match_returned = bool(d.get("no_match", False))
        selected = set(d.get("selected_job_ids", []))
        ranked = {r["job_id"]: r for r in d.get("ranked_jobs", [])}
        elig = {e["job_id"]: e for e in d.get("eligibility_results", [])}

        runs.append({
            "run_id": b.run_id, "scenario_id": b.scenario_id, "variant": b.variant,
            "repeat_index": b.run_index,
            "success": b.run_record.get("success"),
            "failure_code": b.run_record.get("failure_code"),
            "response_type": (b.response or {}).get("response_type"),
            "no_match_returned": no_match_returned,
            "no_match_reason_codes": ";".join(d.get("no_match_reason_codes", [])),
            "turn_count": len([t for t in ((b.dialogue_state or {}).get("turns", []))
                               if t.get("speaker") == "candidate"]) or None,
            "selected_count": len(selected),
            "total_latency_ms": b.run_record.get("total_latency_ms"),
            "config_hash": b.run_record.get("config_hash"),
            "catalog_hash": b.run_record.get("catalog_hash"),
            "prompt_hash": b.run_record.get("prompt_hash"),
            "code_version": b.run_record.get("code_version"),
            "clarification_fields": ";".join((b.clarification or {}).get("target_fields", []) if b.clarification else []),
        })

        for jid in d.get("selected_job_ids", []):
            rj = ranked.get(jid, {})
            er = elig.get(jid, {})
            rank = rj.get("rank")
            recs.append({
                "run_id": b.run_id, "scenario_id": b.scenario_id, "variant": b.variant,
                "repeat_index": b.run_index, "rank": rank, "job_id": jid,
                "total_score": rj.get("total_score"),
                "self_eligible": er.get("eligible"),
                "self_hard_violation_count": er.get("hard_violation_count"),
                "skill_gap_count": len(rj.get("skill_gaps", [])),
                "selected": True,
            })

        for er in d.get("eligibility_results", []):
            for c in er.get("checks", []):
                checks.append({
                    "run_id": b.run_id, "job_id": er["job_id"],
                    "constraint_id": c.get("constraint_id"), "field_name": c.get("field_name"),
                    "outcome": c.get("outcome"), "explanation_code": c.get("explanation_code"),
                })

        for c in b.claims:
            claims.append({
                "run_id": b.run_id, "scenario_id": b.scenario_id, "variant": b.variant,
                "claim_id": c.get("claim_id"), "claim_type": c.get("claim_type"),
                "support_status": c.get("support_status"),
                "supported_binary": 1 if c.get("support_status") == "supported" else 0,
                "evidence_count": len(c.get("evidence_ids", [])),
            })

        for h in b.handoffs:
            handoffs.append({
                "run_id": b.run_id, "handoff_id": h.get("handoff_id"),
                "from_component": h.get("from_component"), "to_component": h.get("to_component"),
                "validation_passed": h.get("validation_passed"), "status": h.get("status"),
                "error_code": h.get("error_code"),
            })

        for log in b.evidence_log:
            logs.append({
                "run_id": b.run_id, "stage": log.get("stage"), "event_type": log.get("event_type"),
                "actor": log.get("actor"), "status": log.get("status"),
                "rule_id": log.get("rule_id"), "error_code": log.get("error_code"),
            })

        for comp, ms in (b.latency or {}).items():
            latency.append({"run_id": b.run_id, "variant": b.variant, "component": comp, "latency_ms": ms})

    return {
        "runs": pd.DataFrame(runs),
        "recommendations": pd.DataFrame(recs),
        "constraint_checks": pd.DataFrame(checks),
        "claims": pd.DataFrame(claims),
        "handoffs": pd.DataFrame(handoffs),
        "decision_logs": pd.DataFrame(logs),
        "component_latency": pd.DataFrame(latency),
    }
