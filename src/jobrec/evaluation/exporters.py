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
from ..utils.observability import write_log_trace
from .manifest import build_run_manifest

# Metadata keys that describe how a call was issued (request side). These are
# non-sensitive tuning parameters -- never prompts, API keys or PII.
_REQUEST_PARAM_KEYS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "max_tokens",
    "max_output_tokens",
    "seed",
    "stop",
    "response_format",
    "request_params",
    "params",
)

# Metadata keys that describe what came back (response side). Token usage and
# finish reasons are safe to surface; response text is intentionally excluded.
_RESPONSE_METADATA_KEYS: tuple[str, ...] = (
    "usage",
    "tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "finish_reason",
    "response_id",
    "system_fingerprint",
    "error",
    "retries",
    "response_metadata",
)


def _dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def _model_call_row(call: Any) -> dict[str, Any]:
    """Build an enriched, non-sensitive output row for a single model call.

    Preserves the previously emitted fields (call_id/purpose/provider/model/
    latency_ms) and adds ``request_params`` and ``response_metadata`` derived
    from the call's ``metadata``. Prompts, raw responses and any secrets/API
    keys are NEVER included.
    """
    metadata = getattr(call, "metadata", None) or {}

    request_params: dict[str, Any] = {
        "purpose": call.purpose,
        "provider": call.provider,
        "model": call.model,
    }
    for key in _REQUEST_PARAM_KEYS:
        if key in metadata:
            request_params[key] = metadata[key]

    response_metadata: dict[str, Any] = {
        "parsed_ok": call.parsed_ok,
        "latency_ms": call.latency_ms,
    }
    for key in _RESPONSE_METADATA_KEYS:
        if key in metadata:
            response_metadata[key] = metadata[key]

    return {
        "call_id": call.call_id,
        "purpose": call.purpose,
        "provider": call.provider,
        "model": call.model,
        "latency_ms": call.latency_ms,
        "request_params": request_params,
        "response_metadata": response_metadata,
    }


def _retrieved_job_row(retrieved: Any) -> dict[str, Any]:
    """One recalled job with its retrieval score and score components (R14.1)."""
    return {
        "job_id": getattr(retrieved, "job_id", None),
        "score": getattr(retrieved, "score", None),
        "components": dict(getattr(retrieved, "components", None) or {}),
    }


def _retrieval_results(result: TurnResult, config: AppConfig) -> dict[str, Any]:
    """Retrieval-layer artifact for one run (R14.1).

    Captures the retrieval layer's own output, kept separate from the ranking
    artifacts so retrieval quality can be assessed independently:

    - ``initial_pool`` / ``initial_pool_size`` — the recalled jobs (with their
      retrieval scores) and how many jobs the retriever matched BEFORE the pool
      was truncated to ``retrieval_pool_size``.
    - ``pool_job_ids`` / ``pool_size`` — the pool actually handed to the
      constraint/ranking layers. This differs from ``retrieved_job_ids`` when the
      empty-recall fallback replaced an empty recall with the full catalog.
    - ``expanded`` / ``expansion_reason`` / ``full_catalog_fallback_count`` — the
      fallback-to-full-catalog signal.
    - ``retrieval_latency_ms`` — the retrieval stage's own latency.

    ``executed`` is False when the turn never reached retrieval (clarification
    short-circuit or a failure before the retrieval stage); the remaining fields
    are then ``None``/empty rather than a misleading zero.
    """
    outcome = result.retrieval_outcome
    decision = result.decision
    latency = (result.run_record.component_latency_ms or {}).get("retrieval")
    retrieved = list(getattr(outcome, "retrieved", None) or []) if outcome else []
    retrieved_ids = [r.job_id for r in retrieved]
    # The pool the downstream layers saw: one eligibility result per pooled job.
    pool_ids = ([e.job_id for e in decision.eligibility_results]
                if decision is not None else retrieved_ids)
    expanded = bool(getattr(outcome, "expanded", False)) if outcome else None
    return {
        "executed": outcome is not None,
        "retrieved_job_ids": retrieved_ids,
        "initial_pool": [_retrieved_job_row(r) for r in retrieved],
        "initial_pool_size": getattr(outcome, "initial_pool_size", None) if outcome else None,
        "pool_job_ids": pool_ids,
        "pool_size": len(pool_ids) if outcome is not None else None,
        "requested_pool_size": config.experiment.retrieval_pool_size,
        "expanded": expanded,
        "expansion_reason": getattr(outcome, "expansion_reason", None) if outcome else None,
        "full_catalog_fallback_count": (1 if expanded else 0) if outcome else 0,
        "retrieval_latency_ms": latency,
    }


def _extracted_value_view(result: Any) -> dict[str, Any]:
    """Concise, JSON-serializable view of what was extracted on a turn (R7.3).

    Collapses the turn's ``ExtractedPreferenceSet`` to a small ``field -> value``
    mapping. Intentionally keeps only the normalized values (never full preference
    objects) so the dialogue trace stays small and deterministic.
    """
    eps = getattr(result, "extracted_preferences", None)
    prefs = getattr(eps, "preferences", None) if eps is not None else None
    if not prefs:
        return {}
    return {pref.field_name: pref.normalized_value for pref in prefs}


def _system_clarification_slot(result: Any) -> str | None:
    """The slot the SYSTEM asked about this turn, or ``None`` if it did not ask."""
    clar = getattr(result, "clarification", None)
    if clar is None:
        return None
    fields = getattr(clar, "target_fields", None) or []
    return fields[0] if fields else None


def _last_candidate_utterance(result: Any) -> str | None:
    """Best-effort recovery of the final candidate utterance from dialogue state."""
    dstate = getattr(result, "dialogue_state", None)
    turns = getattr(dstate, "turns", None) or []
    for turn in reversed(turns):
        if getattr(turn, "speaker", None) == "candidate":
            return getattr(turn, "text", None)
    return None


def trace_record(
    result: Any,
    *,
    user_utterance: str | None,
    clarification_slot: str | None = None,
    extracted_value: Any = None,
    termination_reason: str | None = None,
) -> dict[str, Any]:
    """Build one per-turn dialogue-trace record (R7.3, R7.8).

    Captures ``{user_utterance, system_action, clarification_slot, extracted_value,
    state_version, termination_reason}`` for a single turn. ``system_action`` is the
    turn's ``response_type`` (recommendation / clarification / no_match / error) and
    ``state_version`` carries the dialogue- and candidate-state versions AFTER the turn.
    All fields are kept small and JSON-serializable.
    """
    response = getattr(result, "response", None)
    response_type = getattr(response, "response_type", None)
    dstate = getattr(result, "dialogue_state", None)
    cstate = getattr(result, "candidate_state", None)
    return {
        "user_utterance": user_utterance,
        "system_action": str(response_type) if response_type is not None else None,
        "clarification_slot": clarification_slot,
        "extracted_value": extracted_value,
        "state_version": {
            "dialogue_state": getattr(dstate, "version", None),
            "candidate_state": getattr(cstate, "version", None),
        },
        "termination_reason": termination_reason,
    }


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, indent=2, default=str))


def _write_jsonl(path: Path, items: list) -> None:
    with path.open("w") as fh:
        for item in items:
            fh.write(json.dumps(_dump(item), default=str))
            fh.write("\n")


def write_run_bundle(
    result: TurnResult,
    out_dir: str | Path,
    config: AppConfig,
    dialogue_trace: list[dict] | None = None,
    log_trace: list[dict] | None = None,
) -> Path:
    """Write all per-run artifacts into ``out_dir`` and return the directory.

    When ``dialogue_trace`` is provided it is written verbatim as
    ``dialogue_trace.jsonl`` (one record per turn, R7.3). Callers that do not thread
    a trace (the non-clarification single-pass path) get a single-record trace
    derived from the final turn result so every bundle carries a consistent trace.

    ``log_trace`` follows the same pattern for the structured log records (R27.3):
    multi-turn callers thread the records of every turn, single-pass callers fall
    back to the final turn result's own ``log_trace``.
    """
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
    _write_json(out / "retrieval_results.json", _retrieval_results(result, config))
    _write_json(out / "eligibility_results.json",
                [_dump(e) for e in decision.eligibility_results] if decision else [])
    _write_json(out / "recommendation_decision.json", _dump(decision))
    _write_json(out / "response.json", _dump(result.response))
    _write_json(out / "response_claims.json", [_dump(c) for c in result.response.claims])
    _write_json(out / "clarification.json", _dump(result.clarification))
    _write_jsonl(out / "handoffs.jsonl", result.handoffs)
    _write_jsonl(out / "evidence_log.jsonl", result.evidence_log)
    _write_json(out / "component_latency.json", result.run_record.component_latency_ms)
    _write_jsonl(out / "model_calls.jsonl",
                 [_model_call_row(c) for c in result.model_calls])

    # Reproducibility manifest (R11). Build the versions dict from the run
    # record since write_run_bundle only receives ``result`` and ``config``.
    run_record = result.run_record
    versions = {
        "db_version": getattr(run_record, "db_version", None),
        "migration_version": getattr(run_record, "migration_version", None),
    }
    _write_json(out / "run_manifest.json", build_run_manifest(config, run_record, versions))

    # Per-turn dialogue trace (R7.3/R7.8): one record per turn. The clarification
    # loop threads an explicit multi-record trace; single-pass callers fall back to
    # a single-record trace derived from the final result.
    if dialogue_trace is None:
        dialogue_trace = [
            trace_record(
                result,
                user_utterance=_last_candidate_utterance(result),
                clarification_slot=_system_clarification_slot(result),
                extracted_value=_extracted_value_view(result),
            )
        ]
    _write_jsonl(out / "dialogue_trace.jsonl", dialogue_trace)

    # Per-run structured log trace (R27.3): one JSON record per logged event.
    if log_trace is None:
        log_trace = list(getattr(result, "log_trace", None) or [])
    write_log_trace(out, log_trace)

    (out / "resolved_config.yaml").write_text(yaml.safe_dump(config.model_dump(mode="json"), sort_keys=False))
    return out
