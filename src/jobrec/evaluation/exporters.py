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
from ..domain.enums import RunMode
from ..orchestration.orchestrator import TurnResult
from ..utils.observability import write_log_trace
from ..utils.redaction import redact
from .manifest import build_run_manifest

# Metadata keys that describe how a call was issued (request side). These are
# non-sensitive tuning parameters -- never prompts, API keys or PII. A key is
# emitted only when the provider actually recorded it, so an omitted parameter
# (e.g. the temperature the gpt-5 family rejects) stays absent instead of being
# reported as a value that was never sent.
_REQUEST_PARAM_KEYS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "max_tokens",
    "max_output_tokens",
    "seed",
    "stop",
    "response_format",
    # Whether JSON mode was requested, and the client timeout the call ran under.
    "json_mode",
    "timeout_seconds",
    "request_params",
    "params",
)

# Metadata keys that describe what came back (response side). Token usage, finish
# reasons and the retry/fallback trace are safe to surface; prompts, response text
# and secrets are never emitted here (the raw response has its own gated field,
# see ``_model_call_row``).
_RESPONSE_METADATA_KEYS: tuple[str, ...] = (
    "usage",
    "tokens",
    "prompt_tokens",
    "completion_tokens",
    "total_tokens",
    "finish_reason",
    "response_id",
    "system_fingerprint",
    # ``error`` carries an exception CLASS NAME only, never a message (a transport
    # error message can quote the request, and therefore the credential).
    "error",
    "retries",
    # Retry/fallback accounting: how many HTTP attempts the call took, whether the
    # response_format/temperature-dropping retry fired and the status that caused
    # it, and whether the provider gave up and returned its deterministic fallback.
    "attempts",
    "retried_without_response_format",
    "retry_reason",
    "fell_back",
    # ``failed`` marks an attempt that RAISED. Such a row carries no response body, so
    # without this flag it is indistinguishable from a call that returned nothing.
    "failed",
    "response_metadata",
)


def _dump(obj: Any) -> Any:
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json")
    return obj


def _save_raw_responses(config: AppConfig | None) -> bool:
    """Whether raw model responses are retained (``config.llm.save_raw_responses``)."""
    return bool(config.llm.save_raw_responses) if config is not None else True


def _bundle_raw_response(call: Any, config: AppConfig | None) -> str:
    """Redact a raw model response for the bundle exactly as the DB path does.

    Deliberately mirrors ``SqlRepository._model_call_payload`` so the bundle and
    the database share ONE retention policy: :func:`~jobrec.utils.redaction.redact`
    always strips credential- and PII-shaped substrings, and
    ``config.logging.redact_candidate_text`` replaces the text wholesale.
    """
    return redact(
        getattr(call, "raw_response", "") or "",
        redact_candidate_text=(
            bool(config.logging.redact_candidate_text) if config is not None else False
        ),
    )


def _model_call_row(call: Any, config: AppConfig | None = None,
                    turn_index: int | None = None,
                    turn_run_id: str | None = None) -> dict[str, Any]:
    """Build an enriched, non-sensitive output row for a single model call.

    Preserves the previously emitted fields (call_id/purpose/provider/model/
    latency_ms) and adds ``request_params`` and ``response_metadata`` derived
    from the call's ``metadata`` (token usage, finish reason, retry/fallback
    trace).

    ``raw_response`` is included only while ``config.llm.save_raw_responses`` is on
    (the default) and is redacted on the way out by :func:`_bundle_raw_response`,
    the same policy the database write path applies. When the setting is off the
    field is ABSENT rather than empty, so "not retained" is never confused with
    "the model returned nothing". Prompts and secrets/API keys are NEVER included,
    in either mode.

    ``call_id`` is the ``content_id("call", purpose, prompt)`` the remote provider
    computed, so a row carrying ``call_id`` + ``raw_response`` is replayable by
    :class:`~jobrec.llm.replay.ReplayProvider` without persisting the prompt.

    ``turn_index`` / ``turn_run_id`` attribute the call to the dialogue turn that
    made it. Without them a multi-turn bundle's calls are an unordered heap and no
    per-turn cost/latency question can be answered offline; they are omitted (rather
    than emitted as null) when the caller does not thread turn results, so existing
    single-turn rows keep their exact previous shape.
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

    row: dict[str, Any] = {
        "call_id": call.call_id,
        "purpose": call.purpose,
        "provider": call.provider,
        "model": call.model,
        "latency_ms": call.latency_ms,
        "request_params": request_params,
        "response_metadata": response_metadata,
    }
    if turn_index is not None:
        row["turn_index"] = turn_index
    if turn_run_id is not None:
        row["turn_run_id"] = turn_run_id
    if _save_raw_responses(config):
        row["raw_response"] = _bundle_raw_response(call, config)
    return row


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


def _turn_record_row(turn_index: int, turn: TurnResult) -> dict[str, Any]:
    """One auditable row per dialogue turn of a run.

    ``run_record.json`` only ever described the FINAL turn (its ``run_id`` is the
    final turn's), so for a multi-turn run every earlier turn's outcome, latency and
    call count were unrecoverable from the bundle. This artifact closes that gap
    WITHOUT reshaping ``run_record.json`` into a per-turn list, which would have
    broken every existing reader for no analytical gain.
    """
    rr = turn.run_record
    return {
        "turn_index": turn_index,
        "run_id": rr.run_id,
        "response_type": turn.response.response_type,
        "success": rr.success,
        "failure_code": rr.failure_code,
        "total_latency_ms": rr.total_latency_ms,
        "component_latency_ms": dict(rr.component_latency_ms or {}),
        "model_call_count": len(turn.model_calls or []),
        "asked_clarification": turn.clarification is not None,
    }


def _run_totals(turns: list[TurnResult], config: AppConfig) -> dict[str, Any]:
    """Whole-run aggregates over every turn (R7.3/R11.1).

    Only genuinely additive quantities are summed here: latency and model-call
    accounting. Ranking/HCSR/grounding metrics intentionally keep scoring the final
    turn, because the final response IS the run's answer -- see
    :func:`write_run_bundle`.

    ``model_call_coverage.complete`` is reported, never enforced. A finished run must
    not be destroyed by an accounting assertion, and zero calls is the CORRECT state
    for the deterministic backend; recording the gap instead of raising keeps it an
    auditable data-quality signal rather than a lost run.
    """
    components: dict[str, float] = {}
    for turn in turns:
        for name, ms in (turn.run_record.component_latency_ms or {}).items():
            components[name] = round(components.get(name, 0.0) + float(ms), 3)

    calls_per_turn = [len(t.model_calls or []) for t in turns]
    deterministic = getattr(getattr(config, "llm", None), "mode", None) == (
        RunMode.DETERMINISTIC)
    return {
        "turn_count": len(turns),
        "total_latency_ms": round(
            sum(float(t.run_record.total_latency_ms or 0.0) for t in turns), 3),
        "component_latency_ms": components,
        "final_turn_total_latency_ms": turns[-1].run_record.total_latency_ms,
        "final_turn_run_id": turns[-1].run_record.run_id,
        "model_call_total": sum(calls_per_turn),
        "model_calls_per_turn": calls_per_turn,
        "model_call_coverage": {
            # Deterministic runs legitimately make no calls, so "complete" can only
            # mean "no turn is missing a call" for a model-backed run.
            "expects_calls": not deterministic,
            "turns_without_calls": sum(1 for n in calls_per_turn if n == 0),
            "complete": deterministic or all(n > 0 for n in calls_per_turn),
        },
    }


def write_run_bundle(
    result: TurnResult,
    out_dir: str | Path,
    config: AppConfig,
    dialogue_trace: list[dict] | None = None,
    log_trace: list[dict] | None = None,
    turn_results: list[TurnResult] | None = None,
) -> Path:
    """Write all per-run artifacts into ``out_dir`` and return the directory.

    When ``dialogue_trace`` is provided it is written verbatim as
    ``dialogue_trace.jsonl`` (one record per turn, R7.3). Callers that do not thread
    a trace (the non-clarification single-pass path) get a single-record trace
    derived from the final turn result so every bundle carries a consistent trace.

    ``log_trace`` follows the same pattern for the structured log records (R27.3):
    multi-turn callers thread the records of every turn, single-pass callers fall
    back to the final turn result's own ``log_trace``.

    ``turn_results`` is the ordered list of EVERY turn result of the run (the
    scripted turns plus the clarification loop's answer turns), of which ``result``
    is the last. Threading it makes three artifacts whole-run instead of final-turn:

    * ``model_calls.jsonl`` covers all turns, each row attributed to its turn;
    * ``component_latency.json`` sums each component across turns;
    * ``turn_records.jsonl`` is added, one ``run_record``-shaped row per turn.

    Metric definitions are deliberately NOT affected: ranking, HCSR and grounding
    keep scoring the final turn, because the final response is the run's answer.
    Only the genuinely additive quantities (latency, call accounting) aggregate.

    ``model_calls.jsonl`` carries one :func:`_model_call_row` per recorded model
    call. ``config`` governs raw-response retention there: with
    ``config.llm.save_raw_responses`` on (the default) each row carries a redacted
    ``raw_response`` so the run is replayable offline; with it off the field is
    omitted. Prompts are never written.

    ``evidence_items.jsonl`` carries the session's registered ``EvidenceItem``s so
    the ``evidence_ids`` on ``response_claims.json`` can be resolved to the field,
    source object and value they cite without re-running the pipeline. Note that
    ``EvidenceItem.observed_at`` is a wall-clock stamp, so this artifact is
    reproducible for a given result but (like ``run_record.json`` and
    ``run_manifest.json``) not byte-identical across two separate executions.
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
    # The evidence ITEMS (what a claim cites), distinct from the per-stage decision
    # log above: one record per registered EvidenceItem, so every ``evidence_id`` on
    # a claim in ``response_claims.json`` resolves offline to its source object,
    # field, raw text and normalized value. Written for every bundle from the
    # session-scoped store, so a claim citing an earlier turn still resolves.
    _write_jsonl(out / "evidence_items.jsonl", result.evidence_items)
    # The FINAL TURN's component latency. Deliberately left final-turn-scoped (and
    # shape-unchanged) because ``retrieval_results.json`` above reads the same record,
    # so mixing a whole-run sum in here would silently make that snapshot incoherent.
    # The whole-run sums live in ``run_totals.json`` below.
    _write_json(out / "component_latency.json", result.run_record.component_latency_ms)

    # Whole-run view. ``result`` is the final turn; ``turns`` is every turn of the
    # run, or just the final one when the caller does not thread turn results.
    turns: list[TurnResult] = list(turn_results) if turn_results else [result]
    # ``config`` governs raw-response retention/redaction here exactly as it does
    # on the database write path (``config.llm.save_raw_responses`` +
    # ``config.logging.redact_candidate_text``).
    #
    # Every turn's calls, each attributed to its turn. Previously only the FINAL
    # turn's calls were written, so in a 3-turn hybrid run the earlier turns' calls
    # -- extraction, repair retries, clarification phrasing -- existed in no artifact
    # at all: they were neither replayable nor countable towards tokens/cost.
    _write_jsonl(out / "model_calls.jsonl", [
        _model_call_row(c, config, turn_index=i, turn_run_id=t.run_record.run_id)
        for i, t in enumerate(turns)
        for c in (t.model_calls or [])
    ])
    _write_jsonl(out / "turn_records.jsonl", [_turn_record_row(i, t)
                                              for i, t in enumerate(turns)])
    _write_json(out / "run_totals.json", _run_totals(turns, config))

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
