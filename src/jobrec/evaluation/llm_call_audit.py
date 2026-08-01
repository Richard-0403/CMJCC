"""Classify every LLM call in an experiment and reconcile the fallback counts.

Why this exists
---------------
The first report of this said "32 calls, 22 with provenance, 10 without usable JSON, 4
extraction_model_call_failed, 6 fallbacks". Those five numbers count five different things --
HTTP attempts, successful responses, failed attempts, failed TURNS and affected PREFERENCES --
and mixing them makes any threshold meaningless. Deciding whether an endpoint is fit for the
official batch needs one table with the denominators stated.

Definitions used here, and they are not interchangeable:

``logical_calls``      one request the pipeline asked for. A bounded retry is still one
                       logical call, however many times it went over the wire.
``http_attempts``      individual requests actually sent, summed from each record's
                       ``attempts``. Always >= logical_calls.
``recovered_after_retry``  a logical call that failed at least once and then succeeded. The
                       pipeline saw no failure; the endpoint did.
``failed_logical_calls``   a logical call that never returned usable JSON. This is what
                       "the endpoint did not answer" means.
``final_fallback_*``   turns / runs / preferences whose value came from the rule extractor
                       because the model's could not be used. A failed logical call does not
                       always reach this: schema repair may still rescue the turn.

Failure kinds are read from what the provider recorded, never guessed. A category the provider
does not distinguish is reported as ``unclassified`` rather than folded into a neighbour, so the
gap is visible instead of hidden.

This lives in the package, not in ``scripts/``, because the experiment runner writes the summary
into the experiment manifest and ``scripts/diagnose_llm_fallbacks.py`` prints it. Two
implementations of "logical call" would drift, and the fallback rate the manifest states has to
be the same number the audit script reports.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

#: How a recorded call's metadata maps onto a failure kind. Ordered: the first match wins,
#: so the most specific evidence decides.
_TIMEOUT_MARKERS = ("Timeout", "ReadTimeout", "ConnectTimeout", "LLMTimeout")
_TRANSPORT_MARKERS = ("ConnectError", "ConnectionError", "RemoteProtocolError",
                      "NetworkError", "SSLError")

#: Keys of the compact summary embedded in the experiment manifest. The full report (per-field
#: and per-scenario distributions, every fallback occurrence) stays in the audit artifact; the
#: manifest carries the counts and rates a threshold is checked against.
MANIFEST_SUMMARY_KEYS = ("counts", "rates", "failure_kinds", "fallback_distribution",
                         "system_fingerprint_available")


def failure_kind(record: dict, meta: dict) -> str | None:
    """The failure kind of one call record, or ``None`` when it succeeded."""
    error = str(meta.get("error") or "")
    retry_reason = str(meta.get("retry_reason") or "")
    finish = str(meta.get("finish_reason") or "")
    raw = record.get("raw_response")

    if error:
        if any(m in error for m in _TIMEOUT_MARKERS):
            return "timeout"
        if "InvalidJSON" in error or "JSONDecode" in error:
            return "json_parse_error"
        if "HTTPStatus" in error or retry_reason.startswith("http_"):
            return _http_kind(retry_reason)
        if any(m in error for m in _TRANSPORT_MARKERS) or error == "LLMError":
            # Bare ``LLMError`` is what RemoteLLMProvider._chat raises from httpx.HTTPError,
            # i.e. the request did not come back usable at the transport level. Named rather
            # than left as "other", because the distinction decides the remedy: a transport
            # failure is not fixed by better prompting or by native structured output.
            return "transport_error"
        return f"other_error:{error}"

    if record.get("parsed_ok") is False or meta.get("fell_back"):
        if finish == "length":
            return "truncated_response"
        if not raw or not str(raw).strip():
            return "empty_response"
        return "json_parse_error"

    # Succeeded, but the endpoint had to be asked more than once.
    return None


def _http_kind(retry_reason: str) -> str:
    if not retry_reason.startswith("http_"):
        return "http_error"
    code = retry_reason.removeprefix("http_")
    if code.startswith("4"):
        return f"http_4xx:{code}"
    if code.startswith("5"):
        return f"http_5xx:{code}"
    return f"http_other:{code}"


def _run_label(path: Path) -> str:
    return "/".join(path.parts[-4:-1])


def _rate(numerator: int, denominator: int) -> float | None:
    """A rate, or ``None`` when the denominator is zero -- never a fabricated 0.0 or 1.0."""
    return round(numerator / denominator, 6) if denominator else None


def audit_llm_calls(exp_dir: Path) -> dict[str, Any]:
    """Every LLM call under ``exp_dir``, grouped into logical calls and classified.

    Reads ``model_calls.jsonl`` for what went over the wire and ``dialogue_state.json`` for
    what the SUBSTITUTION actually reached. Both are needed: a failed call that schema repair
    rescued leaves no fallback in the state, and counting failures alone would overstate the
    damage while counting fallbacks alone would understate how often the endpoint dropped.
    """
    logical = 0
    http_attempts = 0
    recovered = 0
    failed_logical = 0
    kinds: Counter = Counter()
    by_purpose: Counter = Counter()
    unavailable_fingerprint = 0
    with_provenance = 0

    fallback_prefs: list[dict] = []
    fallback_turns: set[tuple[str, int]] = set()
    fallback_runs: set[str] = set()
    field_counter: Counter = Counter()
    scenario_counter: Counter = Counter()
    turn_counter: Counter = Counter()

    records = 0
    for calls_path in sorted(exp_dir.rglob("model_calls.jsonl")):
        label = _run_label(calls_path)
        # Records are grouped into LOGICAL calls. ``retry_call`` appends one record per
        # attempt, so counting records as calls both inflates the denominator and hides
        # recovery: a rescued call looks like one failure plus one unrelated success. The
        # grouping key is what the pipeline treats as one request -- the run, the turn it
        # belongs to, and its purpose.
        groups: dict[tuple, list[dict]] = {}
        order: list[tuple] = []
        for line in calls_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            record = json.loads(line)
            records += 1
            key = (label, record.get("turn_run_id"), record.get("turn_index"),
                   record.get("purpose"))
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(record)

        for key in order:
            attempts_records = groups[key]
            logical += 1
            outcomes = []
            for record in attempts_records:
                meta = record.get("response_metadata") or {}
                inner = meta.get("attempts")
                http_attempts += int(inner) if isinstance(inner, int) else 1
                if meta.get("response_id") or meta.get("response_model"):
                    with_provenance += 1
                if not meta.get("system_fingerprint"):
                    unavailable_fingerprint += 1
                outcomes.append((failure_kind(record, meta), record, meta))

            failures = [(k, r, m) for k, r, m in outcomes if k is not None]
            succeeded = any(k is None for k, _r, _m in outcomes)
            if succeeded and failures:
                # The pipeline saw no failure; the endpoint did. Counted separately, because
                # a run that recovered is not evidence the endpoint is healthy.
                recovered += 1
            if not succeeded:
                failed_logical += 1
            # Every failed ATTEMPT is classified, whether or not the call recovered: the
            # failure kinds describe endpoint behaviour, not run outcomes.
            for kind, record, _meta in failures:
                kinds[kind] += 1
                by_purpose[f"{record.get('purpose')}:{kind}"] += 1

    # Final fallbacks are read from the extraction snapshots, which is where the
    # SUBSTITUTION is recorded -- a failed call that schema repair rescued leaves no mark
    # here, which is exactly the distinction being drawn.
    for ds_path in sorted(exp_dir.rglob("dialogue_state.json")):
        label = _run_label(ds_path)
        scenario = ds_path.parts[-3]
        state = json.loads(ds_path.read_text(encoding="utf-8"))
        turns = [t for t in state.get("turns", []) if t.get("speaker") == "candidate"]
        for index, turn in enumerate(turns):
            for pref in (turn.get("extraction_snapshot") or {}).get("preferences", []):
                source = str((pref.get("metadata") or {}).get("extraction_source") or "")
                if "fallback" not in source:
                    continue
                fallback_prefs.append({"run": label, "scenario": scenario, "turn": index,
                                       "field": pref.get("field_name"), "source": source})
                fallback_turns.add((label, index))
                fallback_runs.add(label)
                field_counter[str(pref.get("field_name"))] += 1
                scenario_counter[scenario] += 1
                turn_counter[str(index)] += 1

    total_runs = len({_run_label(p) for p in exp_dir.rglob("dialogue_state.json")})
    total_turns = sum(
        len([t for t in json.loads(p.read_text(encoding="utf-8")).get("turns", [])
             if t.get("speaker") == "candidate"])
        for p in exp_dir.rglob("dialogue_state.json"))

    return {
        "experiment_dir": str(exp_dir),
        "counts": {
            "logical_calls": logical,
            "call_records": records,
            "http_attempts": http_attempts,
            "retry_attempts": http_attempts - logical,
            "calls_with_provenance": with_provenance,
            "calls_without_system_fingerprint": unavailable_fingerprint,
            "recovered_after_retry": recovered,
            "failed_logical_calls": failed_logical,
            "total_runs": total_runs,
            "total_candidate_turns": total_turns,
            "final_fallback_preferences": len(fallback_prefs),
            "final_fallback_turns": len(fallback_turns),
            "final_fallback_runs": len(fallback_runs),
        },
        "rates": {
            "logical_call_success_rate": _rate(logical - failed_logical, logical),
            "retry_recovery_rate": _rate(recovered, recovered + failed_logical),
            "final_fallback_call_rate": _rate(failed_logical, logical),
            "final_fallback_turn_rate": _rate(len(fallback_turns), total_turns),
            "final_fallback_run_rate": _rate(len(fallback_runs), total_runs),
        },
        "failure_kinds": dict(sorted(kinds.items())),
        "failure_kinds_by_purpose": dict(sorted(by_purpose.items())),
        "fallback_distribution": {
            "by_field": dict(field_counter.most_common()),
            "by_scenario": dict(scenario_counter.most_common()),
            "by_turn_index": dict(sorted(turn_counter.items())),
        },
        "fallback_occurrences": fallback_prefs,
        # False when NO record carried one, which is what "this endpoint does not report it"
        # means. Recorded as an explicit fact so the thesis can state it rather than leave a
        # blank field looking like an oversight.
        "system_fingerprint_available": unavailable_fingerprint < records,
    }


def manifest_summary(exp_dir: Path) -> dict[str, Any]:
    """The retry/fallback summary embedded in the experiment manifest.

    Kept compact deliberately: the counts and rates a pre-registered threshold is checked
    against, plus the distribution that shows whether fallbacks CONCENTRATED on one scenario,
    turn or field. The per-occurrence list stays out, because a manifest is read by hand and a
    batch can produce thousands of them; the audit artifact holds the full report.

    Returns an empty dict when the batch made no model calls at all -- a deterministic run has
    nothing to summarise, and writing zeros would claim a measured 0% fallback rate where there
    was no denominator.
    """
    report = audit_llm_calls(exp_dir)
    if not report["counts"]["logical_calls"]:
        return {}
    summary = {key: report[key] for key in MANIFEST_SUMMARY_KEYS}
    summary["note"] = (
        "logical_calls counts requests the pipeline asked for; http_attempts counts requests "
        "sent. A bounded retry is one logical call. final_fallback_* is where the substitution "
        "reached the dialogue state, which a repaired call does not.")
    return summary
