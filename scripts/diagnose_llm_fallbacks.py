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
``final_fallback_*``  turns / runs / preferences whose value came from the rule extractor
                      because the model's could not be used. A failed logical call does not
                      always reach this: schema repair may still rescue the turn.

Failure kinds are read from what the provider recorded, never guessed. A category the
provider does not distinguish is reported as ``unclassified`` rather than folded into a
neighbour, so the gap is visible instead of hidden.

Usage
-----
    python scripts/diagnose_llm_fallbacks.py <experiment_dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

#: How a recorded call's metadata maps onto a failure kind. Ordered: the first match wins,
#: so the most specific evidence decides.
_TIMEOUT_MARKERS = ("Timeout", "ReadTimeout", "ConnectTimeout", "LLMTimeout")
_TRANSPORT_MARKERS = ("ConnectError", "ConnectionError", "RemoteProtocolError",
                      "NetworkError", "SSLError")


def _failure_kind(record: dict, meta: dict) -> str | None:
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


def diagnose(exp_dir: Path) -> dict[str, Any]:
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
                outcomes.append((_failure_kind(record, meta), record, meta))

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


def _rate(numerator: int, denominator: int) -> float | None:
    """A rate, or ``None`` when the denominator is zero -- never a fabricated 0.0 or 1.0."""
    return round(numerator / denominator, 6) if denominator else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir")
    parser.add_argument("--json", default=None, help="also write the report to this path")
    args = parser.parse_args()

    report = diagnose(Path(args.experiment_dir))
    counts, rates = report["counts"], report["rates"]

    print(f"{'=' * 72}\nLLM call diagnosis: {report['experiment_dir']}\n{'=' * 72}")
    print("\ndenominators (these count DIFFERENT things):")
    for key in ("logical_calls", "call_records", "http_attempts", "retry_attempts",
                "calls_with_provenance", "recovered_after_retry", "failed_logical_calls",
                "total_runs", "total_candidate_turns"):
        print(f"  {key:<32} {counts[key]}")
    print("\nfinal fallbacks (the substitution actually reached the state):")
    for key in ("final_fallback_preferences", "final_fallback_turns",
                "final_fallback_runs"):
        print(f"  {key:<32} {counts[key]}")
    print("\nrates:")
    for key, value in rates.items():
        shown = "n/a" if value is None else f"{value:.4%}" if value <= 1 else f"{value}"
        print(f"  {key:<32} {shown}")
    print(f"\nsystem_fingerprint_available: {report['system_fingerprint_available']}"
          f"  ({counts['calls_without_system_fingerprint']} of "
          f"{counts['call_records']} call records omitted it)")
    if report["failure_kinds"]:
        print("\nfailure kinds:")
        for kind, n in report["failure_kinds"].items():
            print(f"  {kind:<32} {n}")
    else:
        print("\nfailure kinds: none")
    dist = report["fallback_distribution"]
    if dist["by_field"]:
        print(f"\nfallback by field    : {dist['by_field']}")
        print(f"fallback by scenario : {dist['by_scenario']}")
        print(f"fallback by turn     : {dist['by_turn_index']}")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, sort_keys=True),
                                   encoding="utf-8")
        print(f"\nwritten: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
