"""Pre-flight invariant check for a hybrid experiment archive.

Run a small hybrid batch, then point this at its run-bundle directory. It asserts the
invariants that must hold before a long, expensive batch is worth starting, and exits
non-zero naming every one that failed.

    python -m jobrec_eval.cli pipeline --config configs/hybrid_vectorengine.yaml \
        --scenarios <small.jsonl> --catalog data/processed/jobs.jsonl \
        --variants full,no_memory,no_context --repeats 1 --out-root <root>
    python scripts/hybrid_smoke_check.py <root>/_runs/<experiment_id> [<analysis_dir>]

Why a script rather than eyeballing: each of these invariants corresponds to a defect
that was actually found in this archive (calls attributed to no turn, clarification
rephrasing discarded at the call site, failed attempts leaving no record, whole-run
latency reported as the final turn's), and an expensive batch should not be started on
the assumption that they stayed fixed.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

#: Metadata keys that carry token usage on a successful call.
_USAGE_KEYS = ("usage", "tokens", "prompt_tokens", "completion_tokens", "total_tokens",
               "input_tokens", "output_tokens")

#: Patterns matching credential VALUES. Deliberately not field names: the archive is
#: SUPPOSED to record ``api_key_env`` (the variable's name) and ``api_key_present`` (a
#: boolean), which is how it evidences that a key was configured without revealing it,
#: and ``work_authorizations`` is a job field that merely contains "authorization".
#: Matching names instead of values produced five false positives and would have taught
#: whoever runs this to ignore the check.
_SECRET_VALUE_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-.]{20,}"),
)

#: Environment variables whose literal value must not appear in any artifact. Scanning for
#: the actual configured secret is the decisive test; the regexes above only catch shapes.
_SECRET_ENV_VARS = ("JOBREC_LLM_API_KEY",)

#: Report phrasings that are false under a hybrid backend.
_WRONG_BACKEND_PHRASES = (
    "deterministic mock provider",
    "does not exercise a real model",
    "Within this controlled, deterministic setup",
    "a real LLM backend are the natural next steps",
    "no model calls recorded",
)


def _jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


class Checker:
    def __init__(self) -> None:
        self.failures: list[str] = []
        self.notes: list[str] = []

    def check(self, ok: bool, label: str, detail: str = "") -> None:
        if not ok:
            self.failures.append(f"{label}{': ' + detail if detail else ''}")

    def note(self, text: str) -> None:
        self.notes.append(text)


def main(exp_dir: Path, analysis_dir: Path | None) -> int:
    c = Checker()
    manifest = _json(exp_dir / "experiment_manifest.json")
    if manifest is None:
        print(f"FAIL: no experiment_manifest.json in {exp_dir}")
        return 2

    # ---- batch accounting -------------------------------------------------
    expected = manifest.get("expected_run_count")
    completed = manifest.get("run_count")
    crashed = manifest.get("crashed_run_count")
    c.check(expected is not None and crashed is not None,
            "manifest records expected_run_count and crashed_run_count")
    if expected is not None and crashed is not None:
        c.check(expected == (completed or 0) + crashed,
                "expected_run_count == completed + crashed",
                f"{expected} != {completed} + {crashed}")
    if crashed:
        c.note(f"{crashed} run(s) crashed; each must appear in failures.csv")

    run_dirs = sorted(p for p in exp_dir.glob("*/*/*") if (p / "run_record.json").exists())
    c.check(len(run_dirs) == (completed or 0),
            "every completed run has a bundle on disk",
            f"{len(run_dirs)} bundles vs run_count {completed}")

    total_calls = 0
    usage_calls = 0
    token_total = 0
    failed_calls = 0
    purposes: set[str] = set()

    for run_dir in run_dirs:
        rel = run_dir.relative_to(exp_dir).as_posix()
        calls = _jsonl(run_dir / "model_calls.jsonl")
        turns = _jsonl(run_dir / "turn_records.jsonl")
        totals = _json(run_dir / "run_totals.json")

        c.check(bool(turns), f"[{rel}] turn_records.jsonl is present and non-empty")
        c.check(totals is not None, f"[{rel}] run_totals.json is present")
        if totals is None or not turns:
            continue

        # ---- turn attribution --------------------------------------------
        turn_ids = {t["turn_index"]: t["run_id"] for t in turns}
        for call in calls:
            total_calls += 1
            purposes.add(str(call.get("purpose")))
            c.check("turn_index" in call and "turn_run_id" in call,
                    f"[{rel}] every model call carries turn_index and turn_run_id")
            index = call.get("turn_index")
            c.check(index in turn_ids,
                    f"[{rel}] call turn_index {index} matches a recorded turn")
            if index in turn_ids:
                c.check(call.get("turn_run_id") == turn_ids[index],
                        f"[{rel}] call turn_run_id matches that turn's run_id")

            meta = call.get("response_metadata") or {}
            if meta.get("failed"):
                failed_calls += 1
                # A failed attempt has no body and no usage: it must say so rather than
                # report zero tokens, which would understate cost as a measurement.
                c.check(not call.get("raw_response"),
                        f"[{rel}] a failed attempt carries no response body")
                c.check(not any(k in meta for k in _USAGE_KEYS),
                        f"[{rel}] a failed attempt reports no token usage "
                        f"(absent, not zero)")
                c.check(str(call.get("call_id", "")).find("#failed") >= 0,
                        f"[{rel}] a failed attempt's call_id is suffixed so it cannot "
                        f"shadow a successful recording")
            else:
                c.check("raw_response" in call,
                        f"[{rel}] a successful call retains its raw_response "
                        f"(save_raw_responses is on)")
                if any(k in meta for k in _USAGE_KEYS):
                    usage_calls += 1
                    usage = meta.get("usage")
                    if isinstance(usage, dict):
                        token_total += int(usage.get("total_tokens") or 0)
                    else:
                        token_total += int(meta.get("total_tokens") or 0)
            c.check("prompt" not in call, f"[{rel}] no row carries the prompt")

        # ---- call accounting ---------------------------------------------
        c.check(totals["model_call_total"] == len(calls),
                f"[{rel}] run_totals.model_call_total == model_calls.jsonl lines",
                f"{totals['model_call_total']} != {len(calls)}")
        per_turn = totals["model_calls_per_turn"]
        c.check(sum(per_turn) == len(calls),
                f"[{rel}] per-turn call counts sum to the archived calls")
        c.check([t["model_call_count"] for t in turns] == per_turn,
                f"[{rel}] turn_records call counts agree with run_totals")
        c.check(totals["model_call_coverage"]["expects_calls"] is True,
                f"[{rel}] a hybrid run is expected to make model calls")
        c.check(totals["model_call_coverage"]["complete"] is True,
                f"[{rel}] every turn recorded at least one call")

        # ---- latency ------------------------------------------------------
        final = totals["final_turn_total_latency_ms"] or 0.0
        c.check(totals["total_latency_ms"] >= final,
                f"[{rel}] whole-run latency is not below the final turn's",
                f"{totals['total_latency_ms']} < {final}")
        if len(turns) > 1:
            c.check(totals["total_latency_ms"] > final,
                    f"[{rel}] a multi-turn run costs more than its last turn alone")

        # ---- sessions -----------------------------------------------------
        c.check(all(t.get("session_id") for t in turns),
                f"[{rel}] every turn records its session_id")

    # ---- archive-wide ------------------------------------------------------
    c.check(total_calls > 0, "the batch recorded at least one model call")
    c.check("intent_extraction" in purposes,
            "extraction calls are archived", f"purposes seen: {sorted(purposes)}")
    c.check("clarification" in purposes,
            "at least one clarification rephrasing call is archived -- include a "
            "clarification scenario in the smoke set",
            f"purposes seen: {sorted(purposes)}")

    leaks = _scan_for_secrets(exp_dir)
    c.check(not leaks, "no archived file contains credential material",
            "; ".join(leaks[:5]))

    if analysis_dir is not None:
        report = analysis_dir / "report" / "analysis_report.md"
        if not report.exists():
            c.check(False, "analysis report exists", str(report))
        else:
            text = report.read_text(encoding="utf-8")
            for phrase in _WRONG_BACKEND_PHRASES:
                c.check(phrase not in text,
                        "hybrid report does not describe a deterministic/mock backend",
                        f"found {phrase!r}")
            c.check("INCOMPLETE EXPERIMENT" not in text
                    or bool(crashed),
                    "the report's completeness claim matches the manifest")

    print(f"runs: {len(run_dirs)}  calls: {total_calls}  failed attempts: {failed_calls}  "
          f"calls with usage: {usage_calls}  total tokens: {token_total}")
    print(f"purposes: {sorted(purposes)}")
    for note in c.notes:
        print(f"note: {note}")
    if c.failures:
        print(f"\nFAIL: {len(c.failures)} invariant(s) violated")
        for failure in dict.fromkeys(c.failures):
            print(f"  - {failure}")
        return 1
    print("\nOK: every pre-flight invariant holds")
    return 0


def _scan_for_secrets(root: Path) -> list[str]:
    """Files containing a credential VALUE: a secret-shaped token, or the configured key.

    Findings name the file and what kind of match it was, never the matched text.
    """
    live_secrets = [value for name in _SECRET_ENV_VARS
                    if (value := os.environ.get(name)) and len(value) >= 12]
    hits: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".csv", ".yaml",
                                                     ".yml", ".md", ".txt", ".log"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        name = path.relative_to(root).as_posix()
        if any(secret in text for secret in live_secrets):
            hits.append(f"{name} contains the CONFIGURED API KEY")
            continue
        for pattern in _SECRET_VALUE_PATTERNS:
            if pattern.search(text):
                hits.append(f"{name} contains a credential-shaped token "
                            f"(/{pattern.pattern}/)")
                break
    return hits


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    raise SystemExit(main(Path(sys.argv[1]),
                          Path(sys.argv[2]) if len(sys.argv) > 2 else None))
