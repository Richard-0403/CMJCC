"""Print the LLM call diagnosis for one experiment directory.

The analysis itself lives in :mod:`jobrec.evaluation.llm_call_audit`, which the experiment
runner also calls to embed the summary in the experiment manifest. One implementation, because
two definitions of "logical call" would drift and the fallback rate a manifest states has to be
the same number this script reports.

See that module for what each denominator counts; they are not interchangeable.

Usage
-----
    python scripts/diagnose_llm_fallbacks.py <experiment_dir> [--json out.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from jobrec.evaluation.llm_call_audit import audit_llm_calls  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_dir")
    parser.add_argument("--json", default=None, help="also write the report to this path")
    args = parser.parse_args()

    report = audit_llm_calls(Path(args.experiment_dir))
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
