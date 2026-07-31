"""The quality gate, defined once and called by both `make gate-local` and CI.

Before this existed the two disagreed by construction. CI ran ruff over
``src tests scripts``, a mypy ratchet written inline in the workflow, and coverage with
``--fail-under=85``; the Makefile ran ruff over ``src tests`` only, ``mypy src || true``
which reports success whatever mypy says, and coverage with no threshold at all. A local
run could be green while the same tree failed CI, which makes a local gate worse than no
gate: it produces confidence rather than information.

Every check lives here. CI invokes ``--only <name>`` per job so it keeps its parallelism
while running this code, not a copy of it.

    python scripts/preflight.py                  # every check, stop at the first failure
    python scripts/preflight.py --list
    python scripts/preflight.py --only lint --only typecheck
    python scripts/preflight.py --keep-going     # run all, report a summary

Exit status is non-zero if any selected check fails.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

#: Interpreter used for every subprocess, so a check can never silently run against a
#: different environment than the one that launched the gate.
PY = sys.executable


@dataclass(frozen=True)
class Check:
    name: str
    why: str
    commands: tuple[tuple[str, ...], ...] = field(default_factory=tuple)


CHECKS: tuple[Check, ...] = (
    Check(
        "lint",
        "ruff over src, tests AND scripts -- scripts was missing from the Makefile target",
        ((PY, "-m", "ruff", "check", "src", "tests", "scripts"),),
    ),
    Check(
        "typecheck",
        "mypy with a non-increasing error ceiling; fails if mypy cannot complete",
        ((PY, str(REPO_ROOT / "scripts" / "mypy_ratchet.py")),),
    ),
    Check(
        "tests",
        "the deterministic suite plus coverage >= 85% over src/jobrec",
        (
            (PY, "-m", "coverage", "run", "-m", "pytest",
             "-m", "not postgres and not perf and not extraction_gate"),
            (PY, "-m", "coverage", "report", "--include=src/jobrec/*",
             "--fail-under=85"),
        ),
    ),
    Check(
        "data-quality",
        "catalog schema plus the R17 scenario data-quality checks",
        # The report goes under artifacts/data_quality/, which .gitignore already covers.
        # Writing it to artifacts/ directly left an untracked artifacts/data_quality_report.json
        # behind, so running the gate dirtied `git status` -- a gate that has to be cleaned
        # up after is a gate people stop running.
        ((PY, str(REPO_ROOT / "scripts" / "validate_catalog.py"),
          "--catalog", "data/processed/jobs.jsonl",
          "--scenarios", "evaluation/data/scenarios.jsonl",
          "--config", "configs/experiment_full.yaml",
          "--report-dir", "artifacts/data_quality/preflight"),),
    ),
    Check(
        "scenarios",
        "the authoritative scenario set is intact and the builder cannot overwrite it",
        # Deliberately the same test module the suite runs, not a second implementation:
        # a duplicate structural check here would be free to drift from the one that
        # actually guards the file. It costs a few seconds to run twice.
        ((PY, "-m", "pytest", "tests/contract/test_authoritative_scenarios.py",
          "-q", "--no-header", "-p", "no:cacheprovider"),),
    ),
    Check(
        "extraction-gate",
        "P0-1: all 42 scenarios match their declared reference, and the hidden "
        "paraphrase suite passes (RED until the extraction fix lands)",
        # Kept out of the `tests` check on purpose. Folding a known-red gate into the main
        # suite would mean every future regression arrives inside an already-failing run,
        # where nobody can tell the new breakage from the expected one.
        ((PY, "-m", "pytest",
          "tests/eval/test_reference_state_gate.py",
          "tests/unit/test_constraint_cue_paraphrases.py",
          "-q", "--no-header", "-p", "no:cacheprovider", "-m", "extraction_gate"),),
    ),
)

BY_NAME = {check.name: check for check in CHECKS}


def run_check(check: Check) -> tuple[bool, float]:
    started = time.monotonic()
    for command in check.commands:
        print(f"\n$ {' '.join(command)}", flush=True)
        result = subprocess.run(command, cwd=REPO_ROOT)
        if result.returncode != 0:
            return False, time.monotonic() - started
    return True, time.monotonic() - started


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--only", action="append", metavar="NAME",
                        help="run just this check (repeatable)")
    parser.add_argument("--list", action="store_true", help="list the checks and exit")
    parser.add_argument("--keep-going", action="store_true",
                        help="run every selected check even after one fails")
    args = parser.parse_args()

    if args.list:
        width = max(len(c.name) for c in CHECKS)
        for check in CHECKS:
            print(f"{check.name:<{width}}  {check.why}")
        return 0

    if args.only:
        unknown = [name for name in args.only if name not in BY_NAME]
        if unknown:
            print(f"unknown check(s): {', '.join(unknown)}. "
                  f"Known: {', '.join(BY_NAME)}", file=sys.stderr)
            return 2
        selected = [BY_NAME[name] for name in args.only]
    else:
        selected = list(CHECKS)

    results: list[tuple[str, bool, float]] = []
    for check in selected:
        print(f"\n{'=' * 72}\n{check.name}: {check.why}\n{'=' * 72}", flush=True)
        ok, seconds = run_check(check)
        results.append((check.name, ok, seconds))
        if not ok and not args.keep_going:
            break

    print(f"\n{'=' * 72}\ngate summary\n{'=' * 72}")
    for name, ok, seconds in results:
        print(f"  {'PASS' if ok else 'FAIL'}  {name:<14} {seconds:6.1f}s")
    skipped = [c.name for c in selected if c.name not in {r[0] for r in results}]
    for name in skipped:
        print(f"  SKIP  {name:<14}        (an earlier check failed)")

    failed = [name for name, ok, _ in results if not ok]
    if failed or skipped:
        print(f"\nGATE FAILED: {', '.join(failed) or 'incomplete'}")
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
