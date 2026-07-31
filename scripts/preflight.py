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
    #: Excluded from a bare run, still available via ``--only``. For a gate that pins a
    #: known defect scheduled for a later batch: it has to be runnable and named, but
    #: leaving it in the default set would make the gate permanently red, and a gate that
    #: is always red is one nobody reads -- at which point it stops reporting the
    #: regression it was added to catch.
    pending: bool = False


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
             "-m", "not postgres and not perf"),
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
        "paraphrase suite passes",
        # Also covered by `tests` now that both modules are unmarked and back in the
        # default suite. Kept as its own named check because it is the P0-1 acceptance
        # criterion: when it breaks, the signal should say "extraction no longer matches
        # the declared references", not "some test failed somewhere".
        ((PY, "-m", "pytest",
          "tests/eval/test_reference_state_gate.py",
          "tests/unit/test_constraint_cue_paraphrases.py",
          "-q", "--no-header", "-p", "no:cacheprovider"),),
    ),
    Check(
        "turn-state",
        "multi-turn strength persistence, explicit relaxation, and per-turn evidence "
        "provenance",
        ((PY, "-m", "pytest", "tests/eval/test_turn_state_persistence.py",
          "-q", "--no-header", "-p", "no:cacheprovider"),),
    ),
    Check(
        "oracle-freshness",
        "PRE-RE-RUN: the frozen canonical oracle still matches its inputs, so a batch "
        "cannot be spent only to fail in the analysis stage",
        ((PY, str(REPO_ROOT / "scripts" / "check_oracle_freshness.py")),),
        # Pending because it is red BY DESIGN right now: the salary fix bumped
        # CANONICAL_ORACLE_VERSION to 4.0.0 so the pre-fix frozen reference would be
        # rejected instead of silently reused. Rebuilding it changes every grade-derived
        # number, so it is an operator decision, not something a gate should do. Drop
        # `pending` once the reference has been rebuilt.
        pending=True,
    ),
)

BY_NAME = {check.name: check for check in CHECKS}
DEFAULT_CHECKS = tuple(check for check in CHECKS if not check.pending)


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
            flag = "  [pending, --only to run]" if check.pending else ""
            print(f"{check.name:<{width}}  {check.why}{flag}")
        return 0

    if args.only:
        unknown = [name for name in args.only if name not in BY_NAME]
        if unknown:
            print(f"unknown check(s): {', '.join(unknown)}. "
                  f"Known: {', '.join(BY_NAME)}", file=sys.stderr)
            return 2
        selected = [BY_NAME[name] for name in args.only]
    else:
        selected = list(DEFAULT_CHECKS)

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
