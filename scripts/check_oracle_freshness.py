"""Fail fast when the frozen canonical oracle no longer matches its inputs.

Why this runs BEFORE an experiment, not during the analysis
-----------------------------------------------------------
``load_or_build_canonical_references`` already refuses a stale frozen artifact -- but it
is called from the ANALYSIS stage, which happens after every run has been executed. On the
official batches that is 210 or 378 runs, and on the hybrid arm those are paid LLM calls
that take hours. Discovering there that the reference has to be rebuilt, and that every
grade-derived number must be re-reported, means the batch was spent before the problem was
visible. This check moves that discovery to before the first run.

It is deliberately NOT wired into the default gate. The salary fix bumped
``CANONICAL_ORACLE_VERSION`` to 4.0.0 precisely so the pre-fix artifact would be rejected,
so this check is red until the reference is rebuilt -- and a gate that is permanently red
is one nobody reads. It is registered as a ``pending`` preflight check instead: named,
runnable with ``--only oracle-freshness``, and required before a re-run.

The check never writes anything. Rebuilding the reference changes the labels every ranking
metric is computed against, so it stays an explicit, operator-initiated step.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from jobrec_eval.oracle_reference import (  # noqa: E402
    CANONICAL_ORACLE_VERSION,
    frozen_artifact_path,
    inputs_fingerprint,
    load_frozen_references,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scenarios", default="evaluation/data/scenarios.jsonl")
    parser.add_argument("--catalog", default="data/processed/jobs.jsonl")
    args = parser.parse_args()

    target = frozen_artifact_path(args.scenarios)
    print(f"scenarios : {args.scenarios}")
    print(f"catalog   : {args.catalog}")
    print(f"reference : {target}")
    print(f"derivation: canonical oracle v{CANONICAL_ORACLE_VERSION}")

    frozen = load_frozen_references(target)
    if frozen is None:
        # Absent is not stale: the analysis derives and freezes it on first use, so there
        # is nothing here that could silently grade against the wrong labels.
        print("\nno frozen reference on disk; it will be derived and frozen on first use.")
        print("OK")
        return 0

    expected = inputs_fingerprint(args.scenarios, args.catalog)
    recorded_version = frozen.provenance.get("canonical_oracle_version")
    print(f"recorded  : version {recorded_version}, fingerprint {frozen.inputs_fingerprint}")
    print(f"expected  : version {CANONICAL_ORACLE_VERSION}, fingerprint {expected}")

    if frozen.inputs_fingerprint == expected:
        print(f"\nthe frozen reference matches the current inputs "
              f"({len(frozen.references)} scenarios).")
        print("OK")
        return 0

    reason = ("the DERIVATION changed" if recorded_version != CANONICAL_ORACLE_VERSION
              else "the scenario file or the catalog changed")
    print(f"\nSTALE: {reason}, so the frozen reference no longer describes these inputs.")
    print("Running the experiment now would spend the whole batch and then fail in the")
    print("analysis stage, because a stale reference is refused rather than reused.")
    print("\nTo proceed:")
    print(f"  1. delete {target}")
    print("  2. re-derive it (the analysis stage freezes it on first use)")
    print("  3. re-report every grade-derived metric: the labels change, so")
    print("     precision@k, nDCG, the relevance grades and anything computed from them")
    print("     are not comparable with previously published numbers")
    print("\nThe sealed release keeps its own checksum-verified copy under")
    print("final_release/inputs/, which this step does not touch.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
