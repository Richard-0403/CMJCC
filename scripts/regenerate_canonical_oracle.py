"""Archive the current frozen canonical oracle, then derive the new one.

Why archive rather than overwrite
---------------------------------
The frozen reference is what every ranking metric is graded against, so replacing it in
place would silently redefine the yardstick and leave nothing to compare against. The
existing artifact is therefore copied to ``archive/canonical_oracle_<stem>.v<version>.json``
first, under the version IT declares, and only then is the canonical path rebuilt. The
archived copy is a tracked file, so "what did v3 say" stays answerable without digging
through git history.

Why it has to be a deliberate step
----------------------------------
``inputs_fingerprint`` covers the scenario file, the catalogue and
``CANONICAL_ORACLE_VERSION`` -- not the derivation code. The salary fix changed the labels
without touching a scenario or a job, so the fingerprint alone would still have matched and
the stale artifact would have been reused. Bumping the version invalidates it; this script is
how the invalidation is resolved, and it prints what moved so the change is reported rather
than absorbed.

Every grade-derived metric must be recomputed and re-reported afterwards: precision@k, nDCG,
the relevance grades and anything built on them are not comparable across versions.

Usage
-----
    python scripts/regenerate_canonical_oracle.py            # report only
    python scripts/regenerate_canonical_oracle.py --write    # archive and rebuild
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from jobrec.config import load_config  # noqa: E402
from jobrec_eval.oracle_reference import (  # noqa: E402
    CANONICAL_ORACLE_VERSION,
    build_canonical_references,
    frozen_artifact_path,
    inputs_fingerprint,
    load_frozen_references,
)

SCENARIOS = "evaluation/data/scenarios.jsonl"
CATALOG = "data/processed/jobs.jsonl"
CONFIG = "configs/experiment_full.yaml"


def _eligible_counts(references: dict) -> dict[str, int]:
    """Per scenario, how many catalogue jobs its reference marks eligible.

    The artifact holds each scenario's ``active_search`` and ``job_context``, not grades --
    grades are derived at analysis time from the reference, so they cannot be diffed here.
    Eligibility IS in the artifact and it is what the salary rule changed, so it is the part
    of the move that this script can report honestly.
    """
    out: dict[str, int] = {}
    for scenario_id, ref in sorted(references.items()):
        context = (ref.get("job_context") if isinstance(ref, dict) else None) or {}
        results = context.get("eligibility_results") or context.get("eligible_job_ids") or []
        if isinstance(results, list) and results and isinstance(results[0], dict):
            out[scenario_id] = sum(1 for r in results if r.get("eligible"))
        else:
            out[scenario_id] = len(results) if isinstance(results, list) else 0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="archive the existing artifact and rebuild the canonical one")
    parser.add_argument("--scenarios", default=SCENARIOS)
    parser.add_argument("--catalog", default=CATALOG)
    args = parser.parse_args()

    target = frozen_artifact_path(args.scenarios)
    expected = inputs_fingerprint(args.scenarios, args.catalog)
    existing = load_frozen_references(target)

    print(f"frozen artifact : {target}")
    print(f"code declares   : canonical oracle v{CANONICAL_ORACLE_VERSION}")
    print(f"expected fprint : {expected}")

    old_version = None
    if existing is not None:
        old_version = existing.provenance.get("canonical_oracle_version")
        print(f"on disk         : v{old_version}, fingerprint {existing.inputs_fingerprint}")
        if existing.inputs_fingerprint == expected:
            print("\nthe frozen artifact already matches the current inputs; nothing to do.")
            return 0
    else:
        print("on disk         : nothing")

    if not args.write:
        print("\nwould archive the existing artifact and rebuild. Re-run with --write.")
        return 0

    archived: Path | None = None
    if existing is not None:
        archived = target.parent / "archive" / f"{target.stem}.v{old_version}.json"
        archived.parent.mkdir(parents=True, exist_ok=True)
        if archived.exists():
            print(f"\nrefusing to overwrite an existing archive at {archived}")
            return 1
        shutil.copy2(target, archived)
        print(f"\narchived v{old_version} -> {archived}")
        before = _eligible_counts(existing.references)
        target.unlink()
    else:
        before = {}

    config = load_config(CONFIG, base_dir="configs")
    built = build_canonical_references(args.scenarios, args.catalog, config)
    target.write_text(
        json.dumps(built.as_artifact(), indent=2, sort_keys=True, default=str),
        encoding="utf-8")
    print(f"wrote v{CANONICAL_ORACLE_VERSION} -> {target}")

    after = _eligible_counts(built.references)
    print(f"\nscenarios       : {len(built.references)}")
    if before:
        moved = {s: (before.get(s, 0), after.get(s, 0))
                 for s in sorted(set(before) | set(after))
                 if before.get(s, 0) != after.get(s, 0)}
        print(f"eligible-count changes: {len(moved)} of {len(after)} scenarios")
        for scenario_id, (b, a) in moved.items():
            print(f"  {scenario_id}: {b} -> {a}  ({a - b:+d})")
        if not moved:
            print("  none -- the derivation moved without changing any eligible set")
    print("\nGrades are derived from this reference at ANALYSIS time, so the grade-level "
          "delta\nis not visible here. Every grade-derived metric must be recomputed and "
          "re-reported:\nprecision@k, nDCG, the relevance labels and anything built on them.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
