"""Copy the authoritative `reference` block from the main scenario file into a subset file.

A scenario must have ONE ground truth. ``evaluation/data/scenarios_subset.jsonl`` holds a
12-scenario slice of the same scenarios the 42-scenario set does, and both are graded by the
same oracle machinery, so a reference declared in one and absent (or different) in the other
would mean the same scenario had two different notions of what the candidate asked for.

Usage:
    python scripts/sync_scenario_references.py                # report
    python scripts/sync_scenario_references.py --write        # apply
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

MAIN = Path("evaluation/data/scenarios.jsonl")
SUBSETS = (Path("evaluation/data/scenarios_subset.jsonl"),)


def _load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--main", default=str(MAIN))
    parser.add_argument("--subset", action="append", default=None)
    args = parser.parse_args()

    source = {row["scenario_id"]: row.get("reference") for row in _load(Path(args.main))}
    targets = [Path(p) for p in (args.subset or [str(p) for p in SUBSETS])]
    exit_code = 0

    for path in targets:
        rows = _load(path)
        updated: list[dict] = []
        synced: list[str] = []
        orphans: list[str] = []
        drifted: list[str] = []
        for row in rows:
            scenario_id = row["scenario_id"]
            reference = source.get(scenario_id)
            if reference is None:
                # Not in the main set: it has its own ground truth to declare, and copying
                # nothing silently would leave it system-derived without saying so.
                orphans.append(scenario_id)
                updated.append(row)
                continue
            if row.get("reference") and row["reference"] != reference:
                drifted.append(scenario_id)
            if row.get("reference") != reference:
                synced.append(scenario_id)
            updated.append({**row, "reference": reference})

        print(f"{path}: {len(rows)} scenarios; synced {len(synced)}; "
              f"not in the main set: {orphans or 'none'}")
        if drifted:
            print(f"  OVERWROTE a differing reference on: {drifted}")
        if orphans:
            exit_code = 1
            print("  -> those scenarios must declare their own reference")
        if args.write and synced:
            path.write_text(
                "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in updated),
                encoding="utf-8")
            print(f"  wrote {path}")
    if not args.write:
        print("\n(report only; pass --write to apply)")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
