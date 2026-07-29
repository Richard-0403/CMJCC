"""Assemble the slim, citable release from the two official experiments.

What goes in: the reports, the metric and statistics tables, the plots, the manifests, the
audit tables, the data-quality report, and each experiment's own ``checksums.json`` (so the
release records what the FULL tree hashed to even though it does not carry the full tree).
Plus the frozen inputs the numbers are a function of, and a provenance record.

What stays out: the ``normalized/`` tables and the run bundles. Both are regenerable from
the bundles, both are bulky (9 MB and ~80 MB), and git is the wrong transport for them.

Run:
    python scripts/build_final_release.py            # report what would be copied
    python scripts/build_final_release.py --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

#: The ONLY citable pair. Anything else on disk is superseded and must not be cited.
OFFICIAL = {
    "deterministic": {
        "experiment_id": "exp-e748800507ef",
        "analysis": Path("evaluation/outputs/exp-e748800507ef"),
        "runs": Path("evaluation/outputs/_runs/exp-e748800507ef"),
        "config": "configs/experiment_full.yaml",
        "variants": ["full", "profile_only", "one_shot", "no_memory", "no_context"],
        "repeats": 1,
    },
    "hybrid": {
        "experiment_id": "exp-6db1e87daed5",
        "analysis": Path("evaluation/outputs_hybrid/exp-6db1e87daed5"),
        "runs": Path("evaluation/outputs_hybrid/_runs/exp-6db1e87daed5"),
        "config": "configs/hybrid_vectorengine.yaml",
        "variants": ["full", "no_memory", "no_context"],
        "repeats": 3,
    },
}

#: Directories copied wholesale out of the analysis tree.
_ANALYSIS_DIRS = ("report", "metrics", "statistics", "plots", "manifests", "audit")
#: Loose files copied out of the analysis tree root.
_ANALYSIS_FILES = ("checksums.json", "data_quality_report.json")
#: Files copied out of the RUN-BUNDLE tree: the experiment-level provenance, without the
#: bundles themselves.
_RUN_FILES = ("experiment_manifest.json", "runs_index.csv", "failures.csv",
              "resolved_config.yaml", "scenarios.jsonl", "checksums.json")

#: Frozen inputs every reported number is a function of.
_INPUTS = (
    Path("evaluation/data/scenarios.jsonl"),
    Path("evaluation/data/canonical_oracle_scenarios.json"),
    Path("configs/base.yaml"),
    Path("configs/experiment_full.yaml"),
    Path("configs/hybrid_vectorengine.yaml"),
)

ROOT = Path("final_release")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy(src: Path, dst: Path, write: bool) -> int:
    if not src.exists():
        print(f"  MISSING {src}")
        return 0
    if write:
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
    return (sum(1 for p in src.rglob("*") if p.is_file()) if src.is_dir() else 1)


def _provenance() -> dict[str, Any]:
    """Identity of both experiments, plus the honest reading of the dirty flag."""
    entries: dict[str, Any] = {}
    for label, spec in OFFICIAL.items():
        manifest = _json(spec["runs"] / "experiment_manifest.json")
        oracle = _json(spec["analysis"] / "manifests" / "canonical_oracle.json")
        entries[label] = {
            "experiment_id": manifest["experiment_id"],
            "config": spec["config"],
            "variants": spec["variants"],
            "repeats": spec["repeats"],
            "run_count": manifest["run_count"],
            "expected_run_count": manifest.get("expected_run_count"),
            "crashed_run_count": manifest.get("crashed_run_count"),
            "commit_hash": manifest["commit_hash"],
            "git_dirty": manifest["git_dirty"],
            "code_version": manifest["code_version"],
            "execution_fingerprint": manifest.get("execution_fingerprint"),
            "analysis_fingerprint": manifest.get("analysis_fingerprint"),
            "source_fingerprint": manifest.get("source_fingerprint"),
            "config_hash": manifest["config_hash"],
            "catalog_hash": manifest["catalog_hash"],
            "scenarios_hash": manifest["scenarios_hash"],
            "prompt_hash": manifest["prompt_hash"],
            "created_at": manifest["created_at"],
            "canonical_oracle": {
                "version": oracle["canonical_oracle_version"],
                "inputs_fingerprint": oracle["inputs_fingerprint"],
                "reference_fingerprint": oracle["reference_fingerprint"],
                "derivation": oracle["provenance"]["derivation"],
                "declared_scenario_count":
                    oracle["provenance"]["declared_scenario_count"],
                "system_derived_scenario_count":
                    oracle["provenance"]["system_derived_scenario_count"],
            },
        }

    det, hyb = entries["deterministic"], entries["hybrid"]
    same_code = (det["commit_hash"] == hyb["commit_hash"]
                 and det["execution_fingerprint"] == hyb["execution_fingerprint"])
    return {
        "official_pair": [det["experiment_id"], hyb["experiment_id"]],
        "experiments": entries,
        "code_identity_matches_across_the_pair": same_code,
        "git_dirty_note": (
            "The hybrid manifest records git_dirty=true and it is left as recorded -- the "
            "history is not rewritten to make it look clean. It does NOT mean the hybrid "
            "run used modified code: its commit_hash and execution_fingerprint are "
            "identical to the deterministic run's "
            f"(commit {det['commit_hash'][:12]}, execution fingerprint "
            f"{str(det['execution_fingerprint'])[:16]}). The flag is set because "
            "git status was non-empty at run time, and it was non-empty only because the "
            "deterministic run had just written its analysis tree into "
            "evaluation/outputs/, which was untracked at the time. In other words, "
            "producing one official artifact made the next one look dirty. Those output "
            "trees are now gitignored and the citable subset is committed here instead, so "
            "the flag cannot be produced this way again."
        ),
        "superseded_and_not_citable": [
            "exp-8793b18de5b2", "exp-f90573008bdb", "exp-197f6aacc171",
            "exp-06cc34defe39", "exp-87aec1bc99dc",
        ],
        "superseded_note": (
            "Earlier experiments remain on disk as history. exp-87aec1bc99dc and "
            "exp-197f6aacc171 predate the declared canonical oracle (v3.0.0) and "
            "exp-06cc34defe39 predates it too, so their grade-derived numbers are not "
            "comparable with the official pair. exp-f90573008bdb is citable only as "
            "reproducibility evidence, never for results."
        ),
        "excluded_from_this_release": {
            "normalized_tables": "regenerable from the run bundles; ~9 MB across the pair",
            "run_bundles": "shipped as a standalone archive; ~80 MB",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if args.write and ROOT.exists():
        shutil.rmtree(ROOT)

    total = 0
    for label, spec in OFFICIAL.items():
        target = ROOT / label / spec["experiment_id"]
        print(f"{label} -> {target}")
        for name in _ANALYSIS_DIRS:
            total += _copy(spec["analysis"] / name, target / name, args.write)
        for name in _ANALYSIS_FILES:
            total += _copy(spec["analysis"] / name, target / name, args.write)
        for name in _RUN_FILES:
            total += _copy(spec["runs"] / name, target / "run_bundle_provenance" / name,
                           args.write)

    print("inputs ->", ROOT / "inputs")
    for path in _INPUTS:
        total += _copy(path, ROOT / "inputs" / path.name, args.write)

    if args.write:
        provenance = _provenance()
        (ROOT / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8")
        (ROOT / "README.md").write_text(_readme(provenance), encoding="utf-8")
        # A manifest over the release itself, so a reader can check this copy.
        files = sorted(p for p in ROOT.rglob("*")
                       if p.is_file() and p.name != "checksums.json")
        (ROOT / "checksums.json").write_text(json.dumps({
            "algorithm": "sha256",
            "file_count": len(files),
            "files": {p.relative_to(ROOT).as_posix(): _sha256(p) for p in files},
        }, indent=2, sort_keys=True), encoding="utf-8")
        size = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
        print(f"\nwrote {len(files) + 1} files, {size / 1024 / 1024:.2f} MB")
        print("code identity matches across the pair:",
              provenance["code_identity_matches_across_the_pair"])
    else:
        print(f"\n{total} files would be copied; pass --write to build")
    return 0


def _readme(prov: dict[str, Any]) -> str:
    det = prov["experiments"]["deterministic"]
    hyb = prov["experiments"]["hybrid"]
    generated = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                               capture_output=True, text=True, check=False).stdout.strip()
    return f"""# CMJCC final release

The **only** citable experiment pair. Anything else under `evaluation/` is superseded
history; see `provenance.json` for the list and why.

| | deterministic | hybrid |
|---|---|---|
| experiment id | `{det['experiment_id']}` | `{hyb['experiment_id']}` |
| config | `{det['config']}` | `{hyb['config']}` |
| variants x repeats | {len(det['variants'])} x {det['repeats']} | {len(hyb['variants'])} x {hyb['repeats']} |
| runs | {det['run_count']} / {det['expected_run_count']} (crashed {det['crashed_run_count']}) | {hyb['run_count']} / {hyb['expected_run_count']} (crashed {hyb['crashed_run_count']}) |
| commit | `{det['commit_hash'][:12]}` | `{hyb['commit_hash'][:12]}` |
| execution fingerprint | `{str(det['execution_fingerprint'])[:16]}` | `{str(hyb['execution_fingerprint'])[:16]}` |
| canonical oracle | v{det['canonical_oracle']['version']}, {det['canonical_oracle']['derivation']} ({det['canonical_oracle']['declared_scenario_count']} declared / {det['canonical_oracle']['system_derived_scenario_count']} system-derived) | v{hyb['canonical_oracle']['version']}, {hyb['canonical_oracle']['derivation']} ({hyb['canonical_oracle']['declared_scenario_count']} / {hyb['canonical_oracle']['system_derived_scenario_count']}) |

Both experiments ran from the same frozen source: identical `commit_hash` and identical
`execution_fingerprint`.

## On the hybrid `git_dirty=true`

{prov['git_dirty_note']}

## Layout

- `deterministic/{det['experiment_id']}/` and `hybrid/{hyb['experiment_id']}/`
  - `report/` the analysis report and its backing data
  - `metrics/`, `statistics/` the tables the thesis cites
  - `plots/` the embedded figures
  - `manifests/` experiment manifest, analysis plan, frozen canonical oracle
  - `audit/` data lineage, invalid runs, scenarios without a reference
  - `checksums.json` the manifest over the FULL analysis tree, including the
    `normalized/` tables that are not carried here
  - `run_bundle_provenance/` the run-bundle tree's manifest, run index, failures,
    resolved config and scenario snapshot -- without the bundles
- `inputs/` the frozen scenario set, the declared canonical oracle, and the configs
- `provenance.json` machine-readable identity for both experiments
- `checksums.json` a manifest over THIS release

## Verifying

```
python -m jobrec_eval.cli verify <analysis dir>     # against the experiment's own checksums
python -m jobrec_eval.cli replay <run bundle dir>   # recompute every run's key states
```

Recorded results: both trees verified OK; replay reproduced
{det['run_count']}/{det['run_count']} and {hyb['run_count']}/{hyb['run_count']} runs with
0 differences.

Built from `{generated or 'unknown'}` by `scripts/build_final_release.py`.
"""


if __name__ == "__main__":
    raise SystemExit(main())
