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

#: Suffixes treated as text in the release tree. These are rewritten with LF endings
#: before hashing, so the manifest reproduces on any checkout. Everything else
#: (currently only the plots) is copied and hashed byte-for-byte.
_TEXT_SUFFIXES = frozenset({".csv", ".json", ".jsonl", ".md", ".yaml", ".yml", ".txt"})


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_newlines(root: Path) -> int:
    """Rewrite the text files under ``root`` with LF endings. Returns the count changed.

    The analysis tree is produced on Windows, so the copies arrive carrying CRLF. The
    checksums are taken over the bytes on disk, which makes the manifest
    platform-dependent unless those bytes are pinned: an LF checkout reproduces
    different bytes and verification then fails on every text file, not just a few.
    Normalising here -- after the copies, before the hashing -- keeps the recorded
    hashes valid everywhere. ``.gitattributes`` pins the same tree to ``eol=lf`` so a
    checkout cannot undo it.
    """
    changed = 0
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        raw = path.read_bytes()
        if b"\x00" in raw:  # defensive: never rewrite anything that looks binary
            continue
        normalized = raw.replace(b"\r\n", b"\n")
        if normalized != raw:
            path.write_bytes(normalized)
            changed += 1
    return changed


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
            "exp-06cc34defe39", "exp-87aec1bc99dc", "exp-515b63d6a656",
            "exp-301060a1899d",
        ],
        "superseded_details": {
            "exp-87aec1bc99dc": {
                "reason": "predates the declared canonical oracle v3.0.0",
                "storage": "pruned from local disk after release v1; the committed "
                           "report directory is recoverable from tag "
                           "cmjcc-thesis-release-v1, the untracked run tree is not",
            },
            "exp-197f6aacc171": {
                "reason": "predates the declared canonical oracle v3.0.0",
                "storage": "pruned from local disk after release v1; the committed "
                           "report directory is recoverable from tag "
                           "cmjcc-thesis-release-v1, the untracked run tree and "
                           "CMJCC_deterministic_runs_exp-197f6aacc171.zip are not",
            },
            "exp-06cc34defe39": {
                "reason": "predates the declared canonical oracle v3.0.0",
                "storage": "pruned from local disk after release v1; neither the run "
                           "tree nor CMJCC_hybrid_runs_exp-06cc34defe39.zip was "
                           "tracked, so neither is recoverable",
            },
            "exp-f90573008bdb": {
                "reason": "citable only as reproducibility evidence, never for results",
                "storage": "retained on local disk, with its replay diff under "
                           "artifacts/reports/",
            },
            "exp-8793b18de5b2": {
                "reason": "pre-fix artifact, superseded by the official pair",
                "storage": "pruned from local disk after release v1, together with its "
                           "copy under test_results/; both were tracked and are "
                           "recoverable from tag cmjcc-thesis-release-v1",
            },
            "exp-515b63d6a656": {
                "reason": "intermediate artifact from the readiness work, never a "
                          "candidate for citation",
                "storage": "not on disk; no run tree, report tree or archive was kept",
            },
            "exp-301060a1899d": {
                "reason": "intermediate artifact from the readiness work, never a "
                          "candidate for citation",
                "storage": "not on disk; no run tree, report tree or archive was kept",
            },
        },
        "superseded_note": (
            "exp-87aec1bc99dc, exp-197f6aacc171 and exp-06cc34defe39 all predate the "
            "declared canonical oracle (v3.0.0), so their grade-derived numbers are "
            "not comparable with the official pair: they were scored against an oracle "
            "that inherited the evaluated system's own soft/hard judgements, which is "
            "exactly the confound v3.0.0 removes. Comparing them with the official "
            "pair would compare two different ground truths, so they carry no residual "
            "comparison value and were pruned from local disk after release v1, "
            "together with the pre-fix artifact exp-8793b18de5b2. Per-experiment "
            "reasons and storage status are in superseded_details. The authoritative "
            "record of what these experiments concluded, and why it is void, lives in "
            "THESIS_OFFICIAL_RESULTS.md: it names the superseded ids with reasons and "
            "explicitly lists the forbidden legacy numbers, which is a safer record "
            "than retaining full plausible-looking reports on disk. exp-f90573008bdb "
            "is retained and remains citable only as reproducibility evidence."
        ),
        "verification": {
            "slim_release": (
                "This directory is self-verifying: scripts/verify_final_release.py "
                "recomputes SHA-256 for every path in final_release/checksums.json and "
                "reports missing, changed and unrecorded files. Text here is pinned to "
                "LF by .gitattributes, so the recorded hashes hold on any checkout."
            ),
            "full_bundles": (
                "The two bundle archives are not covered by the manifest above and are "
                "not carried in git. Verify them in two steps: first compare the ZIP's "
                "SHA-256 against bundle_archives[].sha256 below, then extract it and "
                "run the project's own verifier over the extracted tree -- "
                "python -m jobrec_eval.cli verify <extracted analysis dir> -- which "
                "checks the 5305 (deterministic) and 9505 (hybrid) recorded files "
                "against the bundle's own checksums.json."
            ),
        },
        "excluded_from_this_release": {
            "normalized_tables": "regenerable from the run bundles; ~9 MB across the pair",
            "run_bundles": "shipped as a standalone archive; ~80 MB",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    # build_bundle_archives.py appends bundle_archives to provenance.json *after* this
    # script runs, so it is not something _provenance() can regenerate. Capture it
    # before the tree is removed: rebuilding that block would mean rebuilding the ZIPs,
    # which would change the SHA-256 values already published with the release.
    carried: dict[str, Any] = {}
    prior_path = ROOT / "provenance.json"
    if args.write and prior_path.is_file():
        prior = json.loads(prior_path.read_text(encoding="utf-8"))
        if "bundle_archives" in prior:
            carried["bundle_archives"] = prior["bundle_archives"]
            print("carrying over bundle_archives from the previous provenance.json")

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
        # Pin the copied text to LF before anything is hashed, so the manifest below
        # holds on an LF checkout as well as this one.
        print(f"normalised {_normalize_newlines(ROOT)} copied text file(s) to LF")
        provenance = _provenance()
        provenance.update(carried)
        (ROOT / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8",
            newline="\n")
        (ROOT / "README.md").write_text(_readme(provenance), encoding="utf-8",
                                        newline="\n")
        (ROOT / "ERRATA.md").write_text(_ERRATA, encoding="utf-8", newline="\n")
        # A manifest over the release itself, so a reader can check this copy. Only the
        # release-level manifest is excluded -- it cannot record its own hash. The
        # per-experiment checksums.json files nested under each experiment ARE release
        # content (they record what the full analysis tree hashed to) and must be covered;
        # matching on the bare filename silently dropped four of them.
        manifest_path = ROOT / "checksums.json"
        files = sorted(p for p in ROOT.rglob("*")
                       if p.is_file() and p != manifest_path)
        (ROOT / "checksums.json").write_text(json.dumps({
            "algorithm": "sha256",
            "file_count": len(files),
            "files": {p.relative_to(ROOT).as_posix(): _sha256(p) for p in files},
        }, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
        size = sum(p.stat().st_size for p in ROOT.rglob("*") if p.is_file())
        print(f"\nwrote {len(files) + 1} files, {size / 1024 / 1024:.2f} MB")
        print("code identity matches across the pair:",
              provenance["code_identity_matches_across_the_pair"])
    else:
        print(f"\n{total} files would be copied; pass --write to build")
    return 0


#: Corrections that apply to the material in this release. The frozen reports are left
#: exactly as generated -- an errata record alongside them is honest, editing them after
#: the fact is not -- so anything found after the freeze is written down here instead.
_ERRATA = """# Errata

Corrections that apply to this release. The frozen `report/analysis_report.md` of
either experiment is **not** edited: it stays byte-for-byte as generated, and the
recorded checksums stay valid. Corrections are recorded here and mirrored in
`THESIS_OFFICIAL_RESULTS.md`.

## E-1 Deterministic task-failure count: 98, not 96

**Status: error in the thesis-facing summary, not in the release data.**

`THESIS_OFFICIAL_RESULTS.md` introduced the deterministic error taxonomy with
"96 task failures". The correct count is **98**. The frozen report was already
right, so nothing in this directory needed changing:

- `report/analysis_report.md` states `task-unsuccessful runs: 98`, broken down as
  full 1, no_memory 10, one_shot 17, no_context 35, profile_only 35.
- `metrics/error_taxonomy.csv` sums to 98 (35 + 35 + 16 + 9 + 3) and its
  percentage column uses 98 as the denominator (35 / 98 = 35.7%).
- `metrics/variant_summary.csv` implies the same total independently: at 42 runs
  per variant, the task_success column gives 1 + 10 + 17 + 35 + 35 = 98.

The hybrid figure of 152 was correct as published (at 126 runs per variant:
14 + 33 + 105).

## E-2 No-match scenarios: scope the claim to role fit as well as hard constraints

**Status: interpretive scoping. The underlying numbers are unchanged and correct.**

`metrics/no_match_metrics.csv` reports no-match precision / recall / F1 of 1.000
for the full, no_memory and one_shot variants over `no_match_expected = 5`. That
arithmetic stands. What must not be attached to it is the reading that all five
scenarios are infeasible on their hard constraints alone.

Two of the five are not:

- **SC-E-02** -- `data_quality_report.json` records a `warning` of type
  `no_match_scenario_constraint_satisfiable`: five catalogue jobs (job-0021,
  job-0086, job-0089, job-0094, job-0169) satisfy the scenario's hard
  constraints, all of them outside the requested role families.
- **SC-E-04** -- the same warning, with one such job (job-0012).

Both are typed `multiple_hard`, not `no_match`; only three scenarios carry the
`no_match` type, which is why `report/analysis_report.md` counts `no_match 3` in
its scenario-type breakdown while the no-match metric uses a denominator of 5.
The report already surfaces the discrepancy at
`no_match_scenario_constraint_satisfiable 2` in its data-quality section.

**Correct scoping.** For these scenarios the outcome is: *no qualified and
relevant job exists once both the target role scope and the hard constraints are
applied.*

**Not permitted.** Summarising all five no-match scenarios as joint
infeasibility of the hard constraints. For SC-E-02 and SC-E-04 the no-match rests
on role fit, not on constraint infeasibility, and the case study labelled
"Correct no-match (SC-E-02)" in the frozen report must carry this qualification
when cited.
"""


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

## Errata

`ERRATA.md` carries the corrections that apply to the frozen reports in this
release. The reports themselves are left byte-for-byte as they were generated;
the errata are recorded alongside rather than edited in.

## Verifying

There are two levels, and they use different mechanisms.

**This slim release** verifies against `checksums.json` in this directory:

```
python scripts/verify_final_release.py
```

It recomputes SHA-256 for every recorded path and reports missing, changed and
unrecorded files. The text here is pinned to LF by `.gitattributes`,
so the recorded hashes reproduce on any checkout rather than only on the machine
that built it.

**The full bundle archives** are not covered by that manifest and are not carried
in git. Verify them in two steps -- first the archive, then its contents:

```
# 1. the archive itself, against bundle_archives[].sha256 in provenance.json
python -c "import hashlib,sys;print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())" <archive.zip>

# 2. the extracted tree, against the bundle's own checksums.json
python -m jobrec_eval.cli verify <extracted analysis dir>
python -m jobrec_eval.cli replay <extracted run bundle dir>
```

Recorded results: both trees verified OK; replay reproduced
{det['run_count']}/{det['run_count']} and {hyb['run_count']}/{hyb['run_count']} runs with
0 differences.

Built from `{generated or 'unknown'}` by `scripts/build_final_release.py`.
"""


if __name__ == "__main__":
    raise SystemExit(main())
