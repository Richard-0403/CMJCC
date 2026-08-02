"""Assemble the citable release for the 588-run main experiment (release v2).

Why a second script rather than parameters on ``build_final_release.py``
-----------------------------------------------------------------------
That script is not merely hardcoded to the v1 experiment ids -- its NARRATIVE is v1's. Its
``git_dirty_note`` explains why the v1 hybrid manifest recorded a dirty tree (it does not
apply here: both v2 manifests record ``git_dirty: false``), its errata correct v1's numbers,
and its superseded list predates the v1 pair itself. Rewriting those in place would also put
the published v1 checksums at risk, and those are already committed as the record of a release
someone may be citing. So the copy/normalise/hash machinery is IMPORTED from it -- one
implementation of the tricky part -- and only the release definition and the narrative are new.

What supersedes what
--------------------
Release v1 (``final_release/``, code 0.1.0, canonical oracle 1.0.0) is superseded by this one
and is deliberately left in place: an earlier release is a record, not a mistake to erase. Its
grade-derived numbers are NOT comparable with these, because the canonical oracle moved 1.0.0 ->
4.0.0, which is the same reason v1's own provenance voids seven experiments before it.

What goes in: the reports, metric and statistics tables, plots, manifests, audit tables, the
data-quality report, each experiment's own ``checksums.json``, the audit evidence that the batch
was verified (replay diff, fallback diagnosis, provenance audit), the human annotation labels
with their pre-registered frame and adjudication record, the frozen inputs, and a provenance
record. What stays out: ``normalized/`` tables and the run bundles, both regenerable from the
bundles and both bulky -- the same rule v1 applied.

Run:
    python scripts/build_release_v2.py            # report what would be copied
    python scripts/build_release_v2.py --write
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "src"))

from build_final_release import (  # noqa: E402
    _ANALYSIS_DIRS,
    _ANALYSIS_FILES,
    _INPUTS,
    _RUN_FILES,
    _copy,
    _normalize_newlines,
    _sha256,
)

ROOT = Path("final_release_v2")

#: The two official arms of the 588-run experiment. ``analysis`` points at the tree built with
#: the HUMAN labels available, which is a superset of the oracle-only tree: it carries both
#: sources side by side in ``metrics/relevance_source_comparison.csv`` and, unlike the
#: oracle-only run, its human ranking columns are populated rather than withheld.
OFFICIAL: dict[str, dict[str, Any]] = {
    "deterministic": {
        "experiment_id": "exp-40a9cd647575",
        "analysis": Path("artifacts/main_deterministic/analysis_human/exp-40a9cd647575"),
        "runs": Path("artifacts/main_deterministic/runs/exp-40a9cd647575"),
        "audit": Path("artifacts/main_deterministic"),
        "annotation": Path("artifacts/annotation_official/deterministic"),
        "adjudication": Path("artifacts/annotation_adjudication/deterministic"),
        "config": "configs/experiment_full.yaml",
        "variants": ["full", "profile_only", "one_shot", "no_memory", "no_context"],
        "repeats": 1,
        "concurrency": 1,
    },
    "hybrid": {
        "experiment_id": "exp-2b33b808a0f8",
        "analysis": Path("artifacts/main_hybrid/analysis_human/exp-2b33b808a0f8"),
        "runs": Path("artifacts/main_hybrid/runs/exp-2b33b808a0f8"),
        "audit": Path("artifacts/main_hybrid"),
        "annotation": Path("artifacts/annotation_official/hybrid"),
        "adjudication": Path("artifacts/annotation_adjudication/hybrid"),
        "config": "configs/hybrid_vectorengine.yaml",
        "variants": ["full", "no_memory", "no_context"],
        "repeats": 3,
        "concurrency": 20,
    },
}

#: The serial-latency sub-experiment. Reported SEPARATELY and never merged into the Hybrid arm:
#: it exists only so a single-request latency figure can be quoted without the concurrent
#: batch's contention, and it is one variant over one repeat, so it is not a second Hybrid arm.
LATENCY_SUB = {
    "experiment_id": "exp-e63f05ad75bb",
    "analysis": Path("artifacts/latency_serial/analysis/exp-e63f05ad75bb"),
    "runs": Path("artifacts/latency_serial/runs/exp-e63f05ad75bb"),
    "config": "configs/hybrid_vectorengine.yaml",
    "variants": ["full"],
    "repeats": 1,
    "concurrency": 1,
}

#: Audit evidence copied per arm. Absent files are skipped (the deterministic arm makes no
#: model calls, so it has no fallback diagnosis).
_AUDIT_FILES = ("replay_diff.json", "fallback_diagnosis.json")

#: Human annotation artifacts. The export CSVs are what the analysis consumes; the JSONL dump is
#: the archive that makes the human pass reproducible; the universe is the pre-registered
#: coverage denominator.
_ANNOTATION_FILES = (
    ("export/relevance_labels_human.csv", "relevance_labels_human.csv"),
    ("export/claim_annotations_human.csv", "claim_annotations_human.csv"),
    ("release/annotation_manifest.json", "annotation_manifest.json"),
    ("release/human_annotations.jsonl", "human_annotations.jsonl"),
    ("annotation_universe.json", "annotation_universe.json"),
)


def _json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _provenance() -> dict[str, Any]:
    entries: dict[str, Any] = {}
    for label, spec in {**OFFICIAL, "latency_serial_subexperiment": LATENCY_SUB}.items():
        manifest = _json(spec["runs"] / "experiment_manifest.json")
        oracle_path = spec["analysis"] / "manifests" / "canonical_oracle.json"
        oracle = _json(oracle_path) if oracle_path.is_file() else {}
        entry = {
            "experiment_id": manifest["experiment_id"],
            "config": spec["config"],
            "variants": spec["variants"],
            "repeats": spec["repeats"],
            "concurrency": manifest.get("concurrency"),
            "run_count": manifest["run_count"],
            "expected_run_count": manifest.get("expected_run_count"),
            "crashed_run_count": manifest.get("crashed_run_count"),
            "commit_hash": manifest["commit_hash"],
            "git_dirty": manifest["git_dirty"],
            "code_version": manifest["code_version"],
            "execution_fingerprint": manifest.get("execution_fingerprint"),
            "source_fingerprint": manifest.get("source_fingerprint"),
            "config_hash": manifest["config_hash"],
            "catalog_hash": manifest["catalog_hash"],
            "scenarios_hash": manifest["scenarios_hash"],
            "prompt_hash": manifest["prompt_hash"],
            "retry_policy": manifest.get("retry_policy"),
            "llm_call_summary": manifest.get("llm_call_summary") or None,
            "created_at": manifest["created_at"],
        }
        if oracle:
            entry["canonical_oracle"] = {
                "version": oracle["canonical_oracle_version"],
                "inputs_fingerprint": oracle["inputs_fingerprint"],
                "reference_fingerprint": oracle["reference_fingerprint"],
                "derivation": oracle["provenance"]["derivation"],
            }
        entries[label] = entry

    det, hyb = entries["deterministic"], entries["hybrid"]
    return {
        "release": "v2",
        "official_pair": [det["experiment_id"], hyb["experiment_id"]],
        "total_runs": det["run_count"] + hyb["run_count"],
        "experiments": entries,
        "code_identity_matches_across_the_pair": (
            det["commit_hash"] == hyb["commit_hash"]
            and det["execution_fingerprint"] == hyb["execution_fingerprint"]
            and det["source_fingerprint"] == hyb["source_fingerprint"]),
        "supersedes": {
            "release": "v1 (final_release/)",
            "pair": ["exp-e748800507ef", "exp-6db1e87daed5"],
            "reason": (
                "v1 was produced by code 0.1.0 against canonical oracle 1.0.0. This release is "
                "code 0.2.0 against canonical oracle 4.0.0, so every grade-derived number "
                "(NDCG@5, Precision@5, mean graded relevance, retrieval recall) is measured "
                "against a different ground truth and is NOT comparable with v1's. v1 is left "
                "in place because an earlier release is a record, not a mistake to erase; it "
                "must not be cited alongside these numbers as if the two were one series."),
        },
        "known_limitations": {
            "fallback_concentration_by_variant": (
                "The Hybrid arm's 7 affected runs are not spread evenly across variants: "
                "no_context 6/126 (4.76%), full 1/126 (0.79%), no_memory 0/126. This is an "
                "EXECUTION-ORDER artifact, not a property of the variant. The run plan is "
                "variant-major, so the variants occupy distinct windows of the 30-minute batch "
                "(full at plan positions 0-169, no_memory 115-288, no_context 240-377), and the "
                "endpoint degraded monotonically across it -- failed attempts per sixth of the "
                "batch were 2, 14, 38, 53, 88, 70. All six affected no_context runs sit at "
                "positions 280-317, inside the worst window. Measured impact on the "
                "conclusions is negligible: removing the 7 runs moves no_context's ndcg_at_5 by "
                "+0.0065, task_success by -0.0107, hcsr by +0.0038 and precision_at_5 by "
                "+0.0068 -- all under 0.011, and in BOTH directions, which is noise rather than "
                "systematic degradation. The deterministic arm corroborates independently: it "
                "makes no model calls at all, so it cannot carry an endpoint confound, and it "
                "puts no_context at ndcg 0.6417 / task_success 0.1190 against Hybrid's 0.6275 / "
                "0.1190. A future batch should interleave the plan so drift spreads across "
                "variants instead of loading onto whichever ran last."),
            "latency_is_wall_clock_and_the_hybrid_arm_was_concurrent": (
                "The Hybrid arm ran at concurrency 20, so its component_latency and latency "
                "percentiles are contended and must NOT be read as single-request latency. The "
                "serial sub-experiment (exp-e63f05ad75bb, concurrency 1) exists for that "
                "figure. Measured: per-successful-attempt latency is median 23.4s serial vs "
                "22.5s at concurrency 20, i.e. the endpoint's own processing dominates and "
                "concurrency did not measurably inflate it -- but per-RUN total_latency_ms does "
                "differ, because it accumulates retries. Both manifests record the concurrency "
                "used, which is what makes either number readable."),
            "validator_vs_human_kappa_is_near_zero_by_construction": (
                "validator_vs_human_kappa is 0.0 (deterministic) and -0.014 (hybrid). This is "
                "the kappa paradox, not disagreement: the validator delivers only what it "
                "passed, so the delivered set is filtered by the very thing being measured and "
                "has almost no negative cases. The deterministic confusion matrix has two empty "
                "cells entirely (validator=supported/human=1: 681, human=0: 9; "
                "validator=unsupported: 0 of either). Raw agreement is 99.4%. The informative "
                "figures are the rates: validator false-positive 9/690 = 1.30% deterministic "
                "and 12/568 = 2.11% hybrid. The hybrid arm has negative cases only because 14 "
                "withheld (dropped) claims were sampled into the annotation frame, and both "
                "raters judged all 14 supported -- a false-negative estimate of 14/14 on a "
                "sample of 14, which is directional evidence that the validator is "
                "conservative, not a population rate."),
            "annotation_effort_was_not_timed": (
                "Labels were collected through spreadsheet files rather than the web UI, so no "
                "per-item duration was recorded: annotation_manifest.json reports "
                "timed_annotations 0 and median_duration_ms null. A claim about time per item "
                "cannot be supported from this release."),
            "rater_scale_range": (
                "On the 0-3 relevance scale neither rater used 3, and rater 2 never used 0. "
                "That narrower range is why the unweighted kappa over the 50 newly annotated "
                "delta pairs is low (0.31-0.38) while the weighted kappa over the full 390/396 "
                "pairs is 0.93: the disagreements are overwhelmingly adjacent grades. All 30 "
                "disagreements had a gap of exactly 1."),
        },
        "human_annotation": {
            "claim_unit": "annotation_signature (the proposition), never claim_id",
            "why": (
                "claim_id digests the rendered sentence, so one id covers several propositions "
                "whenever a sentence formats identically at different values. Keying on it "
                "merged 278 of 694 propositions on an earlier pilot."),
            "coverage_rule": (
                "Human ranking metrics and claim kappa are withheld below 100% coverage of the "
                "PRE-REGISTERED annotation universe. An unlabelled returned pair is UNKNOWN, "
                "not irrelevant; scoring it 0 would bias every human figure downward by exactly "
                "the amount of missing annotation."),
            "adjudication_rule": (
                "Concordant raters are their own gold. A disagreement contributes only a "
                "RECORDED verdict; an unadjudicated disagreement is dropped and counted, never "
                "averaged. All 30 disagreements were adjudicated by a third party."),
            "relevance_labels_reused_from_v1": (
                "A relevance judgement is about a scenario and a posting, not about the system "
                "version, so v1's labels stay valid and only the coverage delta was newly "
                "annotated: 22 pairs (deterministic) and 28 (hybrid). Claim labels were NOT "
                "reused -- they overlap the current signature universe by zero, because P0-4 "
                "and P0-5 changed claim predicates and texts."),
        },
        "excluded_from_this_release": {
            "normalized_tables": "regenerable from the run bundles",
            "run_bundles": "~290 MB across the pair; git is the wrong transport",
            "annotation_sqlite_store": (
                "the working store; human_annotations.jsonl is its archival form and is "
                "included"),
        },
    }


def _readme(p: dict[str, Any]) -> str:
    det = p["experiments"]["deterministic"]
    hyb = p["experiments"]["hybrid"]
    lat = p["experiments"]["latency_serial_subexperiment"]
    return f"""# CMJCC final release v2 -- the 588-run main experiment

{p["total_runs"]} runs: **{det["run_count"]} deterministic** + **{hyb["run_count"]} hybrid**.
Code `{det["code_version"]}` at commit `{det["commit_hash"][:12]}`, canonical oracle
`{det.get("canonical_oracle", {}).get("version")}`. Both arms record `git_dirty: false` and share
one source fingerprint, so the pair was produced by identical code.

> **This release supersedes `final_release/` (v1).** v1 used code 0.1.0 and canonical oracle
> 1.0.0, so its grade-derived numbers are measured against a different ground truth and must not
> be cited alongside these. See `provenance.json` -> `supersedes`.

## Layout

| Path | What |
|---|---|
| `deterministic/{det["experiment_id"]}/` | metrics, statistics, plots, report, manifests, audit |
| `hybrid/{hyb["experiment_id"]}/` | the same, for the Hybrid arm |
| `latency_serial/{lat["experiment_id"]}/` | serial sub-experiment for single-request latency |
| `*/audit_evidence/` | replay diff, fallback diagnosis, provenance audit |
| `*/human_annotations/` | the human labels, their pre-registered frame, the adjudication record |
| `inputs/` | scenarios, canonical oracle, configs, merged human relevance labels |
| `checksums.json` | sha256 over every file here |

## Verification

    python -m jobrec_eval.cli verify <run bundle tree>     # bundles, if you have them
    python scripts/verify_release_v2.py                    # this tree against checksums.json

## Headline numbers

Both arms: 0 crashed runs, replay identical for every run (210/210 and {hyb["run_count"]}/{hyb["run_count"]},
0 differences), checksums clean, 0 legacy rule reparse, 0 evidence duplication or turn drift.

Hybrid endpoint behaviour, against the pre-registered thresholds fixed before the batch:
final fallback call rate **{hyb["llm_call_summary"]["rates"]["final_fallback_call_rate"]:.2%}** (limit 1%),
affected run rate **{hyb["llm_call_summary"]["rates"]["final_fallback_run_rate"]:.2%}** (limit 2%),
retry recovery **{hyb["llm_call_summary"]["rates"]["retry_recovery_rate"]:.2%}**.

Human annotation: claim kappa 0.815 (deterministic, n=694) and 0.795 (hybrid, n=588) at 100%
coverage of the pre-registered universe; relevance weighted kappa 0.931 / 0.928 with
oracle-vs-human 0.920 / 0.913.

## Read `provenance.json` -> `known_limitations` before quoting anything

Five are recorded, with measurements: the Hybrid arm's fallback concentration is an
execution-order artifact (impact under 0.011 on every metric); latency in the Hybrid arm is
wall-clock under concurrency 20; `validator_vs_human_kappa` is near zero by construction rather
than through disagreement; annotation was not timed; and neither rater used the top of the
relevance scale.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    if args.write and ROOT.exists():
        import shutil
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
        for name in _AUDIT_FILES:
            src = spec["audit"] / name
            if src.is_file():
                total += _copy(src, target / "audit_evidence" / name, args.write)
        for src in sorted(spec["audit"].glob("p0_2_audit.*.json")):
            total += _copy(src, target / "audit_evidence" / src.name, args.write)
        for rel, name in _ANNOTATION_FILES:
            src = spec["annotation"] / rel
            if src.is_file():
                total += _copy(src, target / "human_annotations" / name, args.write)
        adj = spec["adjudication"] / "adjudication.csv"
        if adj.is_file():
            total += _copy(adj, target / "human_annotations" / "adjudication.csv",
                           args.write)

    lat_target = ROOT / "latency_serial" / LATENCY_SUB["experiment_id"]
    print(f"latency_serial -> {lat_target}")
    for name in ("report", "metrics", "manifests"):
        total += _copy(LATENCY_SUB["analysis"] / name, lat_target / name, args.write)
    for name in _RUN_FILES:
        total += _copy(LATENCY_SUB["runs"] / name,
                       lat_target / "run_bundle_provenance" / name, args.write)

    print("inputs ->", ROOT / "inputs")
    for path in _INPUTS:
        total += _copy(path, ROOT / "inputs" / path.name, args.write)
    for arm in ("deterministic", "hybrid"):
        src = Path("artifacts/human_inputs") / arm / "relevance_labels_human.csv"
        if src.is_file():
            total += _copy(src, ROOT / "inputs" / f"relevance_labels_human.{arm}.csv",
                           args.write)

    if args.write:
        print(f"normalised {_normalize_newlines(ROOT)} copied text file(s) to LF")
        provenance = _provenance()
        (ROOT / "provenance.json").write_text(
            json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8",
            newline="\n")
        (ROOT / "README.md").write_text(_readme(provenance), encoding="utf-8",
                                        newline="\n")
        manifest_path = ROOT / "checksums.json"
        files = sorted(p for p in ROOT.rglob("*") if p.is_file() and p != manifest_path)
        manifest_path.write_text(json.dumps({
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


if __name__ == "__main__":
    raise SystemExit(main())
