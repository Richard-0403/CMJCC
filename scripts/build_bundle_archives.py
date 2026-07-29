"""Build, audit and fingerprint the standalone bundle archive of each official experiment.

`final_release/` carries the slim citable subset. This builds the OTHER half: the complete
evidence for each official experiment -- every run bundle with its model-call records and
raw responses, the normalized tables, the manifests, the verify/replay audits and the frozen
configs -- as one ZIP per experiment. Those are far too large for ordinary git; the ZIP name,
size, sha256 and contents are recorded in `final_release/provenance.json` so the copy stays
verifiable without being committed.

Every archive is audited before its digest is taken, because a digest over an archive nobody
opened only proves the bytes did not change afterwards:

1. secrets scan over the staged tree, matching credential VALUES (the live key from the
   environment, plus secret-shaped tokens) rather than field names -- the archive is
   SUPPOSED to record ``api_key_env`` and ``api_key_present``;
2. ZIP integrity (``testzip``);
3. extract to a temporary directory and re-verify each tree against the ``checksums.json``
   it carries, so the round trip is proven rather than assumed;
4. sha256 of the finished archive.

Run:
    python scripts/build_bundle_archives.py            # report
    python scripts/build_bundle_archives.py --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

# Import a sibling script while keeping the working directory at the repository root: the
# experiment paths in OFFICIAL are repo-relative, so running from ``scripts/`` would find
# nothing -- silently producing an empty archive rather than failing.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_final_release import OFFICIAL  # noqa: E402

DIST = Path("dist")

#: Credential VALUE patterns. Field names are deliberately not matched: see the module
#: docstring.
_SECRET_PATTERNS = (
    re.compile(r"sk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"Bearer\s+[A-Za-z0-9_\-.]{20,}"),
)
_SECRET_ENV_VARS = ("JOBREC_LLM_API_KEY",)

#: Text extensions worth scanning. Binary artifacts (plots) cannot carry a pasted key.
_TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".yaml", ".yml", ".md", ".txt", ".log"}

#: Frozen inputs copied into every archive so it is self-contained.
_CONFIGS = (Path("configs/base.yaml"), Path("configs/experiment_full.yaml"),
            Path("configs/hybrid_vectorengine.yaml"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scan_secrets(root: Path) -> list[str]:
    live = [v for name in _SECRET_ENV_VARS
            if (v := os.environ.get(name)) and len(v) >= 12]
    hits: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in _TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        rel = path.relative_to(root).as_posix()
        if any(secret in text for secret in live):
            hits.append(f"{rel}: CONFIGURED API KEY")
            continue
        for pattern in _SECRET_PATTERNS:
            if pattern.search(text):
                hits.append(f"{rel}: credential-shaped token /{pattern.pattern}/")
                break
    return hits


def _verify_checksums(tree: Path) -> tuple[int, list[str]]:
    """Re-verify ``tree`` against the ``checksums.json`` it carries, after extraction.

    Delegates to the project's own verifier -- the same one ``jobrec_eval.cli verify``
    uses -- rather than reimplementing the manifest format. A hand-rolled version here
    assumed a ``{"files": {...}}`` shape while the real manifest is a flat
    ``{path: sha256}`` mapping, so it found zero entries, reported no problems, and the
    audit silently checked nothing. Reusing the canonical verifier removes that whole
    class of mistake.
    """
    from jobrec.evaluation.checksums import (
        CHECKSUMS_FILENAME,
        MissingChecksumsError,
        verify_checksums,
    )

    manifest_path = tree / CHECKSUMS_FILENAME
    if not manifest_path.exists():
        return 0, [f"{tree.name}: no {CHECKSUMS_FILENAME}"]
    recorded = len(json.loads(manifest_path.read_text(encoding="utf-8")))
    try:
        findings = verify_checksums(tree, report_untracked=True)
    except MissingChecksumsError:
        return 0, [f"{tree.name}: no {CHECKSUMS_FILENAME}"]
    except (NotADirectoryError, ValueError) as exc:
        return recorded, [f"{tree.name}: {exc}"]
    if not recorded:
        return 0, [f"{tree.name}: {CHECKSUMS_FILENAME} records no artifacts"]
    return recorded, [f"{tree.name}: {f.describe()}" for f in findings]


def _members(spec: dict) -> list[tuple[Path, str]]:
    """``(source, name inside the archive)`` for one experiment's complete evidence."""
    experiment_id = spec["experiment_id"]
    members: list[tuple[Path, str]] = []
    for source, prefix in ((spec["runs"], "runs"), (spec["analysis"], "analysis")):
        for path in sorted(source.rglob("*")):
            if path.is_file():
                members.append((path, f"{prefix}/{path.relative_to(source).as_posix()}"))
    audit = Path("artifacts/reports") / f"replay_diff_{experiment_id}.json"
    if audit.exists():
        members.append((audit, "audit/replay_diff.json"))
    for config in _CONFIGS:
        if config.exists():
            members.append((config, f"configs/{config.name}"))
    return members


def _readme(label: str, spec: dict, counts: dict[str, int]) -> str:
    return f"""# CMJCC {label} experiment archive -- {spec['experiment_id']}

Complete evidence for one of the two official CMJCC experiments. The slim citable subset
(reports, tables, plots, manifests) is in the repository under `final_release/`; this
archive is the full underlying record, which is too large for ordinary git.

| | |
|---|---|
| experiment id | `{spec['experiment_id']}` |
| config | `{spec['config']}` |
| variants x repeats | {len(spec['variants'])} x {spec['repeats']} |
| run bundles | {counts['runs']} files |
| analysis tree | {counts['analysis']} files |

## Contents

- `runs/` every run bundle: `run_record.json`, the candidate/dialogue/active-search/
  job-context states, retrieval and eligibility results, the recommendation decision,
  response and claims, the evidence log and evidence items, `handoffs.jsonl`,
  `dialogue_trace.jsonl`, `log_trace.jsonl`, `turn_records.jsonl`, `run_totals.json`,
  `component_latency.json`, and `model_calls.jsonl` with the redacted raw responses.
  Also the experiment-level `experiment_manifest.json`, `runs_index.csv`,
  `failures.csv`, `resolved_config.yaml`, the catalog and scenario snapshots, and
  `checksums.json`.
- `analysis/` the full analysis tree including the `normalized/` tables that
  `final_release/` omits, plus `metrics/`, `statistics/`, `plots/`, `manifests/`
  (experiment manifest, analysis plan, frozen canonical oracle), `audit/`, the
  data-quality report and `checksums.json`.
- `audit/replay_diff.json` the recorded replay result for this experiment.
- `configs/` the frozen configuration files.

## Verifying this archive

Its sha256, size and file count are recorded in `final_release/provenance.json`.
After extracting:

```
python -m jobrec_eval.cli verify <extracted>/analysis
python -m jobrec_eval.cli verify <extracted>/runs
python -m jobrec_eval.cli replay <extracted>/runs
```

No prompt is stored anywhere in this archive, and no credential: raw responses are
redacted on the way out and only `api_key_env` (the variable's NAME) and
`api_key_present` (a boolean) are recorded. The build scans for credential values
before computing the digest.
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    results: dict[str, Any] = {}
    ok = True
    for label, spec in OFFICIAL.items():
        members = _members(spec)
        if not members:
            print(f"FAIL {label}: found no files under {spec['runs']} / "
                  f"{spec['analysis']} -- run this from the repository root")
            return 2
        counts = {
            "runs": sum(1 for _s, n in members if n.startswith("runs/")),
            "analysis": sum(1 for _s, n in members if n.startswith("analysis/")),
        }
        name = f"CMJCC_{label}_{spec['experiment_id']}.zip"
        print(f"{label}: {len(members)} files -> dist/{name}")
        if not args.write:
            continue

        DIST.mkdir(parents=True, exist_ok=True)
        target = DIST / name
        if target.exists():
            target.unlink()
        with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
            for source, arcname in members:
                zf.write(source, arcname)
            zf.writestr("README.md", _readme(label, spec, counts))

        # --- audit ------------------------------------------------------------
        problems: list[str] = []
        with zipfile.ZipFile(target) as zf:
            broken = zf.testzip()
            if broken is not None:
                problems.append(f"ZIP integrity: first bad member {broken}")
            with tempfile.TemporaryDirectory(prefix="cmjcc-archive-audit-") as tmp:
                root = Path(tmp)
                zf.extractall(root)
                leaks = _scan_secrets(root)
                if leaks:
                    problems.extend(f"SECRET {x}" for x in leaks)
                verified = 0
                for tree in ("analysis", "runs"):
                    n, bad = _verify_checksums(root / tree)
                    verified += n
                    problems.extend(bad)

        digest = _sha256(target)
        size = target.stat().st_size
        status = "OK" if not problems else "FAIL"
        print(f"  size {size / 1024 / 1024:.1f} MB | members {len(members) + 1} | "
              f"checksummed files re-verified {verified} | {status}")
        for problem in problems[:10]:
            print(f"    - {problem}")
        ok = ok and not problems
        results[label] = {
            "archive": name,
            "size_bytes": size,
            "size_mb": round(size / 1024 / 1024, 2),
            "sha256": digest,
            "member_count": len(members) + 1,
            "run_bundle_files": counts["runs"],
            "analysis_files": counts["analysis"],
            "checksummed_files_reverified_after_extraction": verified,
            "zip_integrity": "ok" if broken is None else f"bad member {broken}",
            "secrets_scan": "clean" if not any(p.startswith("SECRET") for p in problems)
                            else "FOUND",
            "audit_problems": problems,
            "contents": [
                "runs/ -- every run bundle incl. model_calls.jsonl with redacted raw "
                "responses, turn_records.jsonl, run_totals.json, evidence items and log "
                "traces, plus the experiment manifest, runs index, failures, resolved "
                "config, catalog and scenario snapshots and checksums",
                "analysis/ -- full analysis tree incl. normalized tables, metrics, "
                "statistics, plots, manifests (with the frozen canonical oracle), audit "
                "tables, data-quality report and checksums",
                "audit/replay_diff.json -- recorded replay result",
                "configs/ -- frozen configuration files",
                "README.md",
            ],
        }

    if args.write:
        provenance_path = Path("final_release/provenance.json")
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
        provenance["bundle_archives"] = {
            "location": "dist/ (NOT in version control: too large for ordinary git)",
            "note": (
                "Each archive holds the complete evidence for one official experiment -- "
                "run bundles with model-call records and redacted raw responses, the "
                "normalized tables, manifests, the replay audit and the frozen configs. "
                "Every archive was audited before its digest was taken: secrets scan for "
                "credential VALUES, ZIP integrity, and extraction followed by re-verifying "
                "each tree against the checksums.json it carries. A digest over an archive "
                "nobody opened would only prove the bytes had not changed since."
            ),
            "archives": results,
        }
        provenance["excluded_from_this_release"] = {
            "normalized_tables": "carried in the bundle archives instead",
            "run_bundles": "carried in the bundle archives instead",
        }
        provenance_path.write_text(json.dumps(provenance, indent=2, sort_keys=True),
                                   encoding="utf-8")
        print(f"\nrecorded in {provenance_path}")

        # This script is the LAST writer of provenance.json, so it has to re-stamp the
        # release manifest. Otherwise the release's own checksums.json -- written by
        # build_final_release.py before the archives existed -- records a stale digest for
        # provenance.json, and the release fails its own self-verification.
        release = provenance_path.parent
        manifest_path = release / "checksums.json"
        files = sorted(p for p in release.rglob("*")
                       if p.is_file() and p != manifest_path)
        manifest_path.write_text(json.dumps({
            "algorithm": "sha256",
            "file_count": len(files),
            "files": {p.relative_to(release).as_posix(): _sha256(p) for p in files},
        }, indent=2, sort_keys=True), encoding="utf-8")
        print(f"re-stamped {manifest_path} over {len(files)} files")
        print("all archives audited clean:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
