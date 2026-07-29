# CMJCC final release

The **only** citable experiment pair. Anything else under `evaluation/` is superseded
history; see `provenance.json` for the list and why.

| | deterministic | hybrid |
|---|---|---|
| experiment id | `exp-e748800507ef` | `exp-6db1e87daed5` |
| config | `configs/experiment_full.yaml` | `configs/hybrid_vectorengine.yaml` |
| variants x repeats | 5 x 1 | 3 x 3 |
| runs | 210 / 210 (crashed 0) | 378 / 378 (crashed 0) |
| commit | `f7970b81f653` | `f7970b81f653` |
| execution fingerprint | `f3ef9775f6b6d08a` | `f3ef9775f6b6d08a` |
| canonical oracle | v3.0.0, declared (42 declared / 0 system-derived) | v3.0.0, declared (42 / 0) |

Both experiments ran from the same frozen source: identical `commit_hash` and identical
`execution_fingerprint`.

## On the hybrid `git_dirty=true`

The hybrid manifest records git_dirty=true and it is left as recorded -- the history is not rewritten to make it look clean. It does NOT mean the hybrid run used modified code: its commit_hash and execution_fingerprint are identical to the deterministic run's (commit f7970b81f653, execution fingerprint f3ef9775f6b6d08a). The flag is set because git status was non-empty at run time, and it was non-empty only because the deterministic run had just written its analysis tree into evaluation/outputs/, which was untracked at the time. In other words, producing one official artifact made the next one look dirty. Those output trees are now gitignored and the citable subset is committed here instead, so the flag cannot be produced this way again.

## Layout

- `deterministic/exp-e748800507ef/` and `hybrid/exp-6db1e87daed5/`
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
210/210 and 378/378 runs with
0 differences.

Built from `1a5db3a` by `scripts/build_final_release.py`.
