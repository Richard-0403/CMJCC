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

## Verifying

```
python -m jobrec_eval.cli verify <analysis dir>     # against the experiment's own checksums
python -m jobrec_eval.cli replay <run bundle dir>   # recompute every run's key states
```

Recorded results: both trees verified OK; replay reproduced
210/210 and 378/378 runs with
0 differences.

Built from `aea5625` by `scripts/build_final_release.py`.
