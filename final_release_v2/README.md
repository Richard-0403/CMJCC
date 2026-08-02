# CMJCC final release v2 -- the 588-run main experiment

588 runs: **210 deterministic** + **378 hybrid**.
Code `0.2.0` at commit `40ded11a8222`, canonical oracle
`4.0.0`. Both arms record `git_dirty: false` and share
one source fingerprint, so the pair was produced by identical code.

> **This release supersedes `final_release/` (v1).** v1 used code 0.1.0 and canonical oracle
> 1.0.0, so its grade-derived numbers are measured against a different ground truth and must not
> be cited alongside these. See `provenance.json` -> `supersedes`.

## Layout

| Path | What |
|---|---|
| `deterministic/exp-40a9cd647575/` | metrics, statistics, plots, report, manifests, audit |
| `hybrid/exp-2b33b808a0f8/` | the same, for the Hybrid arm |
| `latency_serial/exp-e63f05ad75bb/` | serial sub-experiment for single-request latency |
| `*/audit_evidence/` | replay diff, fallback diagnosis, provenance audit |
| `*/human_annotations/` | the human labels, their pre-registered frame, the adjudication record |
| `inputs/` | scenarios, canonical oracle, configs, merged human relevance labels |
| `checksums.json` | sha256 over every file here |

## Verification

    python -m jobrec_eval.cli verify <run bundle tree>     # bundles, if you have them
    python scripts/verify_release_v2.py                    # this tree against checksums.json

## Headline numbers

Both arms: 0 crashed runs, replay identical for every run (210/210 and 378/378,
0 differences), checksums clean, 0 legacy rule reparse, 0 evidence duplication or turn drift.

Hybrid endpoint behaviour, against the pre-registered thresholds fixed before the batch:
final fallback call rate **0.97%** (limit 1%),
affected run rate **1.85%** (limit 2%),
retry recovery **96.32%**.

Human annotation: claim kappa 0.815 (deterministic, n=694) and 0.795 (hybrid, n=588) at 100%
coverage of the pre-registered universe; relevance weighted kappa 0.931 / 0.928 with
oracle-vs-human 0.920 / 0.913.

## Read `provenance.json` -> `known_limitations` before quoting anything

Five are recorded, with measurements: the Hybrid arm's fallback concentration is an
execution-order artifact (impact under 0.011 on every metric); latency in the Hybrid arm is
wall-clock under concurrency 20; `validator_vs_human_kappa` is near zero by construction rather
than through disagreement; annotation was not timed; and neither rater used the top of the
relevance scale.
