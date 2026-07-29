# Errata

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
