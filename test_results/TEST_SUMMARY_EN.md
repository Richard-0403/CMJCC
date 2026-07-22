# CMJCC — Test & Evaluation Summary (English)

This folder collects the test results and a summary for the CMJCC conversational
job recommendation prototype. It has two parts: (1) software test results
(unit / contract / integration / e2e / golden), and (2) the RQ4 evaluation
experiment produced by the `jobrec_eval` pipeline.

- Prototype code version: commit on `main` (Phases A–G).
- Evaluation experiment id: `exp-8793b18de5b2` (deterministic run mode).
- Generated tables/plots/report in this folder are copies of
  `evaluation/outputs/exp-8793b18de5b2/` and are fully reproducible.

---

## 1. Software test results

| Suite | What it checks | Result |
|---|---|---|
| unit | extraction, constraints/eligibility, memory conflicts, ranking, claim validator, config hash | pass |
| contract | schema validation rejects malformed input / unknown enums / extra fields | pass |
| integration | full pipeline via AppService, LLM-failure fallback, multi-turn memory | pass |
| e2e | FastAPI endpoints (candidates / sessions / turns / runs) | pass |
| golden | 10 golden scenarios + ablation-difference assertions | pass |
| property | invariants (no hard-violating job selected by full; score == Σ contributions; every claim resolves; top-k respected) | pass |
| eval unit | NDCG hand-calc, McNemar discordant, Holm, bootstrap seed reproducibility | pass |

- **Totals:** 68 tests pass, 1 PostgreSQL-marked test skipped by default
  (verified separately against a live PostgreSQL 15 instance: candidate, run,
  decision, 5 handoffs, 28 claims and 290 evidence items persisted and reloaded).
- **Coverage:** ~83% overall; core logic higher (CMJCC 91%, ranking 98%,
  orchestrator 89%, features 86%).
- **Lint:** `ruff` clean.
- **Determinism:** replaying the same scenario produces identical decisions
  (verified: 0 variance across repeats in the evaluation).

---

## 2. Evaluation experiment (RQ4)

- Design: 42 tagged scenarios × 5 variants × 3 repeats = **630 runs**,
  **0 system failures**. Frozen catalog snapshot, prompts, seeds; reference date
  2026-01-01; top-k = 5.
- Scenario mix by type: complete 6, clarification 5, profile-dialogue conflict 5,
  preference change 12, multiple hard 5, soft trade-off 4, ambiguous role 2,
  no-match 3. Memory-dependent (≥ medium): 16; context-dependent (high): 15.

### 2.1 Overall results by variant (scenario-mean)

| variant | NDCG@5 | P@5 | HCSR | Task success | Grounding | Handoff | Turns | Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **full** | 0.947 | 0.973 | **1.000** | 0.905 | 1.000 | 1.000 | 1.33 | 13.5 |
| no_memory | 0.949 | 1.000 | 1.000 | 0.738 | 1.000 | 1.000 | 1.33 | 10.6 |
| one_shot | 0.949 | 1.000 | 1.000 | 0.738 | 1.000 | 1.000 | 1.33 | 10.6 |
| no_context | 0.690 | 0.558 | **0.558** | 0.310 | 1.000 | 1.000 | 1.33 | 20.6 |
| profile_only | 0.587 | 0.500 | 0.500 | 0.333 | 1.000 | 1.000 | 1.33 | 14.7 |

Source: `tables/variant_summary.csv`.

### 2.2 Job-context contribution (full vs no_context, context-dependent scenarios)

| Metric | full | no_context | Δ | 95% CI | p | effect | n |
|---|---|---|---|---|---|---|---|
| HCSR | 1.000 | 0.400 | **+0.600** | [0.500, 0.725] | 0.008 | 3.24 | 8 |
| Task success | 0.733 | 0.000 | **+0.733** | [0.467, 0.933] | <0.001 | 1.00 | 15 |
| Mean violations/job | 0.000 | 0.875 | **−0.875** | [−1.300, −0.600] | 0.008 | −1.55 | 8 |
| NDCG@5 | 0.892 | 0.609 | +0.284 | [0.158, 0.475] | 0.008 | 1.08 | 8 |

Removing explicit hard/soft orchestration lets hard-constraint-violating jobs
into the results. Source: `tables/context_contribution.csv`.

### 2.3 Memory contribution (full vs no_memory, memory-dependent scenarios)

| Metric | full | no_memory | Δ | 95% CI | p | effect | n |
|---|---|---|---|---|---|---|---|
| Task success | 0.875 | 0.438 | **+0.438** | [0.188, 0.688] | 0.016 (McNemar) | 1.00 | 16 |
| NDCG@5 | 0.939 | 0.925 | +0.014 | [0.000, 0.042] | 1.000 | 0.38 | 7 |

Prior-turn memory recovers scenarios where the role stated in an earlier turn is
otherwise lost. Source: `tables/memory_contribution.csv`.

---

## 3. Key findings

1. The full architecture satisfies all designated hard constraints
   (HCSR = 1.00) and drops no unsupported factual claims (grounding = 1.00,
   handoff success = 1.00).
2. **Job-context orchestration is the largest contributor**: removing it
   (no_context) roughly halves HCSR (1.00 → 0.56) and adds ~0.875 hard-constraint
   violations per recommended job.
3. **Candidate memory clearly helps multi-turn scenarios**: task success
   +0.44 on memory-dependent scenarios (McNemar p = 0.016).
4. Effects concentrate where expected (memory-dependent / context-dependent
   subsets), consistent with the components doing their intended jobs.

---

## 4. Limitations (important for the paper)

- **Relevance is scored by a transparent automatic oracle, not human raters.**
  NDCG@5 / Precision@5 / Mean Graded Relevance therefore measure agreement with a
  deterministic rule-based reference. No inter-rater agreement is reported
  (construct-validity threat). Annotation-export slots exist for real raters.
- **Explanation grounding** uses the system's claim validator (supported / total
  factual claims), not a separate human judgement.
- Runs use a **deterministic mock LLM backend**, so repeats are identical
  (0 variance); a real model backend and ≥3 meaningful repeats are future work.
- Small synthetic catalog and modest scenario count; results do not extrapolate
  to real hiring outcomes. Report claims are limited to this configuration.
- The strongest, oracle-independent evidence is the ablation HCSR / task-success
  deltas with confidence intervals.

---

## 5. How to reproduce

```bash
pip install -e ".[dev,eval]"
python scripts/generate_raw_catalog.py --output data/raw/jobs.csv --count 200
python scripts/prepare_catalog.py --input data/raw/jobs.csv --out-dir data/processed
python scripts/build_eval_scenarios.py --output evaluation/data/scenarios.jsonl
python -m jobrec_eval.cli pipeline --repeats 3 --bootstrap-iters 5000
pytest -m "not postgres"   # software tests
```

Files in this folder: `analysis_report.md` (full report), `tables/*.csv`
(metrics & statistics), `plots/*.png` (figures).
