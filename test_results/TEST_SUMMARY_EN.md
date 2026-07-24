# CMJCC — Test & Evaluation Summary (English)

This folder collects the test results and a summary for the CMJCC conversational
job recommendation prototype: (1) software test results, and (2) the RQ4
evaluation experiment produced by the `jobrec_eval` pipeline (now including
per-constraint compliance, no-match/clarification precision–recall, a root-cause
error taxonomy, and representative case studies).

- Prototype code version: commit on `main` (Phases A–G + evaluation additions).
- Evaluation experiment id: `exp-8793b18de5b2` (deterministic run mode).
- Tables/plots/report here are copies of `evaluation/outputs/exp-8793b18de5b2/`
  and are fully reproducible.

---

## 1. Software test results

| Suite | Result |
|---|---|
| unit (extraction, constraints, memory conflicts, ranking, claim validator, config hash) | pass |
| contract (schema validation rejects malformed input / unknown enums / extra fields) | pass |
| integration (full pipeline, LLM-failure fallback, multi-turn memory) | pass |
| e2e (FastAPI endpoints) | pass |
| golden (10 scenarios + ablation-difference assertions) | pass |
| property (no hard-violating job selected by full; score == Σ contributions; every claim resolves; top-k) | pass |
| eval unit (NDCG hand-calc, McNemar discordant, Holm, bootstrap seed) | pass |

- **Totals:** 68 tests pass; 1 PostgreSQL-marked test skipped by default
  (verified separately against a live PostgreSQL 15 instance).
- **Coverage:** ~83% overall; core logic higher (CMJCC 91%, ranking 98%,
  orchestrator 89%). **Lint:** `ruff` clean. **Determinism:** repeats identical
  (0 variance).
- **Bug fixed during evaluation:** the clarifier no longer re-asks for salary
  currency when it is already established by a prior turn (e.g. "RM8000" then
  "4000 is also fine"); this was previously causing a spurious clarification.

---

## 2. Evaluation experiment (RQ4)

- Design: 42 tagged scenarios × 5 variants × 3 repeats = **630 runs**,
  **0 system failures**. Frozen catalog snapshot, prompts, seeds; reference date
  2026-01-01; top-k = 5. Scenario expectations were QA'd against the catalog
  (two multiple-hard scenarios that are jointly infeasible were relabelled as
  correct no-match).
- Scenario mix: complete 6, clarification 5, profile-dialogue conflict 5,
  preference change 12, multiple hard 5, soft trade-off 4, ambiguous role 2,
  no-match 3. Memory-dependent (≥ medium): 16; context-dependent (high): 15.

### 2.1 Overall results by variant (scenario-mean)

| variant | NDCG@5 | P@5 | HCSR | Task success | Grounding | Handoff |
|---|---|---|---|---|---|---|
| **full** | 0.951 | 0.973 | **1.000** | **1.000** | 1.000 | 1.000 |
| no_memory | 0.949 | 1.000 | 1.000 | 0.786 | 1.000 | 1.000 |
| one_shot | 0.949 | 1.000 | 1.000 | 0.786 | 1.000 | 1.000 |
| no_context | 0.700 | 0.558 | **0.571** | 0.310 | 1.000 | 1.000 |
| profile_only | 0.587 | 0.500 | 0.500 | 0.333 | 1.000 | 1.000 |

Source: `tables/variant_summary.csv`.

### 2.2 Per-constraint compliance (recommended jobs vs authoritative hard constraints)

| constraint field | full | no_context | profile_only |
|---|---|---|---|
| location | 1.000 | 0.600 | 0.073 |
| salary_min | 1.000 | 0.800 | 0.642 |
| work_mode | 1.000 | 0.457 | 0.160 |
| not_expired | 1.000 | 0.846 | 1.000 |

`no_context` also uses "unknown" to pass ~37% of work-mode checks. Source:
`tables/constraint_compliance.csv`.

### 2.3 Job-context contribution (full vs no_context, context-dependent scenarios)

| Metric | full | no_context | Δ | 95% CI | p | n |
|---|---|---|---|---|---|---|
| HCSR | 1.000 | 0.480 | **+0.520** | [0.400, 0.660] | 0.002 | 10 |
| Task success | 1.000 | 0.000 | **+1.000** | [1.000, 1.000] | <0.001 | 15 |
| Mean violations/job | 0.000 | 0.740 | **−0.740** | [−1.120, −0.460] | 0.002 | 10 |
| NDCG@5 | 0.914 | 0.655 | +0.259 | [0.156, 0.418] | 0.002 | 10 |

Source: `tables/context_contribution.csv`.

### 2.4 Memory contribution (full vs no_memory, memory-dependent scenarios)

| Metric | full | no_memory | Δ | 95% CI | p | n |
|---|---|---|---|---|---|---|
| Task success | 1.000 | 0.438 | **+0.562** | [0.312, 0.812] | <0.001 (McNemar) | 16 |
| NDCG@5 | 0.939 | 0.925 | +0.014 | [0.000, 0.042] | 1.000 | 7 |

Source: `tables/memory_contribution.csv`.

### 2.5 No-match & clarification correctness

- No-match: **full precision = recall = F1 = 1.00**; `no_context` and
  `profile_only` recall = 0 (they never correctly detect no-match because they
  do not enforce hard constraints). Source: `tables/no_match_metrics.csv`.
- Clarification: `full` correctly clarifies on missing-role / ambiguous-role
  scenarios; `no_memory`/`one_shot` additionally (correctly) clarify on
  multi-turn scenarios where the role was only stated earlier.

### 2.6 Root-cause error taxonomy (task-unsuccessful runs)

| Category | % | Most-affected variant |
|---|---|---|
| missing constraint enforcement (ablation) | 38.7 | no_context |
| missing dialogue evidence (baseline) | 37.3 | profile_only |
| stale/missing memory (ablation) | 24.0 | no_memory |

`full` has no task failures in this run. Source: `tables/error_taxonomy.csv`.
Five representative case studies (memory-helps, context-helps, correct no-match,
hardest full case, claim-validator) are in `analysis_report.md` §9.1.

---

## 3. Key findings

1. The full architecture satisfies all designated hard constraints
   (HCSR = 1.00), returns no unsupported factual claims (grounding = 1.00),
   correctly detects no-match (F1 = 1.00) and passes all handoffs.
2. **Job-context orchestration is the largest contributor**: removing it drops
   HCSR to 0.57 and adds ~0.74 hard-constraint violations per recommended job;
   on context-dependent scenarios task success drops from 1.00 to 0.00.
3. **Candidate memory clearly helps multi-turn scenarios**: task success +0.56
   on memory-dependent scenarios (McNemar p < 0.001).
4. Effects concentrate where expected (dependency subsets), consistent with the
   components doing their intended jobs.

---

## 4. Status of the three requested extensions

- **(A) Data-derivable additions — DONE.** Scenario QA/relabel, per-constraint
  compliance, no-match/clarification precision–recall, case studies and error
  taxonomy are implemented and in the report.
- **(C) Human-annotation support — READY, pending labels.** The pipeline emits
  annotation templates (`evaluation/outputs/<exp>/annotation/`). Dropping in
  `relevance_labels_human.csv` / `claim_annotations_human.csv` makes the pipeline
  compute weighted Cohen's κ (relevance), Cohen's κ (claims) and oracle-vs-human
  agreement automatically. No human labels are fabricated.
- **(B) Real LLM (hybrid) run — WIRED, pending API key.** A `hybrid` config for
  Vector Engine (gpt-5.5, OpenAI-compatible) is provided
  (`configs/hybrid_vectorengine.yaml`). Set `JOBREC_LLM_API_KEY`,
  `JOBREC_LLM_BASE_URL=https://api.vectorengine.ai/v1`, `JOBREC_LLM_MODEL=gpt-5.5`
  and run the pipeline with `--config configs/hybrid_vectorengine.yaml` to make
  grounding / extraction / latency real numbers.

---

## 5. Limitations (for the paper)

- **Relevance is scored by a transparent automatic oracle, not human raters** in
  this run; NDCG/P@5/MGR measure agreement with a deterministic reference. No
  inter-rater agreement is reported yet (construct-validity threat) — see (C).
- **Grounding and handoff are 1.00 by construction** under the deterministic
  backend (templated explanation emits only pre-validated claims). These are
  correctness guarantees, not empirical variation; they become meaningful under
  the real LLM (B).
- Latency is deterministic-compute only (real cost is the LLM). Response turns do
  not discriminate variants here.
- Small synthetic catalog and modest scenario count; results do not extrapolate
  to real hiring outcomes.

---

## 6. How to reproduce

```bash
pip install -e ".[dev,eval]"
python scripts/generate_raw_catalog.py --output data/raw/jobs.csv --count 200
python scripts/prepare_catalog.py --input data/raw/jobs.csv --out-dir data/processed
python scripts/build_eval_scenarios.py --output evaluation/data/scenarios.jsonl
python -m jobrec_eval.cli pipeline --repeats 3 --bootstrap-iters 5000   # deterministic
# Real LLM (needs key):
JOBREC_LLM_API_KEY=... JOBREC_LLM_BASE_URL=https://api.vectorengine.ai/v1 \
JOBREC_LLM_MODEL=gpt-5.5 \
  python -m jobrec_eval.cli pipeline --config configs/hybrid_vectorengine.yaml --repeats 3
pytest -m "not postgres"
```
