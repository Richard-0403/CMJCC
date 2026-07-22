# CMJCC Evaluation Pipeline (`jobrec_eval`)

A standalone pipeline that reads the run bundles exported by the main prototype
and produces reproducible RQ4 metrics, ablation analysis, statistics, plots and
an analysis report.

## Integrity / honesty notes

- **Relevance is scored by a deterministic automatic oracle, not human raters**
  (`src/jobrec_eval/relevance.py`). NDCG@5 / Precision@5 / Mean Graded Relevance
  therefore measure agreement with a transparent rule-based reference, applied to
  the whole catalog (so IDCG has no pooling bias). Human annotation and
  inter-rater agreement are left as future work and flagged as a construct-
  validity threat in the report.
- **Explanation grounding** uses the system's claim validator (supported / total
  factual claims), not a separate human judgement.
- Runs are **deterministic** (mock LLM provider), so repeats are identical
  (verified: zero variance across repeats). The `repeat_count` is kept for parity
  with the guide; a real LLM backend would need ≥3.
- HCSR and violation counts are recomputed against the **authoritative** hard
  constraints (the `full` variant's `JobContextState` per scenario), so the
  `no_context` ablation is scored against the true constraints rather than its
  own pass-through eligibility.

## Run

```bash
pip install -e ".[eval]"
python scripts/build_eval_scenarios.py --output evaluation/data/scenarios.jsonl
python -m jobrec_eval.cli pipeline \
  --config configs/experiment_full.yaml \
  --scenarios evaluation/data/scenarios.jsonl \
  --repeats 3 --bootstrap-iters 5000
```

Outputs land under `evaluation/outputs/{experiment_id}/`:

```
manifests/   experiment_manifest.json, analysis_plan.yaml
normalized/  runs, recommendations, constraint_checks, claims, handoffs,
             decision_logs, component_latency, relevance_labels (oracle)
metrics/     run_metrics, scenario_variant_metrics, variant_summary,
             scenario_type_summary, memory_contribution, context_contribution,
             latency_metrics
statistics/  paired_comparisons
plots/       ndcg / hcsr / task_success / grounding by variant,
             memory & context deltas, turns-vs-success, latency breakdown
report/      analysis_report.md, analysis_report_data.json
audit/       invalid_runs, data_lineage, checksums.sha256
```

(The bulky raw run bundles under `_runs/` are git-ignored; regenerate with the
command above.)

## What is measured

Job-match relevance (NDCG@5, P@5, MGR — oracle), hard-constraint satisfaction
(HCSR, mean violation count, unknown-hard rate), task success + no-match
correctness, memory contribution (full vs no_memory), job-context contribution
(full vs no_context), agent-handoff success, decision-log completeness,
recommendation trace completeness, explanation grounding, response turns and
component latency — with paired bootstrap 95% CIs, McNemar (binary task
success), Wilcoxon, effect sizes and Holm correction.
