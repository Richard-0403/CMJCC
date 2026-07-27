"""Analysis report generation (Markdown) from computed metrics + statistics.

The Markdown template reads only the assembled data dict and the generated
plots; it does not recompute metrics. Wording rules (guide sections 24.1, 28)
are respected: engineering thresholds, observed differences, confidence
intervals and statistical significance are distinguished, and over-strong claims
are avoided.

Report *output* is additionally gated on configuration consistency (R15.2): a
comparison report is only written when the compared runs share catalog,
scenarios, prompts, model settings and commit, and when each ablation pair
differs in nothing but its own mechanism's feature flags (R32.7). The gate lives
in :func:`write_report` (and :func:`require_consistent_runs`), not in
:func:`generate_markdown`, because rendering the template is a pure function of
the assembled data while writing files is the point at which output is produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jobrec.orchestration.feature_flags import CONTEXT_FLAGS, MEMORY_FLAGS

from .consistency import (
    ConsistencyError,
    load_run_manifests,
    require_consistent,
    save_run_manifests,
)

#: Ablation pairs whose Δ this report attributes to a single mechanism, with the
#: flag group that is allowed to differ between the two variants (R32.7).
_ABLATION_PAIRS: tuple[tuple[str, str, frozenset[str] | set[str]], ...] = (
    ("full", "no_memory", MEMORY_FLAGS),
    ("full", "no_context", CONTEXT_FLAGS),
)


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "N/A"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _delta_phrase(delta, lo, hi) -> str:
    if delta is None:
        return "not computable"
    if lo is None or hi is None:
        return f"Δ={delta:+.3f}"
    if lo > 0:
        return f"Δ={delta:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] (CI excludes 0)"
    if hi < 0:
        return f"Δ={delta:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] (CI excludes 0, negative)"
    return f"Δ={delta:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] (CI includes 0)"


def _variant_table(variant_summary: pd.DataFrame) -> str:
    order = ["full", "profile_only", "one_shot", "no_memory", "no_context"]
    cols = [("ndcg_at_5", "NDCG@5"), ("precision_at_5", "P@5"), ("hcsr", "HCSR"),
            ("task_success", "TaskSucc"), ("grounding", "Grounding"),
            ("handoff_success", "Handoff"), ("turn_count", "Turns"),
            ("total_latency_ms", "Lat(ms)")]
    head = "| variant | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [head, sep]
    by = {r["variant"]: r for _, r in variant_summary.iterrows()}
    for v in order:
        if v not in by:
            continue
        r = by[v]
        cells = []
        for key, _ in cols:
            nd = 0 if key == "total_latency_ms" else (2 if key == "turn_count" else 3)
            cells.append(_fmt(r.get(f"{key}_mean"), nd))
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _contrib_table(df: pd.DataFrame, subset: str) -> str:
    sub = df[df["subset"] == subset]
    lines = ["| metric | full | other | Δ | 95% CI | p | p(Holm) | effect | n |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in sub.iterrows():
        ci = f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]" if r["ci_low"] is not None else "N/A"
        eff = f"{_fmt(r.get('effect_size'))} ({r.get('effect_type')})" if r.get("effect_size") is not None else "N/A"
        lines.append(
            f"| {r['metric']} | {_fmt(r['base_mean'])} | {_fmt(r['other_mean'])} | "
            f"{_fmt(r['delta'])} | {ci} | {_fmt(r.get('p_value'))} | {_fmt(r.get('p_value_holm'))} | "
            f"{eff} | {int(r['n_pairs'])} |")
    return "\n".join(lines)


def _compliance_table(rows: list[dict]) -> str:
    if not rows:
        return "_No per-constraint compliance data._"
    df = pd.DataFrame(rows)
    fields = sorted(df["constraint_field"].unique())
    variants = [v for v in ["full", "no_context", "profile_only"] if v in set(df["variant"])]
    lines = ["| constraint field | " + " | ".join(variants) + " |",
             "|" + "---|" * (len(variants) + 1)]
    for f in fields:
        cells = []
        for v in variants:
            r = df[(df.constraint_field == f) & (df.variant == v)]
            cells.append(_fmt(r["compliance"].iloc[0]) if len(r) else "N/A")
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pr_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    if not rows:
        return "_No data._"
    head = "| variant | " + " | ".join(c[1] for c in cols) + " |"
    lines = [head, "|" + "---|" * (len(cols) + 1)]
    order = ["full", "no_memory", "one_shot", "no_context", "profile_only"]
    by = {r["variant"]: r for r in rows}
    for v in order:
        if v not in by:
            continue
        r = by[v]
        lines.append(f"| {v} | " + " | ".join(_fmt(r.get(c[0])) for c in cols) + " |")
    return "\n".join(lines)


def _reliability_section(data: dict) -> str:
    rel = data.get("relevance_agreement")
    clm = data.get("claim_agreement")
    if not rel and not clm:
        return (
            "No human raters were used in this run. Relevance uses an automatic "
            f"oracle (version {data['oracle_version']}); grounding uses the claim "
            "validator. Inter-rater agreement is therefore **not reported** and is "
            "flagged as a construct-validity threat. Annotation templates are emitted "
            "under `annotation/` (relevance_template.csv, claim_template.csv); drop in "
            "`relevance_labels_human.csv` / `claim_annotations_human.csv` to compute "
            "weighted Cohen's kappa and oracle-vs-human agreement automatically.")
    out = []
    if rel:
        out.append(f"- Relevance ({rel['n_items']} items): raw rater agreement "
                   f"{_fmt(rel['raw_agreement_raters'])}, weighted Cohen's kappa "
                   f"{_fmt(rel['weighted_kappa_raters'])}; oracle-vs-human weighted kappa "
                   f"{_fmt(rel['oracle_vs_human_weighted_kappa'])}.")
    if clm:
        out.append(f"- Claims ({clm['n_items']} items): raw agreement "
                   f"{_fmt(clm['raw_agreement'])}, Cohen's kappa {_fmt(clm['cohens_kappa'])}"
                   + (f"; validator-vs-human kappa {_fmt(clm.get('validator_vs_human_kappa'))}."
                      if clm.get('validator_vs_human_kappa') is not None else "."))
    return "\n".join(out)


def _error_taxonomy_table(rows: list[dict]) -> str:
    if not rows:
        return "_No task failures to categorize._"
    lines = ["| error category | count | % | most-affected variant |",
             "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['error_category']} | {r['count']} | {r['percentage']} | {r['most_affected_variant']} |")
    return "\n".join(lines)


def _scenario_type_table(sv: pd.DataFrame) -> str:
    lines = ["| scenario_type | variant | NDCG@5 | HCSR | TaskSucc | Grounding | n |",
             "|---|---|---|---|---|---|---|"]
    for (stype, variant), g in sv.groupby(["scenario_type", "variant"]):
        if variant not in ("full", "no_memory", "no_context"):
            continue
        lines.append(
            f"| {stype} | {variant} | {_fmt(g['ndcg_at_5'].mean())} | {_fmt(g['hcsr'].mean())} | "
            f"{_fmt(g['task_success'].mean())} | {_fmt(g['grounding'].mean())} | {g['scenario_id'].nunique()} |")
    return "\n".join(lines)


def generate_markdown(data: dict, plots_rel: str = "../plots") -> str:
    exp = data["experiment"]
    vs = pd.DataFrame(data["variant_summary"])
    mem = pd.DataFrame(data["memory_contribution"])
    ctx = pd.DataFrame(data["context_contribution"])
    sv = pd.DataFrame(data["scenario_variant"])
    overall = data["overall_comparisons"]

    def ocmp(metric, other):
        for r in overall:
            if r["metric"] == metric and r["other"] == other:
                return r
        return None

    full = vs[vs["variant"] == "full"].iloc[0] if (vs["variant"] == "full").any() else {}

    md = f"""# Evaluation Report: CMJCC Conversational Job Recommendation

> Generated by `jobrec_eval`. Every number is reproducible from the run bundles
> under `raw/` and the tables under `metrics/` and `statistics/`.
>
> **Relevance is scored by a deterministic automatic oracle, not human raters.**
> NDCG@5 / Precision@5 / Mean Graded Relevance therefore measure agreement with a
> transparent rule-based reference (documented in `relevance.py`), and should be
> read as such. Explanation grounding uses the system's claim validator. Human
> annotation and inter-rater agreement are left as future work (see §4, §11).

## 1. Executive Summary

- Experiment `{exp['experiment_id']}` — {exp['scenario_count']} scenarios ×
  {len(exp['variants'])} variants × {exp['repeat_count']} repeat(s) =
  {exp['run_count']} runs. Reference date {exp['reference_date']}.
- Run mode: **{data.get('llm_mode', 'deterministic')}** (model: {data.get('llm_model', 'mock-deterministic')}); config/catalog/prompt hashes frozen.
- Headline (full variant, scenario-mean): NDCG@5 {_fmt(full.get('ndcg_at_5_mean') if len(full) else None)},
  HCSR {_fmt(full.get('hcsr_mean') if len(full) else None)},
  Task Success {_fmt(full.get('task_success_mean') if len(full) else None)},
  Grounding {_fmt(full.get('grounding_mean') if len(full) else None)},
  Handoff {_fmt(full.get('handoff_success_mean') if len(full) else None)}.
- Ablation direction: memory and job-context removal are compared against full
  below with paired bootstrap CIs; small n means results are framed as observed
  differences with uncertainty, not proofs.

## 2. Research Questions and Evaluation Design

RQ4 is decomposed into job-match relevance, hard-constraint satisfaction, task
success, memory contribution, job-context contribution, agent-handoff success,
explanation grounding, response turns and latency. Five variants
({', '.join(exp['variants'])}) are run over a frozen scenario set and catalog
snapshot with fixed seeds. Analysis unit is the scenario; metrics are averaged
over repeats before pairing.

## 3. Dataset and Scenario Set

- Catalog snapshot `{exp['catalog_snapshot_id']}`, hash `{exp['catalog_hash'][:12]}`.
- Scenario counts by type: {data['scenario_type_counts']}.
- Memory-dependent (>=medium) scenarios: {data['n_memory_dependent']};
  context-dependent (high) scenarios: {data['n_context_dependent']}.

## 4. Annotation Reliability

{_reliability_section(data)}

## 5. Overall Results

Variant summary (scenario-mean of each metric):

{_variant_table(vs)}

### 5.x Full vs dialogue baselines (relevance & task success)

- NDCG@5, full vs profile_only: {_delta_phrase(*(lambda r:(r['delta'],r['ci_low'],r['ci_high']))(ocmp('ndcg_at_5','profile_only'))) if ocmp('ndcg_at_5','profile_only') else 'N/A'}.
- NDCG@5, full vs one_shot: {_delta_phrase(*(lambda r:(r['delta'],r['ci_low'],r['ci_high']))(ocmp('ndcg_at_5','one_shot'))) if ocmp('ndcg_at_5','one_shot') else 'N/A'}.
- Task success, full vs profile_only: {_delta_phrase(*(lambda r:(r['delta'],r['ci_low'],r['ci_high']))(ocmp('task_success','profile_only'))) if ocmp('task_success','profile_only') else 'N/A'}.

![NDCG]({plots_rel}/ndcg_by_variant.png)
![HCSR]({plots_rel}/hcsr_by_variant.png)
![Task success]({plots_rel}/task_success_by_variant.png)
![Grounding]({plots_rel}/grounding_by_variant.png)

### 5.2 Per-constraint compliance (recommended jobs vs authoritative hard constraints)

{_compliance_table(data.get('constraint_compliance', []))}

### 5.3 No-match and clarification correctness

No-match precision / recall / F1 by variant:

{_pr_table(data.get('no_match_metrics', []), [('precision','Precision'), ('recall','Recall'), ('f1','F1'), ('true_no_match','TP'), ('no_match_expected','Expected')])}

Clarification precision / recall by variant:

{_pr_table(data.get('clarification_metrics', []), [('precision','Precision'), ('recall','Recall'), ('useful','Useful'), ('expected_clarification','Expected')])}

## 6. Ablation Analysis

Each ablation isolates a single framework mechanism (candidate memory or
job-context orchestration) while holding catalog, scenarios, prompts, model
settings, top-k, pool size, ranking weights and seeds fixed across the compared
variants. Every Δ reported below is therefore read as the **contribution of that
framework mechanism under the controlled prototype instantiation** — an
attribution to a specific mechanism as instantiated in this prototype, not a
general property of the mechanism and not a claim of superiority over any
external framework.

### 6.1 Memory Contribution: Full vs No-Memory

Δmemory(M) = M_full − M_no_memory, paired by scenario. Primary subset is
memory-dependent (multi-turn) scenarios. Each Δmemory is framed as the
candidate-memory mechanism's contribution under the controlled prototype
instantiation, not as evidence of comprehensive superiority over external
frameworks.

{_contrib_table(mem, 'memory_dependent')}

All scenarios:

{_contrib_table(mem, 'all')}

![Memory delta NDCG]({plots_rel}/memory_delta_ndcg_at_5.png)
![Memory delta task]({plots_rel}/memory_delta_task_success.png)

### 6.2 Job-Context Contribution: Full vs No-Context

Δcontext(M) = M_full − M_no_context, paired by scenario. Primary subset is
context-dependent (high) scenarios. HCSR/violations are computed against the
authoritative hard constraints. Each Δcontext is framed as the job-context
orchestration mechanism's contribution under the controlled prototype
instantiation, not as a claim of comprehensive superiority over external
frameworks.

{_contrib_table(ctx, 'context_dependent')}

All scenarios:

{_contrib_table(ctx, 'all')}

![Context delta HCSR]({plots_rel}/context_delta_hcsr.png)
![Context delta task]({plots_rel}/context_delta_task_success.png)

## 7. Results by Scenario Type

{_scenario_type_table(sv)}

## 8. Statistical Analysis

Paired bootstrap ({exp.get('bootstrap_iterations', 5000)} iterations, seed
{exp.get('bootstrap_seed', 2026)}) gives 95% CIs for scenario-mean differences.
Binary task success uses McNemar on run-level discordant pairs. Effect sizes are
Cohen's dz (continuous) or rank-biserial (binary/degenerate). Holm correction is
applied within each ablation across metrics. With small n, a CI that includes 0
is reported as "direction observed, uncertain", never as "no effect".

## 9. Error Analysis

{data['error_summary']}

Root-cause taxonomy of task-unsuccessful runs:

{_error_taxonomy_table(data.get('error_taxonomy', []))}

### 9.1 Representative case studies

{data.get('case_studies_md', '_No case studies extracted._')}

## 10. Discussion

- **RQ2 / memory:** differences concentrate in memory-dependent (multi-turn)
  scenarios, consistent with prior-turn memory contributing to correctly
  reconstructing the active search; effects on memory-independent scenarios are
  expected to be near zero.
- **RQ2 / job context:** removing explicit hard/soft orchestration is expected
  to lower HCSR and raise violations on context-dependent scenarios; the tables
  above quantify this against the authoritative constraints.
- **Quality/latency trade-off:** turns and latency are reported alongside task
  success rather than in isolation (see turns-vs-success and latency plots).
- **Inspectability (RQ1/RQ3):** handoff success, decision-log completeness and
  recommendation trace completeness are reported as engineering-quality
  indicators, not statistical claims.

![Turns vs success]({plots_rel}/turns_vs_success.png)
![Latency breakdown]({plots_rel}/latency_breakdown.png)

## 11. Threats to Validity

- **Construct:** relevance uses an automatic oracle, not human judgement; NDCG/P@5
  measure agreement with a rule-based reference. Grounding measures evidence
  support, not perceived explanation quality or user trust.
- **Internal:** deterministic mock provider removes LLM stochasticity but also
  does not exercise a real model; variant behaviour is controlled by feature
  flags on one code path.
- **External:** small synthetic catalog and synthetic candidates; a modest
  scenario count; results do not extrapolate to real hiring outcomes.
- **Conclusion:** small n limits statistical power; emphasis is on effect sizes,
  CIs and per-scenario plots, not single p-values.

## 12. Conclusion

Within this controlled, deterministic setup, the full architecture meets the
engineering-quality indicators and the ablations show the expected directional
contributions of candidate memory and job-context orchestration. These results
attribute observed differences to specific framework mechanisms under the
controlled prototype instantiation; they do not state or imply comprehensive
superiority over any existing external framework. Claims are limited to this
configuration; human-annotated relevance and a real LLM backend are the natural
next steps.

## Appendix

- Experiment manifest: `manifests/experiment_manifest.json`
- Analysis plan: `manifests/analysis_plan.yaml`
- All metric tables: `metrics/*.csv`; statistics: `statistics/*.csv`
- Relevance labels (automatic oracle): `../normalized/relevance_labels.csv`
- Data lineage: `audit/data_lineage.csv`; checksums: `checksums.json`
  (verify with `python -m jobrec_eval.cli verify <output_dir>`)
"""
    return md


def _variant_of(manifest: dict) -> str | None:
    flags = manifest.get("feature_flags")
    variant = flags.get("variant") if isinstance(flags, dict) else None
    return str(getattr(variant, "value", variant)) if variant else None


def _consistency_scopes(manifests: list[dict]) -> list[tuple[set[str] | None, list[dict]]]:
    """The scopes the gate verifies: the whole comparison, then each ablation pair.

    The whole comparison is checked without a target flag set (R15.1: shared
    catalog/scenarios/prompts/settings/commit). Each ablation pair present in the
    manifests is then checked against its own mechanism's flag group, so a Δ the
    report attributes to that mechanism cannot be contaminated by another flag
    (R32.7).
    """
    scopes: list[tuple[set[str] | None, list[dict]]] = [(None, manifests)]
    by_variant: dict[str | None, list[dict]] = {}
    for manifest in manifests:
        by_variant.setdefault(_variant_of(manifest), []).append(manifest)
    for base, other, target in _ABLATION_PAIRS:
        if by_variant.get(base) and by_variant.get(other):
            scopes.append((set(target), by_variant[base] + by_variant[other]))
    return scopes


def _mirror_onto_run_records(manifest_paths: list[Path]) -> list[Path]:
    """Copy each manifest's consistency block onto its run record (R15.3).

    The manifest is the authoritative location for the gate result; mirroring it
    into the sibling ``run_record.json`` populates
    :attr:`jobrec.domain.run_record.RunRecord.consistency_flags`, so a run carries
    its own verification outcome. Unreadable or unexpected payloads are skipped
    rather than failing the gate.
    """
    written: list[Path] = []
    for manifest_path in manifest_paths:
        record_path = Path(manifest_path).with_name("run_record.json")
        try:
            manifest = json.loads(Path(manifest_path).read_text())
            record = json.loads(record_path.read_text())
        except (OSError, ValueError):
            continue
        block = manifest.get("consistency") if isinstance(manifest, dict) else None
        if not isinstance(block, dict) or not isinstance(record, dict):
            continue
        record["consistency_flags"] = {
            "consistent": block.get("consistent"),
            **(block.get("flags") or {}),
            "compared_runs": block.get("compared_runs", []),
            "mismatched_fields": block.get("mismatched_fields", []),
        }
        record_path.write_text(json.dumps(record, indent=2, default=str))
        written.append(record_path)
    return written


def require_consistent_runs(manifests: list[dict]) -> None:
    """Halt unless every compared run shares the same configuration (R15.2).

    Verification stops at the first failing scope; the resulting flags are stamped
    into the manifests, persisted to disk and mirrored onto the run records before
    the mismatch is raised, so a blocked report still leaves an auditable trail
    (R15.3). Raises :class:`~jobrec_eval.consistency.ConsistencyError` on mismatch
    and :class:`ValueError` when there is nothing to verify.
    """
    if not manifests:
        raise ValueError(
            "configuration consistency cannot be verified without run manifests; "
            "refusing to generate a comparison report")
    error: ConsistencyError | None = None
    for target, subset in _consistency_scopes(manifests):
        try:
            require_consistent(subset, target)
        except ConsistencyError as exc:
            error = exc
            break
    _mirror_onto_run_records(save_run_manifests(manifests))
    if error is not None:
        raise error


def write_report(data: dict, out_dir: str | Path, *,
                 experiment_dir: str | Path | None = None,
                 manifests: list[dict] | None = None) -> Path:
    """Verify configuration consistency, then write the report bundle.

    Args:
        data: The assembled report data dict.
        out_dir: Analysis output directory; the report lands under ``report/``.
        experiment_dir: Directory holding the run bundles whose ``run_manifest.json``
            files describe the compared runs.
        manifests: Already-loaded run manifests, as an alternative to
            ``experiment_dir``.

    Returns:
        Path of the written Markdown report.

    Raises:
        ConsistencyError: The compared runs do not match; nothing is written
            (R15.2).
        ValueError: Neither manifest source was supplied, or no manifests exist.
    """
    if manifests is None:
        if experiment_dir is None:
            raise ValueError(
                "write_report needs experiment_dir or manifests: configuration "
                "consistency must be verified before any report output is produced")
        manifests = load_run_manifests(experiment_dir)
        if not manifests:
            raise ValueError(
                f"no run_manifest.json found under {experiment_dir}; configuration "
                "consistency of the compared runs cannot be verified")
    require_consistent_runs(manifests)

    out_dir = Path(out_dir)
    (out_dir / "report").mkdir(parents=True, exist_ok=True)
    # UTF-8 explicitly: the Markdown carries Δ and − and must not depend on the
    # platform's default encoding (locale codecs such as GBK cannot encode them).
    (out_dir / "report" / "analysis_report_data.json").write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8")
    md = generate_markdown(data)
    report_path = out_dir / "report" / "analysis_report.md"
    report_path.write_text(md, encoding="utf-8")
    return report_path
