"""Evaluation pipeline CLI.

    python -m jobrec_eval.cli pipeline --config configs/experiment_full.yaml \
        --scenarios evaluation/data/scenarios.jsonl --repeats 1

Sub-stages (validate/run/normalize/compute-metrics/compute-statistics/plot/
report/audit) are also exposed, but `pipeline` runs them end to end.

Artifact integrity is checked with one command (R16.2):

    python -m jobrec_eval.cli verify evaluation/outputs/<experiment_id>

Input data quality is checked before a run (R17); it exits non-zero on any
error-severity violation:

    python -m jobrec_eval.cli validate --scenarios evaluation/data/scenarios.jsonl \
        --catalog data/processed/jobs.jsonl --report-dir evaluation/outputs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from jobrec.catalog import catalog_hash, load_catalog
from jobrec.config import load_config
from jobrec.evaluation.checksums import (
    CHECKSUMS_FILENAME,
    MissingChecksumsError,
    verify_checksums,
    write_checksums,
)
from jobrec.evaluation.experiment_runner import ExperimentRunner

from . import EVAL_VERSION
from .annotation import (
    claim_agreement,
    export_claim_template,
    export_relevance_template,
    relevance_agreement,
)
from .casestudies import error_taxonomy, extract_cases, render_cases_md
from .data_quality import (
    DATA_QUALITY_REPORT_FILENAME,
    validate_dataset,
    write_data_quality_report,
)
from .loaders import load_bundles, normalize
from .metrics import (
    MetricsComputer,
    aggregate_scenario_variant,
    latency_percentiles,
    variant_summary,
)
from .metrics_extra import (
    clarification_metrics,
    extraction_source_metrics,
    no_match_metrics,
    per_constraint_compliance,
    retrieval_metrics,
    topk_contribution_table,
)
from .plots import plot_all
from .relevance import ORACLE_VERSION, build_references, grade_catalog, grade_lookup
from .report import write_report
from .scenarios import load_scenarios
from .statistics import compare, contribution_table

VARIANTS = ["full", "profile_only", "one_shot", "no_memory", "no_context"]
PRIMARY = ["ndcg_at_5", "hcsr", "task_success", "grounding", "mean_violation_count", "turn_count"]


def _write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def run_pipeline(config_path: str, scenarios_path: str, catalog_path: str,
                 out_root: str, repeats: int, experiment_dir: str | None,
                 bootstrap_iters: int, bootstrap_seed: int,
                 variants: list[str] | None = None) -> dict:
    variants = variants or VARIANTS
    cfg = load_config(config_path, base_dir=str(Path(config_path).parent))
    cfg.experiment.repeat_count = repeats
    catalog = load_catalog(catalog_path)
    scenarios = load_scenarios(scenarios_path)

    # ---- Stage 2: run experiments (or reuse) ----------------------------
    runs_root = Path(out_root) / "_runs"
    if experiment_dir:
        exp_dir = Path(experiment_dir)
        runner_manifest = json.loads((exp_dir / "experiment_manifest.json").read_text())
    else:
        runner = ExperimentRunner(cfg, catalog_path, scenarios_path, out_dir=str(runs_root))
        runner_manifest = runner.run(variants)
        exp_dir = Path(runner_manifest["experiment_dir"])

    experiment_id = runner_manifest["experiment_id"]
    out = Path(out_root) / experiment_id
    (out / "normalized").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "statistics").mkdir(parents=True, exist_ok=True)
    (out / "manifests").mkdir(parents=True, exist_ok=True)
    (out / "audit").mkdir(parents=True, exist_ok=True)

    # ---- Stage 3: normalize --------------------------------------------
    bundles = load_bundles(exp_dir)
    tables = normalize(bundles)
    for name, df in tables.items():
        _write_csv(df, out / "normalized" / f"{name}.csv")

    # ---- relevance oracle ----------------------------------------------
    references = build_references(bundles)
    labels = grade_catalog(catalog, references, cfg)
    _write_csv(labels, out / "normalized" / "relevance_labels.csv")

    # ---- Stage 4: metrics ----------------------------------------------
    mc = MetricsComputer(cfg, catalog, references, labels, scenarios,
                         relevance_threshold=2, top_k=cfg.experiment.top_k)
    run_metrics = mc.run_metrics(bundles)
    sv = aggregate_scenario_variant(run_metrics)
    vsum = variant_summary(sv)
    latpct = latency_percentiles(tables["component_latency"])
    _write_csv(run_metrics, out / "metrics" / "run_metrics.csv")
    _write_csv(sv, out / "metrics" / "scenario_variant_metrics.csv")
    _write_csv(vsum, out / "metrics" / "variant_summary.csv")
    _write_csv(latpct, out / "metrics" / "latency_metrics.csv")

    # scenario type summary
    type_summary = (sv.groupby(["scenario_type", "variant"])[["ndcg_at_5", "hcsr", "task_success", "grounding"]]
                    .mean().reset_index())
    _write_csv(type_summary, out / "metrics" / "scenario_type_summary.csv")

    # ---- Stage 5: statistics -------------------------------------------
    mem_dep = {s.scenario_id for s in scenarios.values() if s.memory_dependency in ("medium", "high")}
    ctx_dep = {s.scenario_id for s in scenarios.values() if s.context_dependency == "high"}

    mem_tbl = contribution_table(sv, run_metrics, PRIMARY, "no_memory",
                                 {"all": None, "memory_dependent": mem_dep},
                                 bootstrap_iters, bootstrap_seed)
    ctx_tbl = contribution_table(sv, run_metrics, PRIMARY, "no_context",
                                 {"all": None, "context_dependent": ctx_dep},
                                 bootstrap_iters, bootstrap_seed)
    _write_csv(mem_tbl, out / "metrics" / "memory_contribution.csv")
    _write_csv(ctx_tbl, out / "metrics" / "context_contribution.csv")

    overall = []
    for other in [v for v in ["profile_only", "one_shot", "no_memory", "no_context"] if v in variants]:
        for metric in ["ndcg_at_5", "task_success", "hcsr"]:
            overall.append(compare(sv, run_metrics, metric, "full", other,
                                   bootstrap_iters, bootstrap_seed))
    pd.DataFrame(overall).to_csv(out / "statistics" / "paired_comparisons.csv", index=False)

    # ---- diagnostic metrics (per-constraint compliance, no-match, clarify) ---
    compliance = per_constraint_compliance(mc, bundles)
    nomatch = no_match_metrics(run_metrics)
    clarify = clarification_metrics(run_metrics)
    extraction_sources = extraction_source_metrics(bundles, scenarios)
    retrieval = retrieval_metrics(bundles, labels, relevance_threshold=2)
    topk_contribution = topk_contribution_table(bundles, top_k=cfg.experiment.top_k)
    _write_csv(compliance, out / "metrics" / "constraint_compliance.csv")
    _write_csv(nomatch, out / "metrics" / "no_match_metrics.csv")
    _write_csv(clarify, out / "metrics" / "clarification_metrics.csv")
    _write_csv(extraction_sources, out / "metrics" / "extraction_source_metrics.csv")
    _write_csv(retrieval, out / "metrics" / "retrieval_metrics.csv")
    _write_csv(topk_contribution, out / "metrics" / "topk_contribution.csv")

    # ---- case studies + error taxonomy ---------------------------------
    grade = grade_lookup(labels)
    cases = extract_cases(bundles, run_metrics, grade)
    cases_md = render_cases_md(cases)
    err_tax = error_taxonomy(run_metrics)
    _write_csv(err_tax, out / "metrics" / "error_taxonomy.csv")

    # ---- annotation templates + human agreement (if provided) ----------
    ann_dir = out / "annotation"
    ann_dir.mkdir(parents=True, exist_ok=True)
    export_relevance_template(tables["recommendations"], labels, ann_dir / "relevance_template.csv")
    export_claim_template(tables["claims"], ann_dir / "claim_template.csv")
    scen_dir = Path(scenarios_path).parent
    rel_agree = relevance_agreement(scen_dir / "relevance_labels_human.csv", labels)
    clm_agree = claim_agreement(scen_dir / "claim_annotations_human.csv")

    # ---- Stage 6: plots -------------------------------------------------
    plot_all(sv, latpct, out / "plots")

    # ---- error summary --------------------------------------------------
    failures = run_metrics[~run_metrics["success_run"]]
    task_fail = run_metrics[run_metrics["task_success"] == 0]
    error_summary = (f"Runs: {len(run_metrics)}; system failures: {len(failures)}; "
                     f"task-unsuccessful runs: {len(task_fail)}. "
                     f"Task failures by variant: "
                     f"{task_fail.groupby('variant').size().to_dict()}.")

    # ---- assemble report data ------------------------------------------
    scenario_type_counts = {t: int((pd.Series([s.scenario_type for s in scenarios.values()]) == t).sum())
                            for t in sorted({s.scenario_type for s in scenarios.values()})}
    data = {
        "experiment": {
            "experiment_id": experiment_id,
            "reference_date": cfg.project.reference_date,
            "catalog_snapshot_id": runner_manifest.get("config_hash", "") and
                                   json.loads((Path(catalog_path).parent / "catalog_manifest.json").read_text()).get("catalog_snapshot_id", "catalog"),
            "catalog_hash": catalog_hash(catalog),
            "variants": variants,
            "scenario_count": len(scenarios),
            "repeat_count": repeats,
            "run_count": len(run_metrics),
            "bootstrap_iterations": bootstrap_iters,
            "bootstrap_seed": bootstrap_seed,
            "eval_version": EVAL_VERSION,
        },
        "oracle_version": ORACLE_VERSION,
        "scenario_type_counts": scenario_type_counts,
        "n_memory_dependent": len(mem_dep),
        "n_context_dependent": len(ctx_dep),
        "variant_summary": vsum.to_dict(orient="records"),
        "scenario_variant": sv.to_dict(orient="records"),
        "memory_contribution": mem_tbl.to_dict(orient="records"),
        "context_contribution": ctx_tbl.to_dict(orient="records"),
        "overall_comparisons": overall,
        "error_summary": error_summary,
        "constraint_compliance": compliance.to_dict(orient="records"),
        "no_match_metrics": nomatch.to_dict(orient="records"),
        "clarification_metrics": clarify.to_dict(orient="records"),
        "extraction_source_metrics": extraction_sources.to_dict(orient="records"),
        "retrieval_metrics": retrieval.to_dict(orient="records"),
        "topk_contribution": topk_contribution.to_dict(orient="records"),
        "error_taxonomy": err_tax.to_dict(orient="records"),
        "case_studies_md": cases_md,
        "relevance_agreement": rel_agree,
        "claim_agreement": clm_agree,
        "llm_mode": cfg.llm.mode.value,
        "llm_model": (cfg.llm.provider if cfg.llm.mode.value != "deterministic" else "mock-deterministic"),
    }

    # ---- manifests ------------------------------------------------------
    (out / "manifests" / "experiment_manifest.json").write_text(json.dumps({
        **runner_manifest, "eval_version": EVAL_VERSION, "oracle_version": ORACLE_VERSION,
        "catalog_hash": catalog_hash(catalog),
    }, indent=2))
    (out / "manifests" / "analysis_plan.yaml").write_text(yaml.safe_dump({
        "primary_outcomes": PRIMARY,
        "primary_comparisons": [["full", v] for v in ["profile_only", "one_shot", "no_memory", "no_context"]],
        "confidence_level": 0.95, "bootstrap_iterations": bootstrap_iters,
        "bootstrap_seed": bootstrap_seed, "binary_relevance_threshold": 2,
        "p_value_adjustment": "holm", "relevance_source": "automatic_oracle",
    }, sort_keys=False))

    # Report output is gated on configuration consistency of the compared runs
    # (R15.2/R32.7): on mismatch write_report raises and nothing is written.
    report_path = write_report(data, out, experiment_dir=exp_dir)

    # ---- audit ----------------------------------------------------------
    _write_audit(out, run_metrics, references, scenarios)

    return {"experiment_id": experiment_id, "out_dir": str(out), "report": str(report_path),
            "runs": len(run_metrics)}


def _write_audit(out: Path, run_metrics: pd.DataFrame, references, scenarios):
    invalid = run_metrics[~run_metrics["success_run"]][["run_id", "scenario_id", "variant"]]
    invalid = invalid.assign(reason="run_record.success == false")
    _write_csv(invalid, out / "audit" / "invalid_runs.csv")

    missing_ref = [{"scenario_id": s} for s in scenarios if s not in references]
    _write_csv(pd.DataFrame(missing_ref or [{"scenario_id": ""}]), out / "audit" / "missing_reference_scenarios.csv")

    lineage = [
        {"report_section": "5", "metric_name": "variant_summary", "source_file": "metrics/variant_summary.csv"},
        {"report_section": "6.1", "metric_name": "memory_contribution", "source_file": "metrics/memory_contribution.csv"},
        {"report_section": "6.2", "metric_name": "context_contribution", "source_file": "metrics/context_contribution.csv"},
        {"report_section": "8", "metric_name": "paired_comparisons", "source_file": "statistics/paired_comparisons.csv"},
        {"report_section": "9.x", "metric_name": "relevance_labels", "source_file": "normalized/relevance_labels.csv"},
    ]
    _write_csv(pd.DataFrame(lineage), out / "audit" / "data_lineage.csv")

    # Unified checksum manifest over EVERY artifact in the output directory --
    # normalized data, metrics, statistics, plots, report and audit tables
    # (R16.1). Written last so it covers everything the pipeline produced, and
    # verifiable with `python -m jobrec_eval.cli verify <dir>`.
    write_checksums(out)


def run_validate(
    scenarios_path: str,
    catalog_path: str,
    *,
    config_path: str | None = None,
    relevance_labels: str | None = None,
    report_dir: str | None = None,
    verify_no_match: bool = True,
) -> int:
    """Validate the catalog + scenario set and return an exit code (R17).

    Loading the catalog through :func:`jobrec.catalog.load_catalog` keeps the
    schema check that ``scripts/validate_catalog.py`` performs, then
    :func:`~jobrec_eval.data_quality.validate_dataset` adds the semantic checks.
    Every finding is printed with its offending identifier; the report is written
    when ``report_dir`` is given. Returns non-zero when an ``error``-severity
    violation was found, so CI can gate on it.
    """
    cfg = (load_config(config_path, base_dir=str(Path(config_path).parent))
           if config_path else load_config())
    scenarios = load_scenarios(scenarios_path)
    catalog = load_catalog(catalog_path)

    labels = relevance_labels
    if labels is None:
        default_labels = Path(scenarios_path).parent / "relevance_labels.csv"
        labels = str(default_labels) if default_labels.is_file() else None

    report = validate_dataset(
        catalog, scenarios, config=cfg, relevance_labels=labels,
        verify_no_match=verify_no_match,
    )
    print(report.summary())
    for finding in report.findings:
        print(f"  - {finding.describe()}")
    if report_dir:
        path = write_data_quality_report(report, report_dir)
        print(f"report: {path}")
    return 0 if report.ok else 1


def run_verify(artifact_dir: str, *, report_untracked: bool = True) -> int:
    """Verify a directory against its ``checksums.json`` and return an exit code.

    Returns 0 when every recorded artifact still matches, and non-zero otherwise:
    2 when the directory or its manifest is unusable, 1 when at least one
    artifact mismatches. Each offending artifact is printed by name (R16.2/16.3).
    """
    root = Path(artifact_dir)
    try:
        findings = verify_checksums(root, report_untracked=report_untracked)
    except MissingChecksumsError:
        print(f"FAIL: no {CHECKSUMS_FILENAME} found in {root}")
        return 2
    except (NotADirectoryError, ValueError) as exc:
        print(f"FAIL: {exc}")
        return 2

    if not findings:
        print(f"OK: all artifacts in {root} match {CHECKSUMS_FILENAME}")
        return 0

    print(f"FAIL: {len(findings)} artifact mismatch(es) in {root}")
    for finding in findings:
        print(f"  - {finding.describe()}")
    return 1


def main() -> None:
    parser = argparse.ArgumentParser(description="CMJCC evaluation pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("pipeline", help="run the full evaluation end-to-end")
    p.add_argument("--config", default="configs/experiment_full.yaml")
    p.add_argument("--scenarios", default="evaluation/data/scenarios.jsonl")
    p.add_argument("--catalog", default="data/processed/jobs.jsonl")
    p.add_argument("--out-root", default="evaluation/outputs")
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--experiment-dir", default=None, help="reuse an existing run dir")
    p.add_argument("--bootstrap-iters", type=int, default=5000)
    p.add_argument("--bootstrap-seed", type=int, default=2026)
    p.add_argument("--variants", default=None,
                   help="comma-separated subset of variants (default: all five)")

    v = sub.add_parser("validate", help="validate scenarios + catalog (data quality, R17)")
    v.add_argument("--scenarios", default="evaluation/data/scenarios.jsonl")
    v.add_argument("--catalog", default="data/processed/jobs.jsonl")
    v.add_argument("--config", default=None,
                   help="config supplying the reference date and constraint policies")
    v.add_argument("--relevance-labels", default=None,
                   help="relevance-label CSV (default: relevance_labels.csv beside --scenarios)")
    v.add_argument("--report-dir", default=None,
                   help=f"write {DATA_QUALITY_REPORT_FILENAME} into this directory")
    v.add_argument("--skip-no-match-check", action="store_true",
                   help="skip replaying no-match scenarios against the catalog")

    ver = sub.add_parser("verify", help=f"verify artifacts against {CHECKSUMS_FILENAME}")
    ver.add_argument("artifact_dir",
                     help="experiment or evaluation output directory to verify")
    ver.add_argument("--allow-untracked", action="store_true",
                     help="ignore files added after the manifest was written")

    args = parser.parse_args()
    if args.command == "pipeline":
        variants = args.variants.split(",") if args.variants else None
        result = run_pipeline(args.config, args.scenarios, args.catalog, args.out_root,
                              args.repeats, args.experiment_dir, args.bootstrap_iters,
                              args.bootstrap_seed, variants=variants)
        print(json.dumps(result, indent=2))
    elif args.command == "validate":
        code = run_validate(
            args.scenarios, args.catalog, config_path=args.config,
            relevance_labels=args.relevance_labels, report_dir=args.report_dir,
            verify_no_match=not args.skip_no_match_check,
        )
        if code != 0:
            raise SystemExit(code)
    elif args.command == "verify":
        code = run_verify(args.artifact_dir,
                          report_untracked=not args.allow_untracked)
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
