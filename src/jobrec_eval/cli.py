"""Evaluation pipeline CLI.

    python -m jobrec_eval.cli pipeline --config configs/experiment_full.yaml \
        --scenarios evaluation/data/scenarios.jsonl --repeats 1

Sub-stages (validate/run/normalize/compute-metrics/compute-statistics/plot/
report/audit) are also exposed, but `pipeline` runs them end to end. The pipeline also
saves a `data_quality_report.json` of the frozen inputs into the experiment artifacts
(R17.3) and reports the failure-path rates in a separate fault-injection section
(R10.8/10.9); ablation Δ values are computed over two independently Holm-corrected
outcome families, :data:`PRIMARY` and :data:`SECONDARY`.

Relevance labels come from one of two sources, chosen with `--relevance-source`
(checklist item 10):

    python -m jobrec_eval.cli pipeline --relevance-source human ...

`oracle` (default) uses the deterministic automatic oracle; `human` uses the adjudicated
two-rater labels in `relevance_labels_human.csv` beside `--scenarios` and FAILS when they
are absent rather than falling back to the oracle. Whichever is selected, both are
computed and compared in `metrics/relevance_source_comparison.csv`, the report states
which one produced the ranking metrics, and `manifests/analysis_plan.yaml` records the
real source plus the human file's path and content hash.

The experiment id is content-addressed over the inputs AND the source code that produced
them, so a run of changed code never lands in an older run's directory
(:mod:`jobrec.evaluation.experiment_identity`). Re-running unchanged code resolves to the
same directory on purpose, and overwriting an existing complete experiment must be asked
for:

    python -m jobrec_eval.cli pipeline --allow-overwrite ...

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
from jobrec.evaluation.experiment_identity import (
    EXPERIMENT_MANIFEST_FILENAME,
    ExperimentOverwriteError,
    code_identity,
    guard_output_dir,
)
from jobrec.evaluation.experiment_runner import ExperimentRunner

from . import EVAL_VERSION
from .annotation import (
    AdjudicatedRelevanceLabels,
    MissingAdjudicatedLabelsError,
    claim_agreement,
    export_claim_template,
    export_relevance_template,
    load_adjudicated_relevance_labels,
    relevance_agreement,
)
from .casestudies import error_taxonomy, extract_cases, render_cases_md
from .data_quality import (
    DATA_QUALITY_REPORT_FILENAME,
    validate_dataset,
    write_data_quality_report,
)
from .loaders import load_bundles, model_call_identities, normalize
from .metrics import (
    MetricsComputer,
    aggregate_scenario_variant,
    latency_percentiles,
    variant_summary,
)
from .metrics_extra import (
    clarification_efficiency,
    clarification_metrics,
    extraction_source_metrics,
    failure_metrics,
    no_match_metrics,
    per_constraint_compliance,
    retrieval_metrics,
    topk_contribution_table,
)
from .oracle_reference import (
    CANONICAL_ORACLE_FILENAME,
    CANONICAL_ORACLE_VERSION,
    load_or_build_canonical_references,
)
from .plots import plot_all
from .relevance import ORACLE_VERSION, grade_catalog, grade_lookup
from .report import write_report
from .scenarios import load_scenarios
from .statistics import compare, contribution_table

VARIANTS = ["full", "profile_only", "one_shot", "no_memory", "no_context"]

#: Pre-registered PRIMARY outcome family (``manifests/analysis_plan.yaml``
#: ``primary_outcomes``). Holm correction in :func:`~jobrec_eval.statistics.contribution_table`
#: is applied across the metrics of ONE call, within each scenario subset, so this list must
#: not grow: appending a metric would make every existing primary p-value more conservative
#: and silently change what the pre-registered plan means.
PRIMARY = ["ndcg_at_5", "hcsr", "task_success", "grounding", "mean_violation_count", "turn_count"]

#: SECONDARY outcome family: the two process measures (``manifests/analysis_plan.yaml``
#: ``secondary_outcomes``). Reported as a second, independently Holm-corrected family via a
#: separate :func:`~jobrec_eval.statistics.contribution_table` call, tagged with
#: ``family="secondary"`` in the same contribution CSV, so §6.3 gains its Δ rows while the
#: primary p-values in §6.1/§6.2 keep their exact values (R7.4, R32.4).
SECONDARY = ["response_turns", "clarification_efficiency"]

#: Column tagging which outcome family a contribution row belongs to, so the two families
#: are never mixed or double-counted when the frames are concatenated.
FAMILY_COLUMN = "family"

# ------------------------------------------------------------- relevance source
#: ``--relevance-source`` values, mapped to what ``manifests/analysis_plan.yaml`` records.
#: The plan records what was ACTUALLY used, never a hardcoded constant (checklist item 10).
RELEVANCE_SOURCE_ORACLE = "automatic_oracle"
RELEVANCE_SOURCE_HUMAN = "human_adjudicated"
RELEVANCE_SOURCES = {"oracle": RELEVANCE_SOURCE_ORACLE, "human": RELEVANCE_SOURCE_HUMAN}

#: Human label files the annotation tool exports, read from beside ``--scenarios``.
HUMAN_RELEVANCE_FILENAME = "relevance_labels_human.csv"
HUMAN_CLAIMS_FILENAME = "claim_annotations_human.csv"

#: Metrics derived from the relevance label table (``MetricsComputer.grade``), i.e. the
#: ones that change with ``--relevance-source``. Everything else in the pipeline is
#: label-independent.
RELEVANCE_METRICS = ["ndcg_at_5", "precision_at_5", "mean_graded_relevance"]

#: Columns of ``metrics/relevance_source_comparison.csv``: one row per variant x ranking
#: metric, with the value under each label source and their difference.
RELEVANCE_COMPARISON_COLUMNS = ["variant", "metric", "oracle", "human", "delta",
                                "n_oracle", "n_human"]

#: Retrieval recall keeps the AUTOMATIC ORACLE labels in both modes. Recall@pool needs the
#: full-catalog label universe (``|relevant ∩ pool| / |relevant|`` over every relevant job
#: in the catalog), and human judgements exist only for the pairs that were returned --
#: substituting them would shrink the denominator to the returned set and make recall
#: trivially near 1.000, measuring the pooling of the judgements rather than the retriever.
#: Recorded in the analysis plan as ``retrieval_recall_relevance_source``.
RETRIEVAL_LABEL_SOURCE = RELEVANCE_SOURCE_ORACLE


def _write_csv(df: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _write_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str),
                    encoding="utf-8")


def _opt_float(value) -> float | None:
    """A metric cell as ``float``, or ``None`` when it is missing (module convention)."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return float(value)


def _opt_int(value) -> int | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return int(value)


def select_relevance_labels(
    relevance_source: str, oracle_labels: pd.DataFrame,
    human: AdjudicatedRelevanceLabels | None,
) -> tuple[pd.DataFrame, str]:
    """The label table the primary metrics are computed from, plus what to record.

    ``relevance_source`` is the CLI spelling (``oracle`` / ``human``). The returned table
    is handed to :class:`~jobrec_eval.metrics.MetricsComputer` unchanged, which is the only
    place the choice has any effect -- there is one metric implementation for both sources.

    Raises:
        MissingAdjudicatedLabelsError: ``human`` was requested but no adjudicated labels
            are available. There is deliberately no silent fallback to the oracle: that
            would publish oracle numbers under a human-labels heading.
        ValueError: ``relevance_source`` is not a known source.
    """
    if relevance_source not in RELEVANCE_SOURCES:
        raise ValueError(f"unknown relevance source {relevance_source!r}; expected one of "
                         f"{sorted(RELEVANCE_SOURCES)}")
    if relevance_source == "oracle":
        return oracle_labels, RELEVANCE_SOURCE_ORACLE
    if human is None:
        raise MissingAdjudicatedLabelsError(
            f"--relevance-source human was requested but no adjudicated human relevance "
            f"labels are available: expected a {HUMAN_RELEVANCE_FILENAME} beside the "
            f"scenario file, carrying scenario_id, job_id, rater_1, rater_2 and a filled "
            f"'adjudicated' column. Refusing to fall back to the automatic oracle, which "
            f"would report oracle numbers under a human-labels heading.")
    return human.labels, RELEVANCE_SOURCE_HUMAN


def _ranking_variant_summary(cfg, catalog, references, label_table: pd.DataFrame,
                             scenarios, bundles, top_k: int) -> pd.DataFrame:
    """Variant summary for ONE label table, through the same MetricsComputer path.

    Used only to obtain the second source's ranking metrics for the side-by-side
    comparison; the metric definitions are not duplicated, the class is simply given the
    other label table.
    """
    computer = MetricsComputer(cfg, catalog, references, label_table, scenarios,
                               relevance_threshold=2, top_k=top_k)
    return variant_summary(aggregate_scenario_variant(computer.run_metrics(bundles)))


def relevance_source_comparison(oracle_summary: pd.DataFrame,
                                human_summary: pd.DataFrame | None) -> pd.DataFrame:
    """Oracle-vs-human ranking metrics per variant, with the human − oracle delta.

    One row per variant x metric over :data:`RELEVANCE_METRICS`, with columns
    :data:`RELEVANCE_COMPARISON_COLUMNS`. ``delta`` is ``human - oracle`` (the direction is
    stated in the report). When no adjudicated human labels exist, the ``human``, ``delta``
    and ``n_human`` cells are empty rather than imputed, so the table's shape is the same
    in both modes and an empty cell reads as unmeasured (checklist item 10).
    """
    oracle_by = {row["variant"]: row for _, row in oracle_summary.iterrows()}
    human_by = ({row["variant"]: row for _, row in human_summary.iterrows()}
                if human_summary is not None else {})
    ordered = [v for v in VARIANTS if v in oracle_by]
    ordered += sorted(set(oracle_by) - set(ordered))
    rows = []
    for variant in ordered:
        oracle_row = oracle_by[variant]
        human_row = human_by.get(variant)
        for metric in RELEVANCE_METRICS:
            oracle_value = _opt_float(oracle_row.get(f"{metric}_mean"))
            human_value = (_opt_float(human_row.get(f"{metric}_mean"))
                           if human_row is not None else None)
            rows.append({
                "variant": variant, "metric": metric,
                "oracle": oracle_value, "human": human_value,
                "delta": (human_value - oracle_value
                          if oracle_value is not None and human_value is not None else None),
                "n_oracle": _opt_int(oracle_row.get(f"{metric}_n")),
                "n_human": (_opt_int(human_row.get(f"{metric}_n"))
                            if human_row is not None else None),
            })
    return pd.DataFrame(rows, columns=RELEVANCE_COMPARISON_COLUMNS)


def _catalog_snapshot_id(catalog_path: str, runner_manifest: dict) -> str:
    """Snapshot id of the catalog beside ``catalog_path``, or ``"catalog"`` as a fallback.

    Read from the sibling ``catalog_manifest.json`` written by
    ``scripts/prepare_catalog.py``. A missing or unreadable manifest is not fatal: the id
    only labels the snapshot in the report and on the data-quality replay.
    """
    if not runner_manifest.get("config_hash", ""):
        return ""
    manifest_path = Path(catalog_path).parent / "catalog_manifest.json"
    try:
        return str(json.loads(manifest_path.read_text()).get("catalog_snapshot_id", "catalog"))
    except (OSError, ValueError):
        return "catalog"


def _contribution_families(sv: pd.DataFrame, run_metrics: pd.DataFrame, other: str,
                           subsets: dict[str, set[str] | None],
                           bootstrap_iters: int, bootstrap_seed: int) -> pd.DataFrame:
    """Δ table for one ablation over BOTH outcome families, Holm-corrected separately.

    :func:`~jobrec_eval.statistics.contribution_table` applies Holm across the metrics of
    a single call within each subset, so :data:`PRIMARY` and :data:`SECONDARY` are passed
    in two separate calls: each family is corrected over its own metrics only, and the
    pre-registered primary p-values are identical to what a primary-only run produces.
    The two frames are concatenated with a :data:`FAMILY_COLUMN` tag (``primary`` /
    ``secondary``) so the report can render them apart and nothing is double-counted
    (R7.4, R32.4).
    """
    frames = []
    for family, metrics in (("primary", PRIMARY), ("secondary", SECONDARY)):
        tbl = contribution_table(sv, run_metrics, metrics, other, subsets,
                                 bootstrap_iters, bootstrap_seed)
        frames.append(tbl.assign(**{FAMILY_COLUMN: family}))
    return pd.concat(frames, ignore_index=True)


def run_pipeline(config_path: str, scenarios_path: str, catalog_path: str,
                 out_root: str, repeats: int, experiment_dir: str | None,
                 bootstrap_iters: int, bootstrap_seed: int,
                 variants: list[str] | None = None,
                 relevance_source: str = "oracle",
                 allow_overwrite: bool = False) -> dict:
    """Run the evaluation end to end.

    ``allow_overwrite`` is the explicit opt-in for writing into an output (or run-bundle)
    directory that already holds a complete experiment; without it such a write raises
    :class:`~jobrec.evaluation.experiment_identity.ExperimentOverwriteError` instead of
    silently replacing the earlier artifact.

    ``relevance_source`` selects the label table behind every grade-derived metric
    (:data:`RELEVANCE_METRICS`): ``"oracle"`` (default) uses the automatic oracle,
    ``"human"`` uses the adjudicated human labels from ``relevance_labels_human.csv``
    beside ``scenarios_path`` and raises
    :class:`~jobrec_eval.annotation.MissingAdjudicatedLabelsError` when they are absent.
    Whenever human labels exist BOTH sources are computed and written side by side to
    ``metrics/relevance_source_comparison.csv`` (checklist item 10).
    """
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
        runner_manifest = runner.run(variants, allow_overwrite=allow_overwrite)
        exp_dir = Path(runner_manifest["experiment_dir"])

    experiment_id = runner_manifest["experiment_id"]
    catalog_snapshot_id = _catalog_snapshot_id(catalog_path, runner_manifest)
    out = Path(out_root) / experiment_id
    # The analysis directory is guarded the same way as the run-bundle directory: a
    # complete previous analysis (one with manifests/experiment_manifest.json) is never
    # replaced silently, whatever produced it.
    guard_output_dir(out, manifest_name=f"manifests/{EXPERIMENT_MANIFEST_FILENAME}",
                     allow_overwrite=allow_overwrite)
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

    # ---- relevance labels: automatic oracle + optional adjudicated human --
    # The oracle table is always produced (it is a run artifact and the retrieval-recall
    # label universe). An adjudicated human table, when present, is loaded into the SAME
    # shape, so the selected one is a drop-in substitute for the other.
    #
    # The reference the oracle grades against is CANONICAL: a frozen function of the
    # scenario file and the catalog, derived once under the deterministic ``full``
    # condition and reused by every experiment sharing those inputs. It used to be
    # lifted from this experiment's own first ``full`` bundle, which made the labels a
    # sample of the system's own output (and, with repeats > 1, of one arbitrary repeat).
    scen_dir = Path(scenarios_path).parent
    canonical = load_or_build_canonical_references(scenarios_path, catalog_path, cfg)
    references = canonical.references
    labels = grade_catalog(catalog, references, cfg)
    _write_csv(labels, out / "normalized" / "relevance_labels.csv")
    _write_json(canonical.as_artifact(), out / "manifests" / CANONICAL_ORACLE_FILENAME)

    human_labels_path = scen_dir / HUMAN_RELEVANCE_FILENAME
    human = load_adjudicated_relevance_labels(human_labels_path)
    primary_labels, recorded_source = select_relevance_labels(relevance_source, labels, human)
    if human is not None:
        _write_csv(human.labels, out / "normalized" / "relevance_labels_human.csv")

    # ---- Stage 4: metrics ----------------------------------------------
    mc = MetricsComputer(cfg, catalog, references, primary_labels, scenarios,
                         relevance_threshold=2, top_k=cfg.experiment.top_k)
    run_metrics = mc.run_metrics(bundles)
    sv = aggregate_scenario_variant(run_metrics)
    vsum = variant_summary(sv)
    latpct = latency_percentiles(tables["component_latency"])
    _write_csv(run_metrics, out / "metrics" / "run_metrics.csv")
    _write_csv(sv, out / "metrics" / "scenario_variant_metrics.csv")
    _write_csv(vsum, out / "metrics" / "variant_summary.csv")
    _write_csv(latpct, out / "metrics" / "latency_metrics.csv")

    # ---- oracle vs human relevance, side by side (checklist item 10) --------
    # The selected source's summary is reused; the other one is computed through the same
    # MetricsComputer with the other label table. With no human labels the comparison holds
    # the oracle column only and leaves the human column empty (never imputed).
    oracle_vsum = vsum if recorded_source == RELEVANCE_SOURCE_ORACLE else _ranking_variant_summary(
        cfg, catalog, references, labels, scenarios, bundles, cfg.experiment.top_k)
    human_vsum = vsum if recorded_source == RELEVANCE_SOURCE_HUMAN else (
        _ranking_variant_summary(cfg, catalog, references, human.labels, scenarios, bundles,
                                 cfg.experiment.top_k) if human is not None else None)
    source_comparison = relevance_source_comparison(oracle_vsum, human_vsum)
    _write_csv(source_comparison, out / "metrics" / "relevance_source_comparison.csv")

    # scenario type summary -- the process measures (response_turns and
    # clarification_efficiency) are reported here too, so clarification efficiency reaches
    # per-run metrics, the variant summary, the scenario-type summary and the report (R7.4).
    type_summary = (sv.groupby(["scenario_type", "variant"])[
        ["ndcg_at_5", "hcsr", "task_success", "grounding", "response_turns",
         "clarification_efficiency"]].mean().reset_index())
    _write_csv(type_summary, out / "metrics" / "scenario_type_summary.csv")

    # ---- Stage 5: statistics -------------------------------------------
    mem_dep = {s.scenario_id for s in scenarios.values() if s.memory_dependency in ("medium", "high")}
    ctx_dep = {s.scenario_id for s in scenarios.values() if s.context_dependency == "high"}

    mem_tbl = _contribution_families(sv, run_metrics, "no_memory",
                                     {"all": None, "memory_dependent": mem_dep},
                                     bootstrap_iters, bootstrap_seed)
    ctx_tbl = _contribution_families(sv, run_metrics, "no_context",
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
    # Clarification efficiency per variant, plus the response-turn distribution the report
    # reads its median/IQR from. The per-run column already lives in run_metrics.csv and the
    # scenario/variant aggregates carry it, so this is the missing per-variant view (R7.4).
    clarify_efficiency = clarification_efficiency(run_metrics)
    # The four R10 failure-path rates as one tidy table; reported in the SEPARATE
    # fault-injection section of the report, never mixed into the main results (R10.8/10.9).
    failures_tbl = failure_metrics(run_metrics, bundles)
    extraction_sources = extraction_source_metrics(bundles, scenarios)
    # Retrieval recall stays on the ORACLE labels in both modes -- see
    # :data:`RETRIEVAL_LABEL_SOURCE` for why (full-catalog label universe).
    retrieval = retrieval_metrics(bundles, labels, relevance_threshold=2)
    topk_contribution = topk_contribution_table(bundles, top_k=cfg.experiment.top_k)
    _write_csv(compliance, out / "metrics" / "constraint_compliance.csv")
    _write_csv(nomatch, out / "metrics" / "no_match_metrics.csv")
    _write_csv(clarify, out / "metrics" / "clarification_metrics.csv")
    _write_csv(clarify_efficiency, out / "metrics" / "clarification_efficiency.csv")
    _write_csv(failures_tbl, out / "metrics" / "failure_metrics.csv")
    _write_csv(extraction_sources, out / "metrics" / "extraction_source_metrics.csv")
    _write_csv(retrieval, out / "metrics" / "retrieval_metrics.csv")
    _write_csv(topk_contribution, out / "metrics" / "topk_contribution.csv")

    # ---- data quality of the frozen inputs (R17.3, checklist item 14) --------
    # Saved into the experiment artifacts, not just by the standalone `validate`
    # subcommand, and written BEFORE `write_checksums(out)` so the manifest covers it.
    # Findings are reported, never fatal: `validate` remains the gate that exits non-zero.
    # ``verify_no_match`` stays ON: it is the only check that distinguishes a genuinely
    # infeasible no-match scenario from one that merely fails on role fit, which is exactly
    # what the final report has to state. It replays only the no-match subset, with the rule
    # extractor and no database, so the added cost is bounded and deterministic.
    dq_report = validate_dataset(
        catalog, scenarios, config=cfg, relevance_labels=labels,
        verify_no_match=True, catalog_snapshot_id=catalog_snapshot_id or "catalog",
    )
    write_data_quality_report(dq_report, out)
    dq_payload = dq_report.to_dict()

    # ---- case studies + error taxonomy ---------------------------------
    # Graded from the SELECTED label table, so the grades shown beside a case match the
    # ranking metrics reported for that run.
    grade = grade_lookup(primary_labels)
    cases = extract_cases(bundles, run_metrics, grade, llm_mode=cfg.llm.mode.value)
    cases_md = render_cases_md(cases)
    err_tax = error_taxonomy(run_metrics)
    _write_csv(err_tax, out / "metrics" / "error_taxonomy.csv")

    # ---- annotation templates + human agreement (if provided) ----------
    ann_dir = out / "annotation"
    ann_dir.mkdir(parents=True, exist_ok=True)
    export_relevance_template(tables["recommendations"], labels, ann_dir / "relevance_template.csv")
    export_claim_template(tables["claims"], ann_dir / "claim_template.csv")
    rel_agree = relevance_agreement(human_labels_path, labels)
    clm_agree = claim_agreement(scen_dir / HUMAN_CLAIMS_FILENAME)

    # ---- Stage 6: plots -------------------------------------------------
    plot_all(sv, latpct, out / "plots")

    # ---- error summary --------------------------------------------------
    failures = run_metrics[~run_metrics["success_run"]]
    task_fail = run_metrics[run_metrics["task_success"] == 0]
    # Rendered as prose, not as a Python dict: interpolating the mapping directly put
    # ``{'no_context': 35, ...}`` -- braces and quotes included -- into the report.
    by_variant = task_fail.groupby("variant").size().to_dict()
    by_variant_text = (", ".join(f"{v} {n}" for v, n in sorted(by_variant.items()))
                       or "none")
    error_summary = (f"Runs: {len(run_metrics)}; system failures: {len(failures)}; "
                     f"task-unsuccessful runs: {len(task_fail)}. "
                     f"Task failures by variant: {by_variant_text}.")

    # ---- assemble report data ------------------------------------------
    scenario_type_counts = {t: int((pd.Series([s.scenario_type for s in scenarios.values()]) == t).sum())
                            for t in sorted({s.scenario_type for s in scenarios.values()})}
    data = {
        "experiment": {
            "experiment_id": experiment_id,
            "reference_date": cfg.project.reference_date,
            "catalog_snapshot_id": catalog_snapshot_id,
            "catalog_hash": catalog_hash(catalog),
            "variants": variants,
            "scenario_count": len(scenarios),
            "repeat_count": repeats,
            "run_count": len(run_metrics),
            # Carried from the runner so an incomplete experiment cannot be reported as a
            # complete one: a run that crashed produces no bundle, so it is missing from
            # every table below without anything else saying so.
            "expected_run_count": runner_manifest.get("expected_run_count"),
            "crashed_run_count": runner_manifest.get("crashed_run_count") or 0,
            "bootstrap_iterations": bootstrap_iters,
            "bootstrap_seed": bootstrap_seed,
            "eval_version": EVAL_VERSION,
        },
        "oracle_version": ORACLE_VERSION,
        # What actually produced the ranking metrics, so every section that presents them
        # can say so and no section can claim "no human raters" when there were some.
        "relevance_source": {
            "selected": recorded_source,
            "flag": relevance_source,
            "oracle_version": ORACLE_VERSION,
            "oracle_table": "normalized/relevance_labels.csv",
            # The canonical reference behind the grades: which inputs it is a function
            # of, and the fingerprints that let a reader confirm two experiments were
            # graded on the same yardstick.
            "canonical_reference": {
                "version": CANONICAL_ORACLE_VERSION,
                "inputs_fingerprint": canonical.inputs_fingerprint,
                "reference_fingerprint": canonical.reference_fingerprint,
                "artifact": f"manifests/{CANONICAL_ORACLE_FILENAME}",
                **{k: canonical.provenance.get(k) for k in (
                    "derivation", "variant", "repeats", "llm_mode", "scenario_count",
                    "referenced_scenario_count", "scenarios_without_reference",
                    "declared_scenario_count", "system_derived_scenario_count",
                    "system_derived_scenarios")},
            },
            "human_labels_available": human is not None,
            "human_labels": (human.provenance if human is not None else None),
            "human_label_search_path": str(human_labels_path),
            "human_table": ("normalized/relevance_labels_human.csv"
                            if human is not None else None),
            "retrieval_labels": RETRIEVAL_LABEL_SOURCE,
            "comparison_table": "metrics/relevance_source_comparison.csv",
            "metrics_from_source": RELEVANCE_METRICS,
        },
        "relevance_source_comparison": source_comparison.to_dict(orient="records"),
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
        "clarification_efficiency": clarify_efficiency.to_dict(orient="records"),
        "failure_metrics": failures_tbl.to_dict(orient="records"),
        # Counts only: the findings themselves live in data_quality_report.json, so the
        # report data does not carry a second copy of them.
        "data_quality": {k: dq_payload[k] for k in (
            "reference_date", "job_count", "scenario_count", "ok", "error_count",
            "warning_count", "info_count", "checks_run", "checks_skipped",
            "counts_by_violation_type")},
        "extraction_source_metrics": extraction_sources.to_dict(orient="records"),
        "retrieval_metrics": retrieval.to_dict(orient="records"),
        "topk_contribution": topk_contribution.to_dict(orient="records"),
        "error_taxonomy": err_tax.to_dict(orient="records"),
        "case_studies_md": cases_md,
        "relevance_agreement": rel_agree,
        "claim_agreement": clm_agree,
        "llm_mode": cfg.llm.mode.value,
        # The PROVIDER is what the config selects (the transport); the MODEL is only
        # knowable from the recorded calls, because it comes from the environment. These
        # used to be one field, so a hybrid run was reported as using model "remote".
        "llm_provider": cfg.llm.provider,
        "model_call_identities": model_call_identities(bundles),
    }

    # ---- manifests ------------------------------------------------------
    # Carries the runner's code identity (commit_hash / code_version / git_dirty /
    # source_fingerprint) forward, plus the identity of the code that did THIS analysis --
    # they differ when --experiment-dir reuses bundles produced by an earlier version.
    (out / "manifests" / "experiment_manifest.json").write_text(json.dumps({
        **runner_manifest, "eval_version": EVAL_VERSION, "oracle_version": ORACLE_VERSION,
        "catalog_hash": catalog_hash(catalog),
        "analysis_code_identity": code_identity(),
    }, indent=2))
    # Both outcome families are recorded, so the plan documents that Holm is applied
    # within each family independently and the pre-registered primary family is unchanged.
    (out / "manifests" / "analysis_plan.yaml").write_text(yaml.safe_dump({
        "primary_outcomes": PRIMARY,
        "secondary_outcomes": SECONDARY,
        "primary_comparisons": [["full", v] for v in ["profile_only", "one_shot", "no_memory", "no_context"]],
        "confidence_level": 0.95, "bootstrap_iterations": bootstrap_iters,
        "bootstrap_seed": bootstrap_seed, "binary_relevance_threshold": 2,
        "p_value_adjustment": "holm",
        "p_value_adjustment_scope": "within each outcome family, ablation and subset",
        "outcome_families": ["primary", "secondary"],
        "analysis_unit": "scenario_id",
        "repeat_index_role": "stability and variance analysis only, never an independent sample",
        "binary_task_success_pairing": "scenario-level (repeats collapsed by majority vote; even-repeat ties -> 0)",
        # Recorded from what the run actually used, plus the provenance of the human label
        # file behind it, so a reader can tell WHICH labels produced the numbers.
        "relevance_source": recorded_source,
        "relevance_source_flag": relevance_source,
        "relevance_metrics_from_source": RELEVANCE_METRICS,
        "relevance_oracle_version": ORACLE_VERSION,
        # The reference the oracle grades against is a frozen function of the scenario
        # file and the catalog -- NOT of the experiment being analysed. Recorded here so
        # the plan states the estimand's yardstick rather than leaving it implicit.
        "canonical_reference_version": CANONICAL_ORACLE_VERSION,
        "canonical_reference_derivation": canonical.provenance.get("derivation"),
        "canonical_reference_condition": {
            "variant": canonical.provenance.get("variant"),
            "repeats": canonical.provenance.get("repeats"),
            "llm_mode": canonical.provenance.get("llm_mode"),
        },
        "canonical_reference_inputs_fingerprint": canonical.inputs_fingerprint,
        "canonical_reference_fingerprint": canonical.reference_fingerprint,
        "canonical_reference_artifact": f"manifests/{CANONICAL_ORACLE_FILENAME}",
        "retrieval_recall_relevance_source": RETRIEVAL_LABEL_SOURCE,
        "human_relevance_labels": (human.provenance if human is not None else None),
        "relevance_source_comparison": "metrics/relevance_source_comparison.csv",
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
        {"report_section": "3.1", "metric_name": "data_quality", "source_file": "data_quality_report.json"},
        {"report_section": "5", "metric_name": "variant_summary", "source_file": "metrics/variant_summary.csv"},
        {"report_section": "5.4", "metric_name": "clarification_efficiency", "source_file": "metrics/clarification_efficiency.csv"},
        {"report_section": "6.1", "metric_name": "memory_contribution", "source_file": "metrics/memory_contribution.csv"},
        {"report_section": "6.2", "metric_name": "context_contribution", "source_file": "metrics/context_contribution.csv"},
        {"report_section": "6.3", "metric_name": "secondary_contribution", "source_file": "metrics/memory_contribution.csv"},
        {"report_section": "7", "metric_name": "scenario_type_summary", "source_file": "metrics/scenario_type_summary.csv"},
        {"report_section": "8", "metric_name": "paired_comparisons", "source_file": "statistics/paired_comparisons.csv"},
        {"report_section": "5.5", "metric_name": "relevance_source_comparison", "source_file": "metrics/relevance_source_comparison.csv"},
        # The three pipeline-stage tables. They were written on every run but rendered
        # nowhere, so they had no report section to be traced from either.
        {"report_section": "5.6", "metric_name": "retrieval_metrics", "source_file": "metrics/retrieval_metrics.csv"},
        {"report_section": "5.6", "metric_name": "topk_contribution", "source_file": "metrics/topk_contribution.csv"},
        {"report_section": "5.6", "metric_name": "extraction_source_metrics", "source_file": "metrics/extraction_source_metrics.csv"},
        {"report_section": "9", "metric_name": "error_taxonomy", "source_file": "metrics/error_taxonomy.csv"},
        {"report_section": "9.x", "metric_name": "relevance_labels", "source_file": "normalized/relevance_labels.csv"},
        {"report_section": "10.1", "metric_name": "failure_metrics", "source_file": "metrics/failure_metrics.csv"},
        # The frozen canonical reference every grade-derived metric is measured against.
        {"report_section": "5", "metric_name": "canonical_reference",
         "source_file": f"manifests/{CANONICAL_ORACLE_FILENAME}"},
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


def run_replay(experiment_dir: str, *, catalog_path: str | None = None,
               out_path: str | None = None) -> int:
    """Replay every run bundle under ``experiment_dir`` and write the diff report.

    Returns 0 when every run replayed and reproduced identical key states, 1 otherwise.

    :mod:`jobrec.evaluation.replay_check` had no entry point outside the test suite, so
    the reproducibility claim it backs ("N/N runs replayed, 0 differences") could not be
    reproduced by anyone reading the thesis -- the only way to obtain it was to write
    Python. This is that command.
    """
    from jobrec.evaluation.replay_check import write_replay_diff

    root = Path(experiment_dir)
    if not root.is_dir():
        print(f"FAIL: {root} is not a directory")
        return 2
    report_path = write_replay_diff(root, catalog_path=catalog_path, out_path=out_path)
    payload = json.loads(Path(report_path).read_text(encoding="utf-8"))
    print(json.dumps({
        "root": payload.get("root"),
        "run_count": payload.get("run_count"),
        "replayed_count": payload.get("replayed_count"),
        "identical": payload.get("identical"),
        "difference_count": payload.get("difference_count"),
        "report": str(report_path),
    }, indent=2))
    if payload.get("identical"):
        return 0
    print(f"FAIL: {payload.get('difference_count')} key-state difference(s); "
          f"{payload.get('run_count', 0) - payload.get('replayed_count', 0)} run(s) "
          f"could not be replayed. See {report_path}")
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
    p.add_argument("--allow-overwrite", action="store_true",
                   help=("overwrite an existing complete experiment in --out-root (and its "
                         "run bundles) on purpose. Without this flag the run refuses to "
                         "replace an existing artifact instead of clobbering it silently"))
    p.add_argument("--relevance-source", choices=sorted(RELEVANCE_SOURCES), default="oracle",
                   help=(f"label table behind NDCG@5 / P@5 / mean graded relevance: "
                         f"'oracle' (default, the automatic oracle) or 'human' (the "
                         f"adjudicated labels in {HUMAN_RELEVANCE_FILENAME} beside "
                         f"--scenarios). 'human' fails the run when no adjudicated labels "
                         f"are available instead of falling back to the oracle. Both "
                         f"sources are always compared in "
                         f"metrics/relevance_source_comparison.csv"))

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

    rep = sub.add_parser("replay", help="replay run bundles and diff their key states (R18)")
    rep.add_argument("experiment_dir", help="run-bundle directory to replay")
    rep.add_argument("--catalog", dest="catalog_path", default=None,
                     help="catalog to replay against (default: the snapshot in the tree)")
    rep.add_argument("--out", dest="out_path", default=None,
                     help="where to write the diff report (default: beside the bundles)")

    args = parser.parse_args()
    if args.command == "pipeline":
        variants = args.variants.split(",") if args.variants else None
        try:
            result = run_pipeline(args.config, args.scenarios, args.catalog, args.out_root,
                                  args.repeats, args.experiment_dir, args.bootstrap_iters,
                                  args.bootstrap_seed, variants=variants,
                                  relevance_source=args.relevance_source,
                                  allow_overwrite=args.allow_overwrite)
        except MissingAdjudicatedLabelsError as exc:
            # Loud, non-zero exit: no oracle numbers under a human-labels heading.
            raise SystemExit(f"FAIL: {exc}") from exc
        except ExperimentOverwriteError as exc:
            # Loud, non-zero exit: an existing official artifact is never replaced silently.
            raise SystemExit(f"FAIL: {exc}") from exc
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
    elif args.command == "replay":
        code = run_replay(args.experiment_dir, catalog_path=args.catalog_path,
                          out_path=args.out_path)
        if code != 0:
            raise SystemExit(code)


if __name__ == "__main__":
    main()
