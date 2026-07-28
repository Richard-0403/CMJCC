"""End-to-end pipeline artifacts for the process, failure-path and data-quality wiring.

``clarification_efficiency`` and the four R10 failure-path rates were defined but never
called from ``src/``, so no test could have caught them missing from the pipeline. This
module runs the real deterministic pipeline once over the small CI fixture scenario set and
asserts what actually landed on disk:

- ``metrics/clarification_efficiency.csv`` exists, carries the response-turn distribution,
  and the same numbers reach the report data and the rendered Markdown (checklist item 3).
- ``metrics/failure_metrics.csv`` exists with one tidy row per rate, and the report reads
  the injection rates as N/A over a main-experiment run set (checklist item 5).
- both contribution CSVs carry a ``family`` column, and the PRIMARY rows are byte-identical
  to a primary-only ``contribution_table`` call, i.e. adding the secondary family did not
  make one pre-registered p-value more conservative (checklist item 6).
- ``manifests/analysis_plan.yaml`` records both outcome families.
- ``data_quality_report.json`` is written into the experiment artifacts and covered by
  ``checksums.json`` (checklist item 14).
- both artifact trees a complete run leaves behind -- the run bundles under
  ``_runs/<experiment_id>/`` and this analysis directory -- verify clean against their
  own ``checksums.json`` (R16.2).
- ``manifests/experiment_manifest.json`` records the code identity behind the numbers, so
  two artifacts from different source code can be told apart offline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from jobrec.evaluation.experiment_identity import CODE_IDENTITY_FIELDS, code_identity
from jobrec_eval.cli import PRIMARY, SECONDARY, run_pipeline, run_verify
from jobrec_eval.metrics_extra import clarification_efficiency
from jobrec_eval.statistics import contribution_table

CONFIG = "configs/experiment_full.yaml"
SCENARIOS = "evaluation/data/scenarios_subset.jsonl"
CATALOG = "data/processed/jobs.jsonl"
VARIANTS = ["full", "no_memory", "no_context"]


@pytest.fixture(scope="module")
def pipeline_out(tmp_path_factory) -> Path:
    """Run the deterministic pipeline once into a temporary output root.

    The scenario file is COPIED into ``tmp_path`` first. The pipeline freezes the
    canonical relevance reference beside the scenario file it was given, so pointing it
    at the repository's file would have this test write a derived artifact into
    ``evaluation/data/`` on every run.
    """
    out_root = tmp_path_factory.mktemp("pipeline-artifacts")
    scenarios = out_root / "data" / Path(SCENARIOS).name
    scenarios.parent.mkdir(parents=True, exist_ok=True)
    scenarios.write_text(Path(SCENARIOS).read_text(encoding="utf-8"), encoding="utf-8")
    result = run_pipeline(CONFIG, str(scenarios), CATALOG, str(out_root), 1, None,
                          200, 2026, variants=VARIANTS)
    return Path(result["out_dir"])


@pytest.fixture(scope="module")
def report_md(pipeline_out: Path) -> str:
    return (pipeline_out / "report" / "analysis_report.md").read_text(encoding="utf-8")


@pytest.fixture(scope="module")
def report_data(pipeline_out: Path) -> dict:
    return json.loads((pipeline_out / "report" / "analysis_report_data.json").read_text(
        encoding="utf-8"))


# ------------------------------------------------- checklist item 3: clarification efficiency
def test_clarification_efficiency_csv_is_written_with_the_turn_distribution(pipeline_out):
    table = pd.read_csv(pipeline_out / "metrics" / "clarification_efficiency.csv")

    # ``asked_unresolved`` sits next to ``efficiency_score``: the score now carries an
    # unresolved-dialogue penalty, so the count of abandoned dialogues has to be readable
    # beside it (R7.4/R7.5).
    assert list(table.columns) == [
        "variant", "runs", "necessary_asked", "necessary_missed", "unnecessary_asked",
        "asked_unresolved", "efficiency_score", "response_turns_n",
        "median_response_turns", "q1_response_turns", "q3_response_turns",
        "iqr_response_turns"]
    assert set(table["variant"]) == set(VARIANTS)
    # The deterministic runs all carry a dialogue trace, so the distribution is populated.
    assert (table["response_turns_n"] > 0).all()
    assert table["median_response_turns"].notna().all()
    assert table["iqr_response_turns"].notna().all()


def test_clarification_efficiency_csv_matches_the_per_run_column(pipeline_out):
    """The CSV is the aggregate of the per-run column, not an independent computation."""
    run_metrics = pd.read_csv(pipeline_out / "metrics" / "run_metrics.csv")
    assert "clarification_efficiency" in run_metrics.columns
    assert "response_turns" in run_metrics.columns

    written = pd.read_csv(pipeline_out / "metrics" / "clarification_efficiency.csv"
                          ).set_index("variant")
    recomputed = clarification_efficiency(run_metrics).set_index("variant")
    for variant in VARIANTS:
        assert (written.loc[variant, "efficiency_score"]
                == pytest.approx(recomputed.loc[variant, "efficiency_score"]))
        assert (written.loc[variant, "median_response_turns"]
                == pytest.approx(run_metrics[run_metrics["variant"] == variant]
                                 ["response_turns"].median()))


def test_clarification_efficiency_reaches_the_report_data_and_markdown(report_data, report_md):
    rows = report_data["clarification_efficiency"]
    assert {r["variant"] for r in rows} == set(VARIANTS)

    assert "### 5.4 Clarification efficiency and response-turn distribution" in report_md
    assert "jointly with task success" in report_md
    # The rendered score for the full variant is the one in the data dict.
    full = next(r for r in rows if r["variant"] == "full")
    assert f"| full | {full['median_response_turns']:.2f} |" in report_md


def test_process_measures_reach_the_variant_and_scenario_type_summaries(pipeline_out):
    variant_summary = pd.read_csv(pipeline_out / "metrics" / "variant_summary.csv")
    for column in ("response_turns_mean", "response_turns_median",
                   "clarification_efficiency_mean", "clarification_efficiency_median"):
        assert column in variant_summary.columns
        assert variant_summary[column].notna().all()

    type_summary = pd.read_csv(pipeline_out / "metrics" / "scenario_type_summary.csv")
    assert list(type_summary.columns) == [
        "scenario_type", "variant", "ndcg_at_5", "hcsr", "task_success", "grounding",
        "response_turns", "clarification_efficiency"]
    assert type_summary["clarification_efficiency"].notna().all()


# ------------------------------------------------------ checklist item 5: failure metrics
def test_failure_metrics_csv_is_written_in_tidy_long_form(pipeline_out):
    table = pd.read_csv(pipeline_out / "metrics" / "failure_metrics.csv")

    assert list(table.columns) == ["metric", "source", "numerator", "denominator", "value"]
    assert list(table["metric"]) == ["failure_detection_rate", "recovery_success_rate",
                                     "grounding_rate", "handoff_success_rate"]
    by_metric = table.set_index("metric")
    # A main-experiment run set injects nothing: empty denominator, blank value cell.
    assert by_metric.loc["failure_detection_rate", "denominator"] == 0
    assert pd.isna(by_metric.loc["failure_detection_rate", "value"])
    assert pd.isna(by_metric.loc["recovery_success_rate", "value"])
    # Grounding and handoffs are measured over real artifacts.
    assert by_metric.loc["grounding_rate", "denominator"] > 0
    assert by_metric.loc["handoff_success_rate", "denominator"] > 0


def test_fault_injection_section_reads_na_and_defends_a_grounding_of_one(report_md):
    assert "## 10. Fault-Injection Robustness (separate robustness experiment)" in report_md
    squeezed = " ".join(report_md.split())

    assert ("injected no faults, so the detection and recovery rates read N/A, not 0.0 and "
            "not 1.000") in squeezed
    assert "grounding rate of 1.000 in the main experiment is legitimate, not a defect" in squeezed
    assert "Failure samples are NOT mixed into the main experiment" in squeezed
    # The injection rows carry no fabricated number.
    for rate in ("failure_detection_rate", "recovery_success_rate"):
        row = next(line for line in report_md.splitlines() if line.startswith(f"| {rate} "))
        assert "N/A (empty denominator)" in row


# -------------------------------------------------- checklist item 6: two outcome families
@pytest.mark.parametrize(("csv_name", "other", "subsets"),
                         [("memory_contribution.csv", "no_memory", {"all", "memory_dependent"}),
                          ("context_contribution.csv", "no_context", {"all", "context_dependent"})])
def test_contribution_csv_carries_both_families_without_double_counting(
        pipeline_out, csv_name, other, subsets):
    table = pd.read_csv(pipeline_out / "metrics" / csv_name)

    assert "family" in table.columns
    assert set(table["family"]) == {"primary", "secondary"}
    assert set(table["subset"]) == subsets
    assert set(table[table["family"] == "primary"]["metric"]) == set(PRIMARY)
    assert set(table[table["family"] == "secondary"]["metric"]) == set(SECONDARY)
    # One row per (family, subset, metric): nothing is counted twice.
    assert not table.duplicated(subset=["family", "subset", "metric"]).any()
    assert len(table) == len(subsets) * (len(PRIMARY) + len(SECONDARY))
    assert set(table["other"]) == {other}


def test_secondary_family_leaves_every_primary_p_value_untouched(pipeline_out):
    """Holm is applied per family, so the primary rows equal a primary-only computation.

    This is the methodological guard behind checklist item 6: appending the process
    measures to ``PRIMARY`` would have inflated every primary ``p_value_holm`` (Holm over
    8 metrics instead of 6) and silently changed what the pre-registered
    ``primary_outcomes`` means.
    """
    sv = pd.read_csv(pipeline_out / "metrics" / "scenario_variant_metrics.csv")
    run_metrics = pd.read_csv(pipeline_out / "metrics" / "run_metrics.csv")
    mem_dep = set(sv[sv["memory_dependency"].isin(["medium", "high"])]["scenario_id"])

    expected = contribution_table(sv, run_metrics, PRIMARY, "no_memory",
                                  {"all": None, "memory_dependent": mem_dep}, 200, 2026)
    written = pd.read_csv(pipeline_out / "metrics" / "memory_contribution.csv")
    written = written[written["family"] == "primary"]

    key = ["subset", "metric"]
    compare_cols = ["delta", "p_value", "p_value_holm", "n_pairs"]
    left = expected.set_index(key)[compare_cols].sort_index()
    right = written.set_index(key)[compare_cols].sort_index()
    pd.testing.assert_frame_equal(left, right, check_dtype=False, atol=1e-12)


def test_report_renders_the_secondary_family_in_its_own_sub_table(report_md):
    assert "### 6.3 Process-measure contributions (secondary outcome family)" in report_md
    head, _, secondary_section = report_md.partition("### 6.3")
    ablation = head.partition("## 6. Ablation Analysis")[2]

    # The process measures appear only in 6.3, never in the 6.1/6.2 tables.
    for metric in SECONDARY:
        assert f"| {metric} |" in secondary_section
        assert f"| {metric} |" not in ablation
    # All four sub-tables (memory/context x dependent-subset/all).
    assert secondary_section.count("| metric | full | other |") == 4


def test_analysis_plan_records_both_outcome_families(pipeline_out):
    plan = yaml.safe_load((pipeline_out / "manifests" / "analysis_plan.yaml").read_text())

    assert plan["primary_outcomes"] == PRIMARY
    assert plan["secondary_outcomes"] == SECONDARY
    assert plan["p_value_adjustment"] == "holm"
    assert "within each outcome family" in plan["p_value_adjustment_scope"]
    assert plan["analysis_unit"] == "scenario_id"
    assert "scenario-level" in plan["binary_task_success_pairing"]


# --------------------------------------------- checklist item 4: scenario-level wording
def test_report_section_8_states_scenario_level_pairing(report_md):
    assert "run-level discordant pairs" not in report_md
    squeezed = " ".join(report_md.split())
    assert "scenario-level paired binary outcomes" in squeezed
    assert "number of **validly paired scenarios**, not the number of runs" in squeezed


def test_report_shows_the_pairing_provenance_from_the_comparison_rows(report_md, report_data):
    """n_pairs is visibly a scenario count: the table prints scenarios, runs and repeats."""
    row = next(r for r in report_data["overall_comparisons"]
               if r["metric"] == "task_success" and r["other"] == "no_memory")
    assert row["valid_pairs"] == row["n_pairs"] == row["scenario_count"]
    # total_run_count spans both sides of the comparison, so it is strictly larger than
    # n_pairs -- which is exactly what makes the scenario-level pairing visible.
    assert row["total_run_count"] == row["scenario_count"] * row["repeats_per_scenario"] * 2
    assert row["n_pairs"] < row["total_run_count"]

    line = next(line for line in report_md.splitlines()
                if line.startswith("| full vs no_memory |"))
    cells = [c.strip() for c in line.strip("|").split("|")]
    assert cells[1:6] == [str(row["scenario_count"]), str(row["total_run_count"]),
                          str(row["repeats_per_scenario"]), str(row["valid_pairs"]),
                          str(row["discordant_pairs"])]


# ------------------------------- report header: the run-bundle pointer must be real
def test_report_header_points_at_a_run_bundle_directory_that_exists(pipeline_out, report_md):
    """The reproducibility pointer resolves to real bundles, not to a fictional ``raw/``.

    The header used to promise the numbers were reproducible from ``raw/``, which the
    pipeline never creates. Bundles live in ``<out_root>/_runs/<experiment_id>/<variant>/
    <scenario_id>/<run_index>/``, a sibling of this analysis directory, so this test walks
    the pointer from the header down to a real ``run_manifest.json``.
    """
    experiment_id = pipeline_out.name
    pointer = f"_runs/{experiment_id}/<variant>/<scenario_id>/<run_index>/"

    assert f"`{pointer}`" in report_md
    assert "`raw/`" not in report_md
    assert not (pipeline_out / "raw").exists()

    # Resolve the pointer relative to the analysis directory's parent (the --out-root).
    resolved = pointer.replace("<variant>", VARIANTS[0]).rstrip("/")
    variant_dir = pipeline_out.parent / Path(*resolved.split("/")[:3])
    assert variant_dir.is_dir(), sorted(p.name for p in pipeline_out.parent.iterdir())

    scenario_dirs = sorted(p for p in variant_dir.iterdir() if p.is_dir())
    assert scenario_dirs
    run_dirs = sorted(p for p in scenario_dirs[0].iterdir() if p.is_dir())
    assert [p.name for p in run_dirs] == ["0"]
    assert (run_dirs[0] / "run_manifest.json").is_file()


# ------------------------------------- checklist item 14: data quality in the artifacts
def test_data_quality_report_is_saved_into_the_experiment_artifacts(pipeline_out):
    payload = json.loads((pipeline_out / "data_quality_report.json").read_text())

    assert payload["job_count"] > 0
    assert payload["scenario_count"] > 0
    for key in ("error_count", "warning_count", "info_count", "checks_run"):
        assert key in payload
    # verify_no_match is enabled in the pipeline, so the joint-infeasibility check ran.
    assert "no_match_scenarios_unsatisfiable" in payload["checks_run"]


def test_data_quality_report_is_covered_by_the_checksum_manifest(pipeline_out):
    checksums = json.loads((pipeline_out / "checksums.json").read_text())
    files = checksums.get("files", checksums)
    for artifact in ("data_quality_report.json", "metrics/clarification_efficiency.csv",
                     "metrics/failure_metrics.csv"):
        assert artifact in files, artifact


def test_pipeline_does_not_fail_on_data_quality_warnings(pipeline_out, report_data):
    """Warnings are surfaced, not fatal: the report exists and carries the counts."""
    dq = report_data["data_quality"]
    assert (pipeline_out / "report" / "analysis_report.md").is_file()
    assert dq["error_count"] == 0
    assert dq["warning_count"] >= 0
    assert "error(s)" in _dq_line(pipeline_out)


def _dq_line(pipeline_out: Path) -> str:
    md = (pipeline_out / "report" / "analysis_report.md").read_text(encoding="utf-8")
    return next(line for line in md.splitlines() if line.startswith("- Validated "))


def test_data_quality_counts_appear_in_the_report(pipeline_out, report_data):
    dq = report_data["data_quality"]
    line = _dq_line(pipeline_out)
    assert f"{dq['job_count']} catalog job(s)" in line
    assert f"{dq['scenario_count']} scenario(s)" in line
    assert (f"**{dq['error_count']} error(s), {dq['warning_count']} warning(s), "
            f"{dq['info_count']} acknowledged test fixture(s)**") in line


# ------------------------- R16: both artifact trees verify after a complete run
def test_run_bundle_checksums_verify_after_a_complete_pipeline_run(pipeline_out: Path):
    """`verify <out_root>/_runs/<experiment_id>` returns 0 after the full pipeline (R16.2).

    The runner writes the run-bundle manifest at the end of its own stage, but the
    later analysis stage stamps the consistency block into every
    ``run_manifest.json`` and mirrors it onto every ``run_record.json`` (R15.3).
    Those writes used to land after the digests were recorded, so the manifest was
    stale for two artifacts per bundle and this check could never pass.
    """
    runs_dir = pipeline_out.parent / "_runs" / pipeline_out.name
    assert (runs_dir / "checksums.json").is_file()
    # The consistency block really is on disk, i.e. the rewrite did happen.
    manifest_path = next(runs_dir.rglob("run_manifest.json"))
    assert "consistency" in json.loads(manifest_path.read_text())
    assert "consistency_flags" in json.loads(
        (manifest_path.parent / "run_record.json").read_text())

    assert run_verify(str(runs_dir)) == 0


def test_analysis_output_checksums_verify_after_a_complete_pipeline_run(pipeline_out: Path):
    """`verify <out_root>/<experiment_id>` still returns 0 (R16.2)."""
    assert run_verify(str(pipeline_out)) == 0


# --------- code identity of the artifact (reproducible-freeze provenance)
def test_analysis_manifest_records_the_code_identity_of_the_run(pipeline_out: Path):
    """``manifests/experiment_manifest.json`` tells two artifacts apart offline.

    The experiment id is derived from the code identity, so the directory name and the
    recorded identity must agree, and the manifest must additionally carry the commit hash,
    the dirty-tree flag and the identity of the code that ran the ANALYSIS (which differs
    from the runner's when ``--experiment-dir`` reuses older bundles).
    """
    manifest = json.loads(
        (pipeline_out / "manifests" / "experiment_manifest.json").read_text(encoding="utf-8"))
    identity = code_identity()

    assert manifest["experiment_id"] == pipeline_out.name
    for field in CODE_IDENTITY_FIELDS:
        assert manifest[field] == identity[field], field
    assert manifest["analysis_code_identity"] == identity
